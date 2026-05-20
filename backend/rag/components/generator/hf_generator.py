from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import List, Dict

import httpx

from backend.models.enums.generator_prompts import RAGGeneratorPrompt
from backend.core import get_logger
from .generator_base import AnswerGeneratorBase

logger = get_logger(__name__)


class HuggingFaceGenerator(AnswerGeneratorBase):
    """
    Answer generator using HuggingFace Inference API (OpenAI-compatible endpoint).
    Uses persistent AsyncClient for connection pooling.
    Supports meta-llama/Llama-3.1-8B-Instruct and other chat models.
    """

    def __init__(self, model: str | None = None) -> None:
        super().__init__()

        self._model = model or self.config.GENERATOR_HF_MODEL
        # OpenAI-compatible endpoint for chat models
        self._api_url = "https://router.huggingface.co/v1/chat/completions"
        self._headers = {
            "Authorization": f"Bearer {self.config.GENERATOR_HF_API_TOKEN}",
            "Content-Type": "application/json",
        }
        self._max_tokens = self.config.GENERATOR_MAX_TOKENS
        self._timeout = self.config.GENERATOR_DEFAULT_TIMEOUT

        # Persistent client for connection pooling
        self._client = httpx.AsyncClient(timeout=self._timeout, headers=self._headers)

    async def _call_api(self, messages: list[dict], temperature: float = 0.1) -> str:
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self._max_tokens,
            "top_p": 0.9,
            # Critical for serverless: wait for model to load
            # "options": {"wait_for_model": True, "use_cache": True},
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self._client.post(self._api_url, json=payload)

                # Handle 503: Model loading (common on HF serverless)
                if response.status_code == 503:
                    wait_time = 20 * (attempt + 1)  # Exponential: 20s, 40s, 60s
                    logger.warning(f"HF model loading (attempt {attempt+1}). Waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue

                # Handle 429: Rate limit
                if response.status_code == 429:
                    retry_after = int(response.headers.get("retry-after", 30))
                    logger.warning(f"HF rate limited. Waiting {retry_after}s...")
                    await asyncio.sleep(retry_after)
                    continue

                # Handle 400: Bad request (model not supported, etc.)
                if response.status_code == 400:
                    error_text = response.text[:200]
                    raise ValueError(f"HF API bad request: {error_text}")

                response.raise_for_status()
                result = response.json()

                if "choices" in result and result["choices"]:
                    content = result["choices"][0]["message"]["content"]
                    return content.strip()

                raise ValueError(f"Unexpected HF response format: {result}")

            except httpx.TimeoutException:
                logger.warning(f"HF timeout on attempt {attempt+1}/{max_retries}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

            except httpx.HTTPError as exc:
                logger.warning(f"HF HTTP error on attempt {attempt+1}: {exc}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

        raise Exception(f"HF API failed after {max_retries} retries")

    async def aclose(self) -> None:
        await self._client.aclose()

    async def generate(
        self,
        question: str,
        context: str,
        sources: list[dict] | None = None,
        history: List[Dict[str, str]] | None = None,
    ) -> str:

        system_content = RAGGeneratorPrompt.BASE_SYSTEM.value

        # ── 2. User message ──────────────────────────────────────────────────
        has_context = bool(context and context.strip())

        user_content = RAGGeneratorPrompt.BASE_USER.value.format(
            question=question.strip(),
            context=context if has_context else "No documents available for this query.",
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_content}
        ]

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": user_content})
        # ─────────────────────────────────────────────────────────────────────

        try:
            # Low temperature for factual answers
            answer = await self._call_api(messages, temperature=0.1)
            logger.debug("HF generated answer for: %r", question[:60])
            return answer
        except Exception as exc:
            logger.warning("HF generation failed (%s), returning fallback.", exc)
            return "Sorry, I encountered an error while generating the answer."


@lru_cache(maxsize=1)
def get_hf_generator(model: str | None = None) -> HuggingFaceGenerator:
    """Return singleton HuggingFaceGenerator instance."""
    return HuggingFaceGenerator(model=model)

def reset_hf_generator_cache() -> None:
    """Clear the singleton cache — useful for testing."""
    get_hf_generator.cache_clear()