from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import List, Dict

import httpx

from backend.models.enums.generator_prompts import RAGGeneratorPrompt
from backend.core import get_logger
from .generator_base import AnswerGeneratorBase

logger = get_logger(__name__)


class GroqAnswerGenerator(AnswerGeneratorBase):
    """
    Answer generator using Groq API (Llama-3.3-70B-Versatile).
    Optimized for RAG: strict context adherence + citation + Arabic support.
    """

    def __init__(self, model: str | None = None) -> None:
        super().__init__()

        self._model = model or self.config.GENERATOR_GROQ_MODEL
        self._api_url = self.config.GENERATOR_GROQ_API_URL
        self._headers = {
            "Authorization": f"Bearer {self.config.GENERATOR_GROQ_API_KEY}",
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
            "max_tokens": self._max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self._client.post(self._api_url, json=payload)

                # Handle rate limits (429)
                if response.status_code == 429:
                    retry_after = int(response.headers.get("retry-after", 5))
                    logger.warning(f"Groq rate limit. Retrying in {retry_after}s...")
                    await asyncio.sleep(retry_after)
                    continue

                response.raise_for_status()
                result = response.json()

                if "choices" in result and result["choices"]:
                    content = result["choices"][0]["message"]["content"]
                    return content.strip()

                raise ValueError(f"Unexpected Groq API response: {result}")

            except httpx.TimeoutException:
                logger.warning(f"Groq timeout on attempt {attempt+1}/{max_retries}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

            except httpx.HTTPError as exc:
                logger.warning(f"Groq HTTP error on attempt {attempt+1}: {exc}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

        raise Exception(f"Groq API failed after {max_retries} retries")

    async def aclose(self) -> None:
        await self._client.aclose()

    async def generate(
        self,
        question: str,
        context: str,
        sources: list[dict] | None = None,
        history: List[Dict[str, str]] | None = None,  # ← NEW
    ) -> str:
        if not context or not context.strip():
            return "I don't have enough information to answer this question."

        prompt = RAGGeneratorPrompt.BASE.format(
            question=question.strip(),
            context=context
        )

        # ── Build messages with history ─────────────────────────────
        # History format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        messages: List[Dict[str, str]] = []

        if history:
            messages.extend(history)  # آخر 5 turns (10 messages)

        messages.append({"role": "user", "content": prompt})
        # ────────────────────────────────────────────────────────────

        try:
            answer = await self._call_api(messages, temperature=0.1)
            logger.debug("Groq generated answer for: %r", question[:60])
            return answer
        except Exception as exc:
            logger.warning("Groq generation failed (%s), returning fallback.", exc)
            return "Sorry, I encountered an error while generating the answer."


@lru_cache(maxsize=1)
def get_groq_generator(model: str | None = None) -> GroqAnswerGenerator:
    """Return singleton GroqAnswerGenerator instance."""
    return GroqAnswerGenerator(model=model)

def reset_groq_generator_cache() -> None:
    """Clear the singleton cache — useful for testing."""
    get_groq_generator.cache_clear()