from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import List, Dict

import httpx

from backend.models.enums.generator_prompts import RAGGeneratorPrompt
from backend.core import get_logger
from .generator_base import AnswerGeneratorBase

logger = get_logger(__name__)


class GroqRateLimitError(Exception):
    """Raised when Groq returns 429 — lets SmartGeneratorGuard fallback to HF."""
    pass


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

                # Handle rate limits (429) — raise immediately so SmartGeneratorGuard
                # can catch it and fallback to HF instead of waiting 700+ seconds
                if response.status_code == 429:
                    retry_after = int(response.headers.get("retry-after", 60))
                    logger.warning(
                        f"Groq rate limit hit. retry-after={retry_after}s. "
                        f"Raising GroqRateLimitError for fallback to HF."
                    )
                    raise GroqRateLimitError(
                        f"Groq rate limit exceeded. retry-after={retry_after}s"
                    )

                response.raise_for_status()
                result = response.json()

                if "choices" in result and result["choices"]:
                    content = result["choices"][0]["message"]["content"]
                    return content.strip()

                raise ValueError(f"Unexpected Groq API response: {result}")

            except GroqRateLimitError:
                # Re-raise immediately — no retry, let the guard handle it
                raise

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
        history: List[Dict[str, str]] | None = None,
    ) -> str:

        # ── 1. System message يحمل الـ rules كاملة (مرة واحدة بس) ──────────
        system_content = RAGGeneratorPrompt.BASE_SYSTEM.value

        # ── 2. User message ──────────────────────────────────────────────────
        has_context = bool(context and context.strip())

        user_content = RAGGeneratorPrompt.BASE_USER.value.format(
            question=question.strip(),
            context=context if has_context else "No documents available for this query.",
        )

        # ── 3. بناء الـ messages: system → history → user حالي ──────────────
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_content}
        ]

        if history:
            # history = [{"role": "user", "content": "سؤال خام"}, {"role": "assistant", "content": "إجابة"}, ...]
            messages.extend(history)

        messages.append({"role": "user", "content": user_content})
        # ─────────────────────────────────────────────────────────────────────

        try:
            print("#%$"*20)
            print("messages:",messages)
            print("#%$"*20)
            answer = await self._call_api(messages, temperature=0.1)
            logger.debug("Groq generated answer for: %r", question[:60])
            return answer
        except GroqRateLimitError:
            # Re-raise so SmartGeneratorGuard falls back to HF
            raise
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