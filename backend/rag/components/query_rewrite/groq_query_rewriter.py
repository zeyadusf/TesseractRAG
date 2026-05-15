from __future__ import annotations

import asyncio
import re
from functools import lru_cache
import httpx

from backend.models.enums.query_prompts import QueryReWritePrompt
from backend.core import get_logger
from .query_base import QueryRewriteBase

logger = get_logger(__name__)


class GroqQueryRewriter(QueryRewriteBase):
    """
    Production-ready Groq Llama-3 rewriter.
    Includes strict prompting, hallucination filtering, and Arabic-safe cleaning.
    """

    def __init__(self) -> None:
        super().__init__()
        
        self._headers = {
            "Authorization": f"Bearer {self.config.QUERY_GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        self._max_tokens: int = self.config.QUERY_MAX_NEW_TOKENS
        self._api_url: str = self.config.QUERY_GROQ_API_URL
        self._model: str = self.config.QUERY_MODEL

        self._client = httpx.AsyncClient(
            timeout=self.config.QUERY_DEFAULT_TIMEOUT,
            headers=self._headers,
        )

    # ── Smart Cleaning & Validation ───────────────────────────────────────────
    def _clean_and_validate(self, raw: str, original_query: str, is_expand: bool = False) -> str:
        text = raw.strip()
        # 1. Strip quotes, markdown, take first meaningful line
        text = re.sub(r'^["\'\u201c\u201d`]+|["\'\u201c\u201d`]+$', '', text).strip()
        text = text.split('\n')[0].strip()
        text = re.sub(r'^```[\w]*\s*|```$', '', text).strip()

        # 2. Remove boolean operators & common LLM filler
        text = re.sub(r'\b(OR|AND|NOT|Here|are|the|terms|query|keywords)\b', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()

        # 3. Extract valid words (English + Arabic + Numbers)
        words = re.findall(r'[A-Za-z0-9\u0600-\u06FF\u0660-\u0669]+', text)
        
        if is_expand:
            # Deduplicate case-insensitively
            seen = set()
            clean_words = []
            for w in words:
                low = w.lower()
                if low not in seen:
                    seen.add(low)
                    clean_words.append(w)
            text = ' '.join(clean_words)
            
            # Safety: Reject hallucinations or useless outputs
            blacklist = {'english query', 'n/a', 'none', 'undefined', 'keywords'}
            if text.lower() in blacklist or len(text) < 3:
                return ""

        return text

    # ── API Call ──────────────────────────────────────────────────────────────
    async def _call_api(self, prompt: str, temperature: float = 0.1) -> str:
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self._max_tokens,
            "temperature": temperature,
        }

        response = await self._client.post(self._api_url, json=payload)

        if response.status_code == 429:
            retry_after = int(response.headers.get("retry-after", 2))
            logger.warning(f"Groq rate limit hit. Retrying in {retry_after}s...")
            await asyncio.sleep(retry_after)
            response = await self._client.post(self._api_url, json=payload)

        response.raise_for_status()
        result = response.json()

        if "choices" in result and result["choices"]:
            return result["choices"][0]["message"]["content"]
        raise ValueError(f"Unexpected Groq API response format: {result}")

    async def aclose(self) -> None:
        await self._client.aclose()

    async def rewrite(self, query: str) -> str:
        if not query or not query.strip():
            return query

        prompt = QueryReWritePrompt.BASE.format(query=query.strip())
        try:
            raw = await self._call_api(prompt, temperature=0.1)
            cleaned = self._clean_and_validate(raw, query, is_expand=False)
            logger.debug("Query rewrite: %r → %r", query, cleaned)
            return cleaned or query
        except Exception as exc:
            logger.warning("Query rewrite failed (%s), using original.", exc)
            return query

    async def expand(self, query: str) -> str:
        if not query or not query.strip():
            return query

        prompt = QueryReWritePrompt.EXPAND.format(query=query.strip())
        try:
            raw = await self._call_api(prompt, temperature=0.0)  # Deterministic for keywords
            cleaned = self._clean_and_validate(raw, query, is_expand=True)
            logger.debug("Query expand: %r → %r", query, cleaned)
            return cleaned or query  # Fallback to original if cleaned is empty
        except Exception as exc:
            logger.warning("Query expand failed (%s), using original.", exc)
            return query

    async def rewrite_and_expand(self, query: str) -> dict[str, str]:
        rewritten, expanded = await asyncio.gather(
            self.rewrite(query),
            self.expand(query),
        )
        return {
            "original": query,
            "rewritten": rewritten,
            "expanded": expanded,
        }


@lru_cache(maxsize=1)
def get_groq_rewriter() -> GroqQueryRewriter:
    return GroqQueryRewriter()


def reset_groq_rewriter_cache() -> None:
    """Clear the singleton cache — useful for testing."""
    get_groq_rewriter.cache_clear()