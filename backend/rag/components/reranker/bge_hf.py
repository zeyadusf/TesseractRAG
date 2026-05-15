from __future__ import annotations

import asyncio
from functools import lru_cache

import httpx

from backend.core import get_config, get_logger

logger = get_logger(__name__)

import math

def _sigmoid_to_logit(p: float) -> float:
    p = max(1e-7, min(1 - 1e-7, p))
    return math.log(p / (1 - p))
class CrossEncoderReranker:
    """
    Reranker using HuggingFace Inference API — no model loaded in memory.
    Uses the same pattern as HuggingFaceGenerator.
    Compatible with BAAI/bge-reranker-base and similar cross-encoder models.
    """

    def __init__(self, batch_size: int = 16) -> None:
        config = get_config()
        self._model = config.RERANKER_MODEL  # e.g. BAAI/bge-reranker-base
        self._batch_size = batch_size
        self._timeout = getattr(config, "GENERATOR_DEFAULT_TIMEOUT", 30)

        # HF feature-extraction endpoint (returns raw scores, no chat wrapper)
        self._api_url = f"https://router.huggingface.co/hf-inference/models/{self._model}"

        self._client = httpx.AsyncClient(
            timeout=self._timeout,
            headers={
                "Authorization": f"Bearer {config.GENERATOR_HF_API_TOKEN}",
                "Content-Type": "application/json",
            },
        )

    # Internal API call

    async def _call_api(self, query: str, documents: list[str]) -> list[float]:
        """
        Call HF rerank endpoint and return a relevance score per document.
        Handles 503 (model loading) and 429 (rate limit) with retries.
        """
        # payload = {
        #     "model": self._model,
        #     "query": query,
        #     "documents": documents,
        #     "return_documents": False,  # we only need scores
        # }
        payload = {"inputs": [{"text": query, "text_pair": doc} for doc in documents]}


        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self._client.post(self._api_url, json=payload)

                if response.status_code == 503:
                    wait = 20 * (attempt + 1)
                    logger.warning(f"Reranker model loading (attempt {attempt+1}). Waiting {wait}s…")
                    await asyncio.sleep(wait)
                    continue

                if response.status_code == 429:
                    retry_after = int(response.headers.get("retry-after", 30))
                    logger.warning(f"Reranker rate limited. Waiting {retry_after}s…")
                    await asyncio.sleep(retry_after)
                    continue

                if response.status_code == 400:
                    raise ValueError(f"Reranker bad request: {response.text[:200]}")

                response.raise_for_status()
                result = response.json()

                # HF /rerank response: {"results": [{"index": 0, "relevance_score": 0.9}, ...]}
                inner: list = result[0] if isinstance(result[0], list) else result
                return [_sigmoid_to_logit(item["score"]) for item in inner]              

                raise ValueError(f"Unexpected reranker response: {result}")

            except httpx.TimeoutException:
                logger.warning(f"Reranker timeout (attempt {attempt+1}/{max_retries})")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

            except httpx.HTTPError as exc:
                logger.warning(f"Reranker HTTP error (attempt {attempt+1}): {exc}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

        raise RuntimeError(f"Reranker API failed after {max_retries} retries")

    # Public interface  (same signature as the old CrossEncoderReranker)

    async def arerank(self, query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
        """Async rerank — preferred entry-point."""
        if not chunks:
            return []

        top_k = min(top_k, len(chunks))
        documents = [chunk["content"] for chunk in chunks]

        all_scores: list[float] = []

        # Batch the documents to stay within API payload limits
        for i in range(0, len(documents), self._batch_size):
            batch_docs = documents[i : i + self._batch_size]
            batch_scores = await self._call_api(query, batch_docs)
            all_scores.extend(batch_scores)

        scored = sorted(zip(chunks, all_scores), key=lambda x: x[1], reverse=True)

        results = []
        for chunk, score in scored[:top_k]:
            out = dict(chunk)
            out["relevance_score"] = round(float(score), 4)
            results.append(out)

        logger.info(
            "Reranker: %d → top %d (best=%s)",
            len(chunks),
            len(results),
            results[0]["relevance_score"] if results else "n/a",
        )
        return results

    def rerank(self, query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
        """
        Sync wrapper — keeps backward-compatibility with callers that aren't async.
        Runs the async method in a new event loop if none is running.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already inside an async context — caller should use `await arerank()`
            raise RuntimeError(
                "Cannot call blocking rerank() inside a running event loop. "
                "Use `await reranker.arerank(...)` instead."
            )
        return asyncio.run(self.arerank(query, chunks, top_k))

    async def aclose(self) -> None:
        await self._client.aclose()


# Singleton helper  (mirrors get_hf_generator pattern)

@lru_cache(maxsize=1)
def get_reranker(batch_size: int = 16) -> CrossEncoderReranker:
    """Return a singleton CrossEncoderReranker (no GPU/model in memory)."""
    return CrossEncoderReranker(batch_size=batch_size)


def reset_reranker_cache() -> None:
    """Clear singleton cache — useful for testing."""
    get_reranker.cache_clear()