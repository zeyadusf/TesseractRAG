from __future__ import annotations

import asyncio
from functools import lru_cache

import httpx

from backend.core import get_config, get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    """
    Reranker using HuggingFace Inference API — no model loaded in memory.
    Uses the same pattern as HuggingFaceGenerator.
    Compatible with BAAI/bge-reranker-v2-m3 (multilingual cross-encoder).

    bge-reranker-v2-m3 returns raw logits directly (not probabilities),
    so no sigmoid-to-logit conversion is needed.
    Higher logit = more relevant.
    """

    def __init__(self, batch_size: int = 16) -> None:
        config = get_config()
        self._model = config.RERANKER_MODEL  # BAAI/bge-reranker-v2-m3
        self._batch_size = batch_size
        self._timeout = getattr(config, "GENERATOR_DEFAULT_TIMEOUT", 30)

        # HF feature-extraction endpoint (returns raw logits, no chat wrapper)
        self._api_url = f"https://router.huggingface.co/hf-inference/models/{self._model}"

        self._client = httpx.AsyncClient(
            timeout=self._timeout,
            headers={
                "Authorization": f"Bearer {config.GENERATOR_HF_API_TOKEN}",
                "Content-Type": "application/json",
            },
        )

    async def warmup(self) -> None:
        try:
            await self._call_api("test", ["test"])
            logger.info("Reranker warmed up successfully.")
        except Exception:
            logger.warning("Reranker warmup failed — will retry on first real request.")
    # ── Internal API call ─────────────────────────────────────────────────────

    async def _call_api(self, query: str, documents: list[str]) -> list[float]:
        """
        Call HF rerank endpoint and return a relevance logit per document.
        bge-reranker-v2-m3 returns raw logits — higher is more relevant.
        Handles 503 (model loading) and 429 (rate limit) with retries.
        """
        payload = {
            "inputs": [
                {"text": query, "text_pair": doc}
                for doc in documents
            ]
        }

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


                scores: list[float] = []
                pairs = result[0] if isinstance(result[0], list) else result

                for item in pairs:
                    if isinstance(item, list):
                        # nested list: [[{"label":..,"score":..}, ...], ...]
                        label1 = next(
                            (x["score"] for x in item if x.get("label") == "LABEL_1"),
                            None,
                        )
                        if label1 is not None:
                            scores.append(float(label1))
                        else:
                            # fallback: take max score in the pair
                            scores.append(max(x["score"] for x in item))
                    elif isinstance(item, dict) and "score" in item:
                        # flat list: [{"label": "LABEL_1", "score": 0.99}, ...]
                        scores.append(float(item["score"]))
                    else:
                        # unexpected format — use 0.0 as fallback
                        logger.warning(f"Unexpected reranker item format: {item}")
                        scores.append(0.0)

                if not scores:
                    raise ValueError(f"Unexpected reranker response format: {result}")

                return scores

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

    # ── Public interface ──────────────────────────────────────────────────────

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


# ── Singleton helper ──────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_reranker(batch_size: int = 16) -> CrossEncoderReranker:
    """Return a singleton CrossEncoderReranker (no GPU/model in memory)."""
    return CrossEncoderReranker(batch_size=batch_size)


def reset_reranker_cache() -> None:
    """Clear singleton cache — useful for testing."""
    get_reranker.cache_clear()