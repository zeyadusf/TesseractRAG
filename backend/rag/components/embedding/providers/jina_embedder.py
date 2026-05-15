from ..base_embedder import BaseEmbedder
from backend.models.embedding import EmbeddingChunk, EmbeddingMeta
from backend.models.enums.embedding_enum import  JinaEmbedTasks
from backend.core import get_logger
from typing import Optional, Union, AsyncGenerator, List
import httpx
import asyncio


logger = get_logger(__name__)


class JinaProvider(BaseEmbedder):

    def __init__(self):
        super().__init__()
        self._headers = {
            "Authorization": f"Bearer {self.config.JINA_API_KEY}",
            "Content-Type": "application/json",
        }
        self._client: Optional[httpx.AsyncClient] = None
        self._owns_client = False

    async def _ensure_client(self) -> None:
        """Lazy-init httpx client with connection pooling."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.config.TIME_OUT,
                limits=httpx.Limits(
                    max_connections=self.config.MAX_CONNECTIONS,
                    max_keepalive_connections=self.config.MAX_CONNECTIONS,
                ),
                follow_redirects=True,
            )
            self._owns_client = True
            logger.debug("[JinaEmbedder] HTTP client initialized")

    async def aclose(self) -> None:
        """Close the HTTP client if we own it."""
        if self._client and self._owns_client:
            await self._client.aclose()
            self._client = None
            logger.debug("[JinaEmbedder] HTTP client closed")

    def _make_batches(self, texts: List[str], late_chunking: bool) -> List[List[str]]:
        """
        Split text list into batches, respecting late_chunking token limits.

        When late_chunking=True we conservatively cap batch size so the total
        tokens per request stays under 8 192 (assuming ~256 tokens/text avg).
        """
        if not texts:
            return []

        effective_batch_size = self.config.BATCH_SIZE
        if late_chunking:
            max_for_late = max(1, self.config.MAX_TOKENS_LATE_CHUNKING // 256)
            effective_batch_size = min(effective_batch_size, max_for_late)
            logger.debug(
                "[JinaEmbedder] late_chunking=True: adjusted batch_size to %d",
                effective_batch_size,
            )

        return [
            texts[i : i + effective_batch_size]
            for i in range(0, len(texts), effective_batch_size)
        ]

    async def _embed_with_retry(self,texts: List[str],task: str,late_chunking: bool,) -> dict:
        """
        Call Jina API for one batch with exponential-backoff retry.
        Returns
        -------
        {
            "index":           List[int],
            "embeddings":      List[List[float]],
            "total_tokens":    int,
            "model":           str,
            "per_item_tokens": List[int],   # only when available for every item
        }
        """
        payload = {
            "model":         self.config.JINA_MODEL,
            "input":         texts,
            "task":          task,           
            "normalized":    True,
            "dimensions":    self.config.EMBED_DIM,
            "late_chunking": late_chunking,
        }

        for attempt in range(self.config.MAX_RETRIES):
            try:
                if self._client is None:
                    raise RuntimeError(
                        "HTTP client not initialized. Call _ensure_client() first."
                    )

                response = await self._client.post(
                    self.config.JINA_BASE_URL,
                    json=payload,
                    headers=self._headers,
                )
                response.raise_for_status()
                data = response.json()

                embeddings: List[List[float]] = []
                indexes: List[int] = []
                per_item_tokens: List[Optional[int]] = []

                for item in data.get("data", []):
                    embeddings.append(item["embedding"])
                    indexes.append(item["index"])
                    usage = item.get("usage", {})
                    per_item_tokens.append(usage.get("prompt_tokens"))

                if any(t is None for t in per_item_tokens):
                    per_item_tokens = []

                result = {
                    "index":        indexes,
                    "embeddings":   embeddings,
                    "total_tokens": data["usage"]["total_tokens"],
                    "model":        data["model"],
                }
                if per_item_tokens:
                    result["per_item_tokens"] = per_item_tokens

                return result

            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                error_body = exc.response.text[:200]

                if status == 429:
                    wait = min(2 ** attempt, 30)
                    logger.warning(
                        "[JinaEmbedder] Rate limited (attempt %d/%d). "
                        "Retrying in %ds... | Response: %s",
                        attempt + 1, self.config.MAX_RETRIES, wait, error_body,
                    )
                    await asyncio.sleep(wait)
                    continue

                elif status == 402:
                    logger.error("[JinaEmbedder] Free quota exhausted (1M tokens/month)")
                    raise RuntimeError(
                        "Jina free quota exhausted. "
                        "Check usage at jina.ai/api-dashboard or upgrade your plan."
                    ) from exc

                elif status == 401:
                    logger.error("[JinaEmbedder] Invalid API key")
                    raise RuntimeError(
                        "Invalid Jina API key. Get yours free at jina.ai"
                    ) from exc

                elif status == 400:
                    logger.error("[JinaEmbedder] Bad request: %s", error_body)
                    if late_chunking:
                        raise RuntimeError(
                            f"Request exceeded token limit (8192) with late_chunking=True. "
                            f"Try reducing BATCH_SIZE or disable late_chunking. Details: {error_body}"
                        ) from exc
                    raise

                else:
                    logger.error("[JinaEmbedder] HTTP %d: %s", status, error_body)
                    raise

            except httpx.TimeoutException:
                wait = min(2 ** attempt, 30)
                logger.warning(
                    "[JinaEmbedder] Timeout after %.1fs (attempt %d/%d). Retrying in %ds...",
                    self.config.TIME_OUT, attempt + 1, self.config.MAX_RETRIES, wait,
                )
                await asyncio.sleep(wait)
                continue

            except httpx.RequestError as exc:
                logger.warning(
                    "[JinaEmbedder] Request error (attempt %d): %s: %s",
                    attempt + 1, type(exc).__name__, exc,
                )
                await asyncio.sleep(1)
                continue

        raise RuntimeError(
            f"[JinaEmbedder] Failed after {self.config.MAX_RETRIES} retries."
        )

    async def embed_documents(
        self,
        texts: List[str],
        late_chunking: bool = False,
    ) -> AsyncGenerator[Union[EmbeddingChunk, EmbeddingMeta], None]:

        if not texts:
            logger.warning("[JinaEmbedder] embed_documents called with empty list")
            yield EmbeddingMeta(
                model=self.config.JINA_MODEL,
                total_tokens=0,
                total_chunks=0,
                dimensions=self.config.EMBED_DIM,)
            return

        await self._ensure_client()

        total_tokens = 0
        global_index = 0

        for batch_texts in self._make_batches(texts, late_chunking):
            batch_result = await self._embed_with_retry(
                texts=batch_texts,
                task=JinaEmbedTasks.EmbedDoc,
                late_chunking=late_chunking,
            )

            batch_len = len(batch_texts)
            embeddings_list: List[List[float]] = batch_result["embeddings"]
            indexes_list: List[int] = batch_result.get("index", list(range(batch_len)))
            per_item_tokens: List[int] = batch_result.get("per_item_tokens", [])

            for idx, text in enumerate(batch_texts):
                if per_item_tokens and idx < len(per_item_tokens):
                    item_tokens = per_item_tokens[idx]
                    is_estimate = False
                else:
                    item_tokens = (
                        batch_result["total_tokens"] // batch_len if batch_len else 0
                    )
                    is_estimate = True

                embedding = embeddings_list[idx] if idx < len(embeddings_list) else []
                chunk_index = indexes_list[idx] if idx < len(indexes_list) else global_index

                yield EmbeddingChunk(
                    index=chunk_index,
                    text=text,
                    embedding=embedding,
                    tokens=item_tokens,
                    is_estimate=is_estimate,
                )

                total_tokens += item_tokens
                global_index += 1

            await asyncio.sleep(0.1)  #  friendly to rate limits between batches

        yield EmbeddingMeta(
            model=self.config.JINA_MODEL,
            total_tokens=total_tokens,
            total_chunks=global_index,
            dimensions=self.config.EMBED_DIM,
        )
        logger.info(
            "[JinaEmbedder] Completed: %d chunks, %d tokens, model=%s",
            global_index, total_tokens, self.config.JINA_MODEL,
        )

    async def embed_query(self, query: str) -> List[float]:
        """
        Embed a single user search query.
        Task: retrieval.query — MUST differ from passage (asymmetric retrieval).

        Returns
        -------
        List[float] — vector ready for pgvector similarity search.
        """
        if not query or not query.strip():
            raise ValueError("Query text cannot be empty")

        await self._ensure_client()

        result = await self._embed_with_retry(
            texts=[query],
            task=JinaEmbedTasks.EmbedQuery,
            late_chunking=False,
        )
        return result["embeddings"][0]