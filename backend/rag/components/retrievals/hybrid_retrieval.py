"""
Hybrid retriever: BM25 (lexical) + Vector (semantic) fused via
Reciprocal Rank Fusion (RRF).

Accepts chunks as a list OR a generator/iterator — build() materialises
the iterator once so both BM25 and the chunk-map share the same data.
"""

from __future__ import annotations

from typing import Iterator
from uuid import UUID

from backend.core import get_logger
from backend.storage.vector_db.base import BaseVectorStore
from .bm25_retrieval import BM25Retriever

logger = get_logger(__name__)

_RRF_K = 60  # Cormack & Clarke 2009


def _reciprocal_rank_fusion(
    bm25_results: list[dict],
    vector_results: list[dict],
    rrf_k: int = _RRF_K,
) -> list[dict]:
    """
    Merge two ranked lists with RRF.

    Score per chunk = Σ  1 / (k + rank)  across both lists.
    Both lists must carry a ``chunk_id`` key (str or UUID).
    """
    scores: dict[str, float] = {}

    for rank, item in enumerate(bm25_results, start=1):
        cid = str(item["chunk_id"])
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)

    for rank, item in enumerate(vector_results, start=1):
        cid = str(item["chunk_id"])
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)

    sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)
    return [
        {
            "chunk_id": sorted_ids[i],
            "rrf_score": round(scores[sorted_ids[i]], 6),
            "rank": i + 1,
        }
        for i in range(len(sorted_ids))
    ]


class HybridRetriever:
    """
    BM25 + vector search fused with Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore | None,
        bm25_top_k: int = 20,
        vector_top_k: int = 20,
        final_top_k: int = 10,
    ) -> None:
        self._vector_store = vector_store
        self._bm25 = BM25Retriever()
        self._bm25_top_k = bm25_top_k
        self._vector_top_k = vector_top_k
        self._final_top_k = final_top_k
        self._chunk_map: dict[str, dict] = {}

    # ── Indexing ──────────────────────────────────────────────────────────────

    def build(self, chunks: list[dict] | Iterator[dict]) -> None:
        """
        Index chunks into BM25 and populate the internal chunk-id lookup.
        Each chunk must have at minimum::
            {"chunk_id": UUID | str, "content": str}
        """
        # Materialise once — generators are single-pass
        if not isinstance(chunks, list):
            logger.debug("HybridRetriever.build: materialising chunk iterator")
            chunks = list(chunks)

        if not chunks:
            logger.warning("HybridRetriever.build called with zero chunks")
            return

        self._bm25.build(chunks)
        self._chunk_map = {str(c["chunk_id"]): c for c in chunks}
        logger.info(f"HybridRetriever indexed {len(chunks)} chunks")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        session_id: UUID,
        query_vector: list[float],
        model_name: str = "default",
    ) -> list[dict]:
        """
        Run BM25 + vector search then fuse with RRF.

        Returns enriched chunk dicts (original fields + ``rrf_score`` +
        ``rank``) sorted by ``rrf_score`` descending.

        NOTE: requires self._vector_store to be set (not None).
        """
        if self._vector_store is None:
            raise ValueError(
                "retrieve() requires a vector_store. "
                "Use fuse() if you already have vector results."
            )

        # 1. BM25 (sync)
        bm25_raw = self._bm25.retrieve(query, top_k=self._bm25_top_k)
        bm25_results = [{"chunk_id": str(c["chunk_id"]), **c} for c in bm25_raw]

        # 2. Vector (async)
        vector_raw = await self._vector_store.search(
            session_id=session_id,
            query_vector=query_vector,
            model_name=model_name,
            top_k=self._vector_top_k,
        )
        vector_results = [{**r, "chunk_id": str(r["chunk_id"])} for r in vector_raw]

        # 3. RRF
        fused = _reciprocal_rank_fusion(bm25_results, vector_results)
        fused = fused[: self._final_top_k]

        # 4. Enrich with full chunk content / metadata
        enriched: list[dict] = []
        for item in fused:
            chunk = self._chunk_map.get(item["chunk_id"], {})
            enriched.append({**chunk, **item})

        logger.info(
            f"HybridRetriever: bm25={len(bm25_results)} "
            f"vector={len(vector_results)} fused→{len(enriched)}"
        )
        return enriched

    async def fuse(
        self,
        *,
        query: str,
        chunks: list[dict],
        vector_results: list[dict],
    ) -> list[dict]:
        # 1. BM25
        bm25_raw = self._bm25.retrieve(query, top_k=self._bm25_top_k)
        bm25_results = [{"chunk_id": str(c["chunk_id"]), **c} for c in bm25_raw]

        # 2. Normalize vector results
        normalised_vector = [
            {**r, "chunk_id": str(r["chunk_id"])}
            for r in vector_results
        ]

        # 3. RRF
        fused = _reciprocal_rank_fusion(bm25_results, normalised_vector)
        fused = fused[: self._final_top_k]

        # 4. Enrich — normalize chunk_map keys to str to avoid UUID vs str mismatch
        chunk_map = {str(c["chunk_id"]): c for c in chunks}

        results: list[dict] = []
        for item in fused:
            cid = str(item["chunk_id"])
            chunk = chunk_map.get(cid) or self._chunk_map.get(cid)
            
            if not chunk:
                logger.warning(f"[HYBRID] chunk_id '{cid}' not found in any map — skipping")
                continue                         
            
            results.append({
                **chunk,                          
                "chunk_id":  cid,                
                "rrf_score": item["rrf_score"],
                "rank":      item["rank"],
                "score":     item["rrf_score"], 
            })

        logger.info(
            f"HybridRetriever.fuse: bm25={len(bm25_results)} "
            f"vector={len(normalised_vector)} fused→{len(results)}"
        )
        return results