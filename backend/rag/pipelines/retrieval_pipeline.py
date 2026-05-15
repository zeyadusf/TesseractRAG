from __future__ import annotations

from typing import List, Dict

from backend.core import get_logger
from backend.rag.components.retrievals.retrieval_router import RetrievalRouter
from backend.rag.components.retrievals.hybrid_retrieval import HybridRetriever
from backend.rag.components.retrievals.bm25_retrieval import BM25Retriever
from backend.rag.components.reranker.bge_hf import get_reranker,reset_reranker_cache,CrossEncoderReranker

logger = get_logger(__name__)


class RetrievalPipeline:
    """
    Orchestrates retrieval strategy selection and execution.

    A single BM25Retriever instance is built once per run() call and
    reused across the lexical and hybrid paths — avoids rebuilding the
    index twice when the router falls back to hybrid after an initial
    lexical consideration.
    """
    def __init__(self, *, router: RetrievalRouter | None = None,reranker:CrossEncoderReranker|None = None  ,rerank_top_k: int = 5) -> None:
        self.router = router or RetrievalRouter()
        self._reranker = reranker or get_reranker()
        self._rerank_top_k = rerank_top_k

    async def run(
        self,
        *,
        query: str,
        query_vector: List[float],
        chunks: List[Dict],
        vector_results: List[Dict],
        strategy: str = "auto",
    ) -> List[Dict]:

        # ── 1. Choose strategy ─────────────────────────────────────────────
        strategy = self.router.route(query, strategy)
        logger.info(f"[RETRIEVAL] strategy = {strategy}")

        # ── 2. Build BM25 once ─────────────────────────────────────────────
        bm25 = BM25Retriever()
        bm25.build(chunks)

        # ── 3. Dispatch ────────────────────────────────────────────────────
        if strategy == "semantic":
            results = self._semantic_only(vector_results, chunks)

        elif strategy == "lexical":
            results = self._lexical_only(query, bm25, chunks)

        else:
            results = await self._hybrid(
                query=query,
                chunks=chunks,
                vector_results=vector_results,
                bm25=bm25,
            )

        # ── 4. Rerank ──────────────────────────────────────────────────────
        if self._reranker and results:
            results  = await self._reranker.arerank(query, results, top_k=self._rerank_top_k)
            logger.info(f"[RERANK] top {len(results)} after reranking")

        return results
    
    async def aclose(self):
        await self._reranker.aclose()
        reset_reranker_cache()

    # ── Strategy implementations ──────────────────────────────────────────────

    def _semantic_only(
        self,
        vector_results: List[Dict],
        chunks: List[Dict],
    ) -> List[Dict]:
        # normalize keys to str to avoid UUID vs str mismatch
        chunk_map = {str(c["chunk_id"]): c for c in chunks}

        results: List[Dict] = []
        for rank, item in enumerate(vector_results, start=1):
            cid = str(item["chunk_id"])
            chunk = chunk_map.get(cid)

            if not chunk:
                logger.warning(f"[SEMANTIC] chunk_id '{cid}' not found in chunk_map — skipping")
                continue                              # ← skip بدل ما يحط chunk فاضي

            results.append({
                **chunk,
                "chunk_id": cid,                     # ensure string
                "score":    float(item.get("score") or 0.0),
                "rank":     rank,
            })

        # logger.info(f"[SEMANTIC] returned {len(results)}/{len(vector_results)} enriched chunks")
        return results

    def _lexical_only(
        self,
        query: str,
        bm25: BM25Retriever,
        chunks: List[Dict],  
        rrf_k:int = 60,
    ) -> List[Dict]:
        """Return BM25 top-k results with rank and normalized score."""
        lexical_results = bm25.retrieve(query, top_k=10)
        # return [{"rank": i + 1, **c} for i, c in enumerate(results)]

        chunk_map = {str(c.get("chunk_id")): c for c in chunks}
        total = len(lexical_results)
        results = []

        for rank, item in enumerate(lexical_results, start=1):
            cid = str(item.get("chunk_id", ""))
            base = chunk_map.get(cid, {})
            if not base:
                continue

            score = 1.0 / (rrf_k + rank)


            results.append({
                **base,       # content, source_doc, chunk_index, 
                "chunk_id": cid,
                "score": round(score, 4),
                "rank": rank,
                "document_name": (
                    base.get("document_name")
                    or base.get("source_doc")
                    or base.get("document_id")
                    or "Unknown"
                ),
            })
        return results

    async def _hybrid(
        self,
        *,
        query: str,
        chunks: List[Dict],
        vector_results: List[Dict],
        bm25: BM25Retriever,
    ) -> List[Dict]:
        """Fuse BM25 + vector results via RRF using the pre-built BM25 index."""
        # Pass the already-built BM25 retriever into HybridRetriever so we
        # don't rebuild the index a second time.
        hybrid = HybridRetriever(vector_store=None)
        hybrid._bm25 = bm25                    # inject pre-built index
        hybrid._chunk_map = {str(c["chunk_id"]): c for c in chunks}

        #  fuse() is async — must be awaited
        results = await hybrid.fuse(
            query=query,
            chunks=chunks,
            vector_results=vector_results,
        )
        return results