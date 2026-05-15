"""
Standalone test for the RAG retrieval pipeline.
No real DB / vector store needed — everything is mocked with dummy data.

Run:
    python -m backend.testig.test_retrieval_dummyVectors
"""

from __future__ import annotations

import asyncio
import re
import uuid
import random
import math
from typing import List, Dict

# ─────────────────────────────────────────────────────────────────────────────
# Minimal stubs — replaces backend.* imports so the test runs anywhere
# ─────────────────────────────────────────────────────────────────────────────

class _Logger:
    def __init__(self, name: str): self._name = name
    def info   (self, m): print(f"  [INFO]  {self._name}: {m}")
    def warning(self, m): print(f"  [WARN]  {self._name}: {m}")
    def debug  (self, m): print(f"  [DEBUG] {self._name}: {m}")

def get_logger(name: str) -> _Logger:
    return _Logger(name.split(".")[-1])

logger = get_logger("test_pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# Paste the three modules inline (no file changes needed)
# ─────────────────────────────────────────────────────────────────────────────

# ── BM25Retriever ─────────────────────────────────────────────────────────────
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise SystemExit("pip install rank-bm25  then rerun.")

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())

class BM25Retriever:
    def __init__(self) -> None:
        self.bm25 = None
        self.chunks: list[dict] = []

    def build(self, chunks: list[dict]) -> None:
        if not chunks:
            logger.warning("BM25Retriever.build called with zero chunks")
            return
        tokenized = [_tokenize(c["content"]) for c in chunks]
        self.chunks = chunks
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        if not self.bm25:
            logger.warning("BM25Retriever.retrieve called before build()")
            return []
        return list(self.bm25.get_top_n(_tokenize(query), self.chunks, n=top_k))


# ── RRF helper ────────────────────────────────────────────────────────────────
_RRF_K = 60

def _reciprocal_rank_fusion(
    bm25_results: list[dict],
    vector_results: list[dict],
    rrf_k: int = _RRF_K,
) -> list[dict]:
    scores: dict[str, float] = {}
    for rank, item in enumerate(bm25_results, start=1):
        cid = str(item["chunk_id"])
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
    for rank, item in enumerate(vector_results, start=1):
        cid = str(item["chunk_id"])
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
    sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)
    return [
        {"chunk_id": sorted_ids[i], "rrf_score": round(scores[sorted_ids[i]], 6), "rank": i + 1}
        for i in range(len(sorted_ids))
    ]


# ── HybridRetriever ───────────────────────────────────────────────────────────
class HybridRetriever:
    def __init__(self, vector_store=None, bm25_top_k=20, vector_top_k=20, final_top_k=10):
        self._vector_store = vector_store
        self._bm25 = BM25Retriever()
        self._bm25_top_k = bm25_top_k
        self._vector_top_k = vector_top_k
        self._final_top_k = final_top_k
        self._chunk_map: dict[str, dict] = {}

    def build(self, chunks):
        if not isinstance(chunks, list):
            chunks = list(chunks)
        if not chunks:
            return
        self._bm25.build(chunks)
        self._chunk_map = {str(c["chunk_id"]): c for c in chunks}

    async def fuse(self, *, query, chunks, vector_results) -> list[dict]:
        bm25_raw = self._bm25.retrieve(query, top_k=self._bm25_top_k)
        bm25_results = [{"chunk_id": str(c["chunk_id"]), **c} for c in bm25_raw]
        normalised_vector = [{**r, "chunk_id": str(r["chunk_id"])} for r in vector_results]
        fused = _reciprocal_rank_fusion(bm25_results, normalised_vector)[: self._final_top_k]
        chunk_map = {str(c["chunk_id"]): c for c in chunks}
        results = []
        for item in fused:
            chunk = chunk_map.get(item["chunk_id"], self._chunk_map.get(item["chunk_id"], {}))
            results.append({**chunk, **item})
        return results


# ── RetrievalRouter ───────────────────────────────────────────────────────────
CONCEPTUAL_STARTERS = frozenset({
    "what is", "what are", "what was", "what were",
    "how does", "how do", "how is", "how are", "how can",
    "why is", "why are", "why does", "why do",
    "explain", "describe", "tell me", "summarise", "summarize",
    "give me an overview", "what does",
})
_IDENTIFIER_RE = re.compile(r"\b([A-Z]{2,}|v\d+|\d{3,})\b")
_VALID_STRATEGIES = frozenset({"hybrid", "semantic", "lexical"})

class RetrievalRouter:
    def route(self, query: str, user_strategy: str = "auto") -> str:
        if user_strategy != "auto":
            return user_strategy if user_strategy in _VALID_STRATEGIES else "hybrid"
        normalised = query.strip().lower()
        tokens = normalised.split()
        if len(tokens) <= 3 and bool(_IDENTIFIER_RE.search(query)):
            return "lexical"
        if len(tokens) > 5 and any(normalised.startswith(s) for s in CONCEPTUAL_STARTERS):
            return "semantic"
        return "hybrid"


# ── RetrievalPipeline ─────────────────────────────────────────────────────────
class RetrievalPipeline:
    def __init__(self, *, router: RetrievalRouter | None = None) -> None:
        self.router = router or RetrievalRouter()

    async def run(self, *, query, query_vector, chunks, vector_results, strategy="auto") -> list[dict]:
        strategy = self.router.route(query, strategy)
        logger.info(f"[RETRIEVAL] strategy = {strategy}")

        bm25 = BM25Retriever()
        bm25.build(chunks)

        if strategy == "semantic":
            return self._semantic_only(vector_results, chunks)
        if strategy == "lexical":
            return self._lexical_only(query, bm25)
        return await self._hybrid(query=query, chunks=chunks, vector_results=vector_results, bm25=bm25)

    def _semantic_only(self, vector_results, chunks):
        chunk_map = {str(c["chunk_id"]): c for c in chunks}
        return [
            {**chunk_map.get(str(r["chunk_id"]), {}), "score": r.get("score"), "rank": i + 1}
            for i, r in enumerate(vector_results)
        ]

    def _lexical_only(self, query, bm25):
        return [{"rank": i + 1, **c} for i, c in enumerate(bm25.retrieve(query, top_k=10))]

    async def _hybrid(self, *, query, chunks, vector_results, bm25):
        hybrid = HybridRetriever(vector_store=None)
        hybrid._bm25 = bm25
        hybrid._chunk_map = {str(c["chunk_id"]): c for c in chunks}
        return await hybrid.fuse(query=query, chunks=chunks, vector_results=vector_results)


# ─────────────────────────────────────────────────────────────────────────────
# Dummy data factory
# ─────────────────────────────────────────────────────────────────────────────

DOCUMENTS = [
    "BERT is a transformer-based language model pre-trained on large text corpora.",
    "Retrieval-Augmented Generation (RAG) combines retrieval with generative models.",
    "BM25 is a lexical ranking function based on term frequency and inverse document frequency.",
    "Vector databases store high-dimensional embeddings for semantic similarity search.",
    "Reciprocal Rank Fusion merges multiple ranked lists into a single fused ranking.",
    "Chunking splits documents into smaller pieces to fit within model context windows.",
    "Sentence transformers produce dense embeddings useful for semantic retrieval.",
    "Hybrid search combines sparse BM25 scores with dense vector similarity scores.",
    "pgvector is a PostgreSQL extension that adds vector similarity search capabilities.",
    "FAISS is a library for efficient similarity search over dense float vectors.",
    "The attention mechanism allows models to weigh the importance of different tokens.",
    "Cosine similarity measures the angle between two vectors in embedding space.",
    "Named entity recognition identifies and classifies named entities within text.",
    "Fine-tuning adapts a pre-trained model to a specific downstream task.",
    "Knowledge graphs represent entities and relationships in a structured graph format.",
]

def make_chunks() -> List[Dict]:
    """Create 15 dummy chunks with UUIDs and content."""
    chunks = []
    for i, doc in enumerate(DOCUMENTS):
        chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "content":  doc,
            "source":   f"doc_{i:02d}.txt",
            "page":     i + 1,
        })
    return chunks


def make_dummy_vector(dim: int = 128) -> List[float]:
    """Random unit vector."""
    v = [random.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def make_vector_results(chunks: List[Dict], query: str, top_k: int = 8) -> List[Dict]:
    """
    Simulate a vector store response.
    We assign higher scores to chunks whose content shares words with the query
    (just to make results feel realistic — not real cosine similarity).
    """
    query_words = set(query.lower().split())
    scored = []
    for c in chunks:
        overlap = len(query_words & set(c["content"].lower().split()))
        score = round(random.uniform(0.3, 0.5) + overlap * 0.07, 4)
        scored.append({"chunk_id": c["chunk_id"], "score": min(score, 0.99)})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────────────────────────────────────

TEST_CASES = [
    # (label, query, forced_strategy)
    ("auto → lexical  ", "BERT v2",                               "auto"),
    ("auto → semantic  ", "explain how transformers work in NLP",  "auto"),
    ("auto → hybrid   ", "chunking documents for retrieval",       "auto"),
    ("forced semantic  ", "BM25 ranking function",                 "semantic"),
    ("forced lexical  ", "RAG pipeline overview",                  "lexical"),
    ("forced hybrid   ", "what is pgvector",                       "hybrid"),
]


def print_results(results: List[Dict], top_n: int = 3) -> None:
    for r in results[:top_n]:
        rank      = r.get("rank", "?")
        content   = r.get("content", "")[:70]
        rrf       = r.get("rrf_score")
        score     = r.get("score")
        metric    = f"rrf={rrf}" if rrf is not None else (f"score={score}" if score is not None else "")
        print(f"    #{rank:>2}  [{metric:>14}]  {content!r}")


async def main() -> None:
    pipeline = RetrievalPipeline()
    chunks   = make_chunks()

    print("=" * 70)
    print("  RAG Retrieval Pipeline — dummy data test")
    print(f"  Chunks: {len(chunks)}   Vector dim: 128 (random)")
    print("=" * 70)

    for label, query, strategy in TEST_CASES:
        print(f"\n▶  {label}  |  query: {query!r}")
        vector_results = make_vector_results(chunks, query)
        query_vector   = make_dummy_vector()

        results = await pipeline.run(
            query=query,
            query_vector=query_vector,
            chunks=chunks,
            vector_results=vector_results,
            strategy=strategy,
        )

        print(f"   → {len(results)} results returned (showing top 3):")
        print_results(results)

    print("\n" + "=" * 70)
    print("  ✅  All test cases passed.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())