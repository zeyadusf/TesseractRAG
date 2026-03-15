# Run from backend/ with: python -m app.temp

from app.dependencies import get_session_manager
from app.core.ingestion.embedder import Embedder
from app.core.retrieval.hybrid_retriever import HybridRetriever
from app.core.retrieval.reranker import CrossEncoderReranker
from app.core.retrieval.router import RetrievalRouter

# ── Setup ─────────────────────────────────────────────────────────────
manager = get_session_manager()
session = manager.get_session("4f3b95e4-6b3f-46c4-ba18-f5cd90ccd9b0")
QUERY = "What is TesseractRAG and RAG ,i think rag is system ?"
# QUERY = "What deployment platforms does TesseractRAG support?"


# ── Retrieve ──────────────────────────────────────────────────────────
retriever = HybridRetriever(
    bm25_retriever=session.bm25_retriever,
    faiss_indexer=session.faiss_index,
    embedder=Embedder(),
    chunks=session.chunks
)

# ───Router──────────────────────────────────────
router = RetrievalRouter()
strategy = router.route(QUERY, user_strategy="auto")
print(f"Router selected strategy: {strategy}")

pre_rerank = retriever.retrieve(QUERY, strategy=strategy, top_k=10)

print("=" * 60)
print(f"BEFORE RERANKING — top 10 {strategy} results")
print("=" * 60)
for i, chunk in enumerate(pre_rerank):
    print(f"  [{i}] {chunk['document_name']} | {chunk['content'][:80]}...")

# ── Rerank ────────────────────────────────────────────────────────────
reranker = CrossEncoderReranker()
post_rerank = reranker.rerank(QUERY, pre_rerank, top_k=3)

print("\n" + "=" * 60)
print("AFTER RERANKING — top 3")
print("=" * 60)
for i, chunk in enumerate(post_rerank):
    print(f"  [{i}] {chunk['document_name']} | {chunk['content'][:80]}...")

# ── Order comparison ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ORDER CHANGE ANALYSIS")
print("=" * 60)
pre_indices = [c['chunk_index'] for c in pre_rerank[:3]]
post_indices = [c['chunk_index'] for c in post_rerank]
print(f"  Pre-rerank  top 3 chunk indices: {pre_indices}")
print(f"  Post-rerank top 3 chunk indices: {post_indices}")
if pre_indices != post_indices:
    print("  ✅ Reranker changed the order")
else:
    print("  ⚠️  Order unchanged — reranker agreed with retrieval")