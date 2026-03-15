# Run from backend/ with: python -m tests.unit.test_retrieval

from app.dependencies import get_session_manager
from app.core.ingestion.embedder import Embedder
from app.core.retrieval.hybrid_retriever import HybridRetriever

manager = get_session_manager()
id_session = "4f3b95e4-6b3f-46c4-ba18-f5cd90ccd9b0" # !!
session = manager.get_session(id_session)
QUERY = "What deployment platforms does TesseractRAG support?"

retriever = HybridRetriever(
    bm25_retriever=session.bm25_retriever,
    faiss_indexer=session.faiss_index,
    embedder=Embedder(),
    chunks=session.chunks
)

for strategy in ["hybrid", "semantic", "lexical"]:
    print("=" * 60)
    print(f"STRATEGY: {strategy.upper()}")
    print("=" * 60)
    results = retriever.retrieve(QUERY, strategy=strategy, top_k=5)
    for i, chunk in enumerate(results):
        print(f"  [{i}] {chunk['document_name']} | {chunk['content'][:80]}...")
    print()