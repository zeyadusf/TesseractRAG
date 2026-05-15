import re
from rank_bm25 import BM25Okapi
from backend.core import get_logger

logger = get_logger(__name__)


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:

    def __init__(self) -> None:
        self.bm25: BM25Okapi | None = None
        self.chunks: list[dict] = []

    def build(self, chunks: list[dict]) -> None:
        if not chunks:
            logger.warning("BM25Retriever.build called with zero chunks")
            return

        tokenized = [tokenize(c["content"]) for c in chunks]
        self.chunks = chunks
        self.bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 built with {len(chunks)} chunks")

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        if not self.bm25:
            logger.warning("BM25Retriever.retrieve called before build()")
            return []

        tokenized_query = tokenize(query)
        results = self.bm25.get_top_n(tokenized_query, self.chunks, n=top_k)
        return list(results)