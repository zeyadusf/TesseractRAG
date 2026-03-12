from rank_bm25 import BM25Okapi
from app.utils.logger import get_logger

logger = get_logger(__name__)

class BM25Retriever:
    def __init__(self):
        self.bm25 = None          # BM25Okapi instance — None until built
        self.chunks = []          # the chunk dicts — needed to return results

    def build(self, chunks: list[dict]) -> None:
        tokenized  =[c['content'].lower().split() for c in chunks]
        self.chunks = chunks
        self.bm25 = BM25Okapi(tokenized )
        logger.info('build bm25')

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        # If self.bm25 is None, return []
        if self.bm25 is None : return []
        # Tokenize the query the same way as chunks
        tokenized_query = query.lower().split() 
        # Call self.bm25.get_top_n(tokenized_query, self.chunks, n=top_k)
        result = self.bm25.get_top_n(tokenized_query, self.chunks, n=top_k)
        # Return the result
        return result