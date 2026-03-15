from sentence_transformers import SentenceTransformer
from app.config import get_settings
from app.utils.logger import get_logger
import numpy as np

logger = get_logger(__name__)
_embedding_model = None

def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        settings = get_settings()
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info("Embedding model loaded")
    return _embedding_model

class Embedder:
    def __init__(self):
        self.model = get_embedding_model()
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_chunks(self, chunks: list[dict]) -> np.ndarray:
        # Extract the "content" field from each chunk dict
        content = [c['content'] for c in chunks]
        # Encode all texts in one batch using self.model.encode()
        # normalize_embeddings=True is critical — required for cosine similarity via dot product
        encoded_content = self.model.encode(content,normalize_embeddings=True)
        # Return shape: (len(chunks), dimension) float32 array
        return encoded_content.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        prefix =  "Represent this sentence for searching relevant passages: "
        query = query.strip()
        # BGE models require a special prefix for query embedding
        # (different from chunk embedding — this is a BGE-specific requirement)
        # Prefix: "Represent this sentence for searching relevant passages: "
        # normalize_embeddings=True
        encoded_query = self.model.encode(prefix + query, normalize_embeddings=True )
        # Return shape: (1, dimension) float32 array
        return encoded_query.reshape(1, -1).astype(np.float32)
    

if __name__ == '__main__':
    embedder = Embedder()
    print(f"Model dimension: {embedder.dimension}")

    chunks = [
        {"content": "FAISS is a library for efficient similarity search.", "document_name": "test.txt", "chunk_index": 0, "word_count": 9},
        {"content": "BM25 is a lexical ranking function used in information retrieval.", "document_name": "test.txt", "chunk_index": 1, "word_count": 11},
    ]

    chunk_vectors = embedder.embed_chunks(chunks)
    print(f"Chunk vectors shape: {chunk_vectors.shape}")
    print(f"Chunk vectors dtype: {chunk_vectors.dtype}")

    query_vector = embedder.embed_query("What is FAISS?")
    print(f"Query vector shape:  {query_vector.shape}")
    print(f"Query vector dtype:  {query_vector.dtype}")
