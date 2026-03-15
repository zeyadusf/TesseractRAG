import faiss
import numpy as np
from pathlib import Path
from app.utils.logger import get_logger

logger = get_logger(__name__)

class FAISSIndexer:
    def __init__(self, dimension: int):
        self.dimension = dimension
        # IndexFlatIP = Inner Product (dot product)
        # With L2-normalized vectors, dot product = cosine similarity
        self.index = faiss.IndexFlatIP(dimension)
        logger.info(f'FAISSIndexer ready — dimension={self.dimension}, is_trained={self.index.is_trained}')

    def add(self, vectors: np.ndarray) -> None:
        # Add vectors to the index
        # vectors shape: (n, dimension)
        self.index.add(vectors.astype(np.float32))
        logger.info(f'indexer add {vectors.shape} vector(s)')

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        # Search for top_k nearest vectors
        scores, indices = self.index.search(query_vector.astype(np.float32),top_k)
        return (scores, indices)

    def save(self, path: Path) -> None:
        # Save index to disk as binary file
        faiss.write_index(self.index, str(path))
        logger.info(f'Save indexer to disk {str(path)}')

    def load(self, path: Path) -> None:
        # Load index from disk
        self.index = faiss.read_index(str(path))
        logger.info(f'Loaded indexer from disk {str(path)}')


    @property
    def total_vectors(self) -> int:
        # Return number of vectors currently in the index
        return self.index.ntotal
