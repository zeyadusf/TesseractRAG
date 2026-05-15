from abc import ABC, abstractmethod
from typing import Generator
from backend.models.metadata import Metadata, ChunkMetadata
from backend.core import get_config

class BaseChunker(ABC):

    def __init__(self):
        self.config = get_config()
        self.chunk_size = self.config.CHUNK_SIZE
        self.chunk_overlap = self.config.CHUNK_OVERLAP
        self.min_chunk_len = self.config.CHUNK_MIN_SIZE
    
    @abstractmethod
    def chunk(
        self,
        text: str,
        metadata: Metadata,
    ) -> Generator[ChunkMetadata, None, None]:
        """
        Split *text* into chunks and yield a ChunkMetadata per chunk.

        Parameters
        ----------
        text:
            Cleaned document text from the cleaner.
        metadata:
            Document-level metadata from the parser.
        """
        ...