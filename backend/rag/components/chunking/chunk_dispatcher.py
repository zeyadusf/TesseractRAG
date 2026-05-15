from .recursive_chunker.RecursiveChunker import RecursiveChunker
from backend.core import get_logger
from backend.models.metadata import Metadata
from .base_chunker import BaseChunker
from typing import Generator, Dict

logger = get_logger(__name__)


class ChunkerDispatcher(BaseChunker):

    def __init__(self, chunker_technique: str = None):
        super().__init__()

        self.supported_chunks = set(self.config.SUPPORTED_CHUNKS)

        self.chunker_technique = (
            chunker_technique
            if chunker_technique in self.supported_chunks
            else self.config.DEFAULT_CHUNK
        )

        self.chunkers = {
            "recursive": RecursiveChunker
        }

    def chunk(
        self,
        text: str,
        metadata: Metadata,
    ) -> Generator[Dict, None, None]:

        chunker_cls = self.chunkers.get(self.chunker_technique,
                                        self.chunkers[self.config.DEFAULT_CHUNK])
        chunker = chunker_cls()

        logger.info(f"Chunker technique : {self.chunker_technique}")

        yield from chunker.chunk(text, metadata)