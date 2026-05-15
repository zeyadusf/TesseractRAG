from abc import ABC, abstractmethod
from typing import List, AsyncGenerator, Union
from backend.core import get_config
from backend.models.embedding import EmbeddingChunk, EmbeddingMeta


class BaseEmbedder(ABC):

    def __init__(self):
        self.config = get_config()
        self.embed_dim = self.config.EMBED_DIM

    @abstractmethod
    async def embed_documents(
        self,
        texts: List[str],
        late_chunking: bool = False,
    ) -> AsyncGenerator[Union[EmbeddingChunk, EmbeddingMeta], None]:
        """Embed a list of document chunks for indexing."""
        ...

    @abstractmethod
    async def embed_query(self, query: str) -> List[float]:
        """Embed a single search query."""
        ...

    async def aclose(self) -> None:
        """Default no-op. Override if provider holds external resources."""
        pass