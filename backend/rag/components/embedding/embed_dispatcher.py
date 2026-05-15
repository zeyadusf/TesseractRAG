from .providers.jina_embedder import JinaProvider
from .base_embedder import BaseEmbedder
from backend.core import get_logger
from backend.models.embedding import EmbeddingChunk, EmbeddingMeta
from typing import List, AsyncGenerator, Union


class EmbedderDispatcher(BaseEmbedder):

    _EMBEDDERS = {
        "jina": JinaProvider,
    }

    def __init__(self, embed_model_name: str = None):
        super().__init__() 
        self.logger = get_logger(__name__)

        self.cur_embed_model = (
            embed_model_name.lower() if embed_model_name else self.config.DEFAULT_EMBED
        )
        self._embedder_instance = self._get_instance()

    def _get_instance(self) -> BaseEmbedder:
        supported = set(self.config.SUPPORTED_EMBED)

        if self.cur_embed_model not in supported:
            self.logger.warning(
                "[EmbedderDispatcher] Unsupported embedder requested: '%s'",
                self.cur_embed_model,
            )
            raise ValueError(
                f"Unsupported embedder: '{self.cur_embed_model}'. "
                f"Supported: {sorted(supported)}"
            )

        embedder_cls = self._EMBEDDERS.get(self.cur_embed_model)
        if not embedder_cls:
            # supported_embed and _EMBEDDERS are out of sync — developer error
            self.logger.critical(
                "[EmbedderDispatcher] No class registered for supported model '%s'",
                self.cur_embed_model,
            )
            raise RuntimeError(
                f"Embedder class missing for model '{self.cur_embed_model}'. "
                "Update EmbedderDispatcher._EMBEDDERS."
            )

        self.logger.debug(
            "[EmbedderDispatcher] Instantiating %s", embedder_cls.__name__
        )
        return embedder_cls()

    async def embed_documents(self,texts: List[str],late_chunking: bool = False,
                    ) -> AsyncGenerator[Union[EmbeddingChunk, EmbeddingMeta], None]:
        
        async for item in self._embedder_instance.embed_documents(texts, late_chunking=late_chunking):
            yield item

    async def embed_query(self, query: str) -> List[float]:
        return await self._embedder_instance.embed_query(query)

    async def aclose(self) -> None:
        """Close underlying provider resources (e.g., HTTP clients). Safe to call multiple times."""
        await self._embedder_instance.aclose()