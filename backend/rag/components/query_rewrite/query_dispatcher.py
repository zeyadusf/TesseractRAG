from .groq_query_rewriter import get_groq_rewriter

from .query_base import QueryRewriteBase


class QueryRewriteDispatcher(QueryRewriteBase):
    def __init__(self, provider: str = "groq"):  
        super().__init__()

        self._providers = {
            "groq": get_groq_rewriter,
        }

        factory = self._providers.get(provider, self._providers["groq"])
        self._instance = factory()

    async def aclose(self):
        await self._instance.aclose()

    async def rewrite(self, query: str) -> str:
        return await self._instance.rewrite(query)

    async def expand(self, query: str) -> str:
        return await self._instance.expand(query)

    async def rewrite_and_expand(self, query: str) -> dict:
        return await self._instance.rewrite_and_expand(query)