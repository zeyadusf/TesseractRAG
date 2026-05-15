from __future__ import annotations

from abc import ABC,abstractmethod
from backend.core import get_config

class QueryRewriteBase(ABC):

    def __init__(self):
        self.config = get_config()

    @abstractmethod
    async def aclose(self) -> None: pass
    
    @abstractmethod
    async def rewrite(self, query: str) -> str:pass
    
    @abstractmethod
    async def expand(self, query: str) -> str:pass
    
    @abstractmethod
    async def rewrite_and_expand(self, query: str) -> dict[str, str]:pass


