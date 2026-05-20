from __future__ import annotations
from typing import Optional,List,Dict

from abc import ABC,abstractmethod
from backend.core import get_config

class QueryRewriteBase(ABC):

    def __init__(self):
        self.config = get_config()

    @abstractmethod
    async def aclose(self) -> None: pass
    
    @abstractmethod
    async def rewrite(self, query: str,history:Optional[List[Dict[str, str]]] = None) -> str:pass
    
    @abstractmethod
    async def expand(self, query: str,history:Optional[List[Dict[str, str]]] = None) -> str:pass
    
    @abstractmethod
    async def rewrite_and_expand(self, query: str,history:Optional[List[Dict[str, str]]] = None) -> dict[str, str]:pass


