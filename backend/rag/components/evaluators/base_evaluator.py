from abc import ABC,abstractmethod
from typing import List,Any,Dict,Optional
from backend.core import get_config
class BaseEvaluator(ABC):
    def __init__(self):
        self._settings = get_config()

    @abstractmethod
    async def evaluate(
        self,
        query: str,
        response_text: str,
        contexts: List[str],
        model: Optional[str] = None
    ) -> Dict[str, Any]: pass

    @abstractmethod
    async def aclose(self) -> None:
        pass