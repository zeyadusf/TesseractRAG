from .base_evaluator import BaseEvaluator
from .cohere_evaluator import CohereRAGEvaluator

from typing import List, Dict, Any, Optional


class EvaluatorDispatcher(BaseEvaluator):

    def __init__(self, provider: str = "cohere"):
        super().__init__()

        self._providers = {
            "cohere": CohereRAGEvaluator,
        }

        factory = self._providers.get(provider, self._providers["cohere"])
        self._instance = factory()

    async def evaluate(
        self,
        query: str,
        response_text: str,
        contexts: List[str],
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        return await self._instance.evaluate(
            query=query,
            response_text=response_text,
            contexts=contexts,
            model=model,
        )

    async def aclose(self):
        return await self._instance.aclose()
