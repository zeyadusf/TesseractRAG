from .generator_base import AnswerGeneratorBase
from .groq_generator import get_groq_generator
from .hf_generator import get_hf_generator
from .guard import get_smart_guard       
from typing import List, Dict


class GeneratorDispatcher(AnswerGeneratorBase):
    """
    Dispatcher that selects generator based on config.
    Supports: 'groq', 'hf', 'smart_guard'
    """

    def __init__(self, provider: str | None = None):
        super().__init__()
        provider = provider or self.config.DEFAULT_GENERATOR_PROVIDER

        if provider == "smart_guard":
            self._instance = get_smart_guard( 
                groq_daily_limit=self.config.GENERATOR_GROQ_DAILY_LIMIT,
                soft_threshold_pct=self.config.GENERATOR_SOFT_THRESHOLD_PCT,
                hard_threshold_pct=self.config.GENERATOR_HARD_THRESHOLD_PCT,
            )
        elif provider == "hf":
            self._instance = get_hf_generator()
        else:
            self._instance = get_groq_generator()

    async def aclose(self):
        await self._instance.aclose()

    async def generate(
        self,
        question: str,
        context: str,
        sources: list[dict] | None = None,
        history: List[Dict[str, str]] | None = None,  # ← NEW

    ) -> str:
        return await self._instance.generate(question, context, sources,history)