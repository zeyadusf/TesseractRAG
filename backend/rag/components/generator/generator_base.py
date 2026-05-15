from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict

from backend.core import get_config


class AnswerGeneratorBase(ABC):
    def __init__(self):
        self.config = get_config()

    def _format_context_with_sources(self, context: str, sources: list[dict] | None) -> str:
        """
        Format context chunks with source numbers for citation.
        Assumes context is chunks separated by '\n\n'.
        """
        if not sources:
            return context

        chunks = [c.strip() for c in context.split("\n\n") if c.strip()]
        formatted = []

        for i, (chunk, src) in enumerate(zip(chunks, sources[:len(chunks)]), 1):
            title = src.get("title", src.get("source", f"Source {i}"))
            formatted.append(f"[Source {i}] {title}:\n{chunk}")

        return "\n\n".join(formatted)
    
    
    @abstractmethod
    async def generate(
        self,
        question: str,
        context: str,
        sources: list[dict] | None = None,
        history: List[Dict[str, str]] | None = None,  
    ) -> str:
        pass
    
    @abstractmethod
    async def aclose(self) -> None:
        pass
