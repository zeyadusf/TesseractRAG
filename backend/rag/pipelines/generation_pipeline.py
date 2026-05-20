from __future__ import annotations

import time
import asyncio
from typing import List, Dict, Optional
from uuid import UUID
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from backend.core import get_logger
from backend.rag.components.context.context_builder import ContextBuilder
from backend.rag.components.generator.generator_dispatcher import GeneratorDispatcher

logger = get_logger(__name__)

from backend.models.chat import ChatRequest, ChatResponse, SourceChunkOut
from backend.models.enums.retrieval_strategy import RetrievalStrategy


class GenerationInput(BaseModel):
    original_query: str
    query_vector: Optional[List[float]] = None
    chunks: List[Dict]
    vector_results: Optional[List[Dict]] = None
    strategy: RetrievalStrategy = RetrievalStrategy.AUTO
    history: Optional[List[Dict[str, str]]] = None


class GenerationOutput(BaseModel):
    answer: str
    sources: List[Dict]
    context_stats: Dict
    timing: Dict[str, float]
    metadata: Dict


class GenerationPipeline:
    """Orchestrates context building and answer generation."""

    _executor = ThreadPoolExecutor(max_workers=4)

    def __init__(
        self,
        *,
        context_builder: Optional[ContextBuilder] = None,
        generator: Optional[GeneratorDispatcher] = None,
    ):
        self._context_builder = context_builder or ContextBuilder()
        self._generator = generator or GeneratorDispatcher()

    async def run(self, input_data: GenerationInput) -> GenerationOutput:
        start_time = time.perf_counter()
        timing: Dict[str, float] = {}

        original_query = input_data.original_query
        logger.info(f"[GENERATION] Starting pipeline for query: {original_query[:60]}...")

        # Stage 1: Context Building
        context_str, context_stats, timing["context_build"] = \
            await self._run_context_building(input_data.chunks)

        # Stage 2: Answer Generation
        answer, timing["generate"] = \
            await self._run_answer_generation(
                question=original_query,
                context=context_str,
                context_stats=context_stats,
                history=input_data.history,
            )

        timing["total"] = time.perf_counter() - start_time

        logger.info(
            f"[GENERATION] Complete in {timing['total'] * 1000:.0f}ms | "
            f"chunks: {context_stats.get('chunks_used', 0)} | "
            f"answer_len: {len(answer)}"
        )

        return self._assemble_output(
            answer=answer,
            original_query=original_query,
            context_stats=context_stats,
            timing=timing,
        )

    async def _run_context_building(
        self,
        chunks: List[Dict],
    ) -> tuple[str, Dict, float]:
        start = time.perf_counter()
        context_str = ""
        context_stats: Dict = {}

        try:
            loop = asyncio.get_running_loop()
            context_str, context_stats = await loop.run_in_executor(
                self._executor,
                lambda: self._context_builder.build(chunks)
            )

            if not context_str.strip():
                logger.warning("[CONTEXT] No valid chunks to build context.")
                context_str = "No relevant documents found for this query."
                context_stats = {"chunks_used": 0, "total_chars": 0, "sources": []}

        except Exception as exc:
            logger.error(f"[CONTEXT] Build failed: {exc}")
            context_str = "Error building context from retrieved documents."
            context_stats = {"chunks_used": 0, "total_chars": 0, "sources": [], "error": str(exc)}

        return context_str, context_stats, time.perf_counter() - start

    async def _run_answer_generation(
        self,
        question: str,
        context: str,
        context_stats: Dict,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> tuple[str, float]:
        start = time.perf_counter()
        try:
            answer = await self._generator.generate(
                question=question,
                context=context,
                sources=context_stats.get("sources"),
                history=history,
            )
        except Exception as exc:
            logger.error(f"[GENERATE] Failed: {exc}")
            answer = "Sorry, I encountered an error while generating the answer. Please try again."

        return answer, time.perf_counter() - start

    def _assemble_output(
        self,
        *,
        answer: str,
        original_query: str,
        context_stats: Dict,
        timing: Dict[str, float],
    ) -> GenerationOutput:
        return GenerationOutput(
            answer=answer,
            sources=context_stats.get("sources", []),
            context_stats={
                "chunks_used": context_stats.get("chunks_used", 0),
                "total_chars": context_stats.get("total_chars", 0),
                "chunks_deduped": context_stats.get("chunks_deduped", 0),
            },
            timing=timing,
            metadata={
                "original_query": original_query,
                "generator_provider": getattr(self._generator, "_instance", None).__class__.__name__
                    if hasattr(self._generator, "_instance") else "unknown",
            },
        )

    def to_chat_response(
        self,
        message_id: UUID,
        request: ChatRequest,
        output: GenerationOutput,
        retrieval_latency_ms: int,
        total_latency_ms: int,
        llm_model: str,
        embedding_model: str,
        chunks_lookup: dict | None = None,
    ) -> ChatResponse:

        lookup = chunks_lookup or {}

        sources = [
            SourceChunkOut(
                chunk_id=str(src.get("chunk_id") or src.get("id") or ""),
                content=lookup.get(src.get("chunk_index"), {}).get("content", ""),
                score=float(lookup.get(src.get("chunk_index"), {}).get("score") or 0.0),
                source_doc=str(
                    src.get("source_doc")
                    or src.get("document_name")
                    or lookup.get(src.get("chunk_index"), {}).get("source_doc", "")
                    or ""
                ),
                chunk_index=int(src.get("chunk_index") or 0),
            )
            for src in output.sources
            if src and (src.get("chunk_id") or src.get("id"))
        ]

        return ChatResponse(
            message_id=message_id,
            question=request.question,
            answer=output.answer,
            strategy_used=request.strategy.value,
            sources=sources if request.show_context else [],
            retrieval_latency_ms=retrieval_latency_ms,
            total_latency_ms=total_latency_ms,
            llm_model=llm_model,
            embedding_model=embedding_model,
        )

    async def aclose(self) -> None:
        await self._generator.aclose()
        self._executor.shutdown(wait=True)