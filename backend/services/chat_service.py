# backend/services/chat_service.py
from __future__ import annotations

import time
from typing import List, Dict, Optional
from uuid import UUID, uuid4
from fastapi import HTTPException, status as http_status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core import get_logger, get_config
from backend.services.base_service import BaseService
from backend.storage.db.db_dispatcher import DBDispatcher
from backend.rag.pipelines.generation_pipeline import (
    ChatRequest,
    ChatResponse,
    SourceChunkOut,
    GenerationInput,
    GenerationPipeline,
)
from backend.models.chat import (
    ChatHistoryOut,
    ChatTurnOut,
)

from backend.models.enums.retrieval_strategy import RetrievalStrategy
from backend.rag.pipelines.retrieval_pipeline import RetrievalPipeline
from backend.rag.components.embedding.embed_dispatcher import EmbedderDispatcher
from backend.rag.components.query_rewrite.query_dispatcher import QueryRewriteDispatcher
from backend.storage.vector_db.vec_dispatcher import get_vector_store
from backend.rag.components.retrievals.cross_language_strategy import (
    get_cross_language_selector,
)

logger = get_logger(__name__)
config = get_config()


class ChatService(BaseService):
    def __init__(
        self,
        db: DBDispatcher,
        session: AsyncSession,
        embedder: Optional[EmbedderDispatcher] = None,
        retrieval_pipeline: Optional[RetrievalPipeline] = None,
        generation_pipeline: Optional[GenerationPipeline] = None,
        query_rewriter: Optional[QueryRewriteDispatcher] = None,
    ) -> None:
        super().__init__(db)
        self._session = session
        self._embedder = embedder or EmbedderDispatcher()
        self._retrieval = retrieval_pipeline or RetrievalPipeline()
        self._generation = generation_pipeline or GenerationPipeline(enable_rewrite=False)
        self._query_rewriter = query_rewriter or QueryRewriteDispatcher()

        self._llm_model = config.DEFAULT_GENERATOR_PROVIDER
        self._embedding_model = config.DEFAULT_EMBED

    # ── Public API ─────────────────────────────────────────────────────────────

    async def chat(self, session_id: UUID, request: ChatRequest) -> ChatResponse:
        total_start = time.perf_counter()

        # ── 0. Validate session ────────────────────────────────────────────
        session = await self.db.sessions.get_by_id(session_id)
        if session is None:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found.",
            )
        if not session.is_active:
            raise HTTPException(
                status_code=http_status.HTTP_403_FORBIDDEN,
                detail="This session has been deactivated.",
            )

        # ── 1. Persist user message ────────────────────────────────────────────
        message = await self.db.messages.create_user_message(
            session_id=session_id,
            content=request.question,
        )

        # ── 1.5 Fetch last N turns for history context ─────────────────────────
        all_messages = await self.db.messages.list_by_session(
            session_id=session_id,
            limit=config.HISTORY_TURNS * 2 + 1,
            offset=0,
        )

        history_messages: List[Dict[str, str]] = []
        for msg in all_messages:
            if str(msg.id) == str(message.id):
                continue
            history_messages.append({
                "role": msg.role,
                "content": msg.content,
            })
        history_messages = history_messages[-(config.HISTORY_TURNS * 2):]

        logger.info(
            f"[CHAT] history loaded | turns={len(history_messages)//2} "
            f"| messages={len(history_messages)}"
        )

        # ── 2. Query Rewrite ───────────────────────────────────────────────────
        rewrite_result = await self._query_rewriter.rewrite_and_expand(
            request.question,
            history=history_messages,
        )
        rewritten_query = rewrite_result["rewritten"] or request.question
        expanded_query  = rewrite_result["expanded"]  or request.question

        logger.info(
            f"[REWRITE] original='{request.question}' "
            f"| rewritten='{rewritten_query}' "
            f"| expanded='{expanded_query}'"
        )

        # ── 3. Embed rewritten query ───────────────────────────────────────────
        query_vector = await self._embedder.embed_query(expanded_query)

        # ── 4. Fetch candidate chunks ──────────────────────────────────────────
        retrieval_start = time.perf_counter()

        raw_chunks = await self.db.chunks.list_by_session(session_id=session_id)

        chunks: List[Dict] = [
            {
                "chunk_id":    str(c.id),
                "content":     c.content,
                "chunk_index": c.chunk_index,
                "document_id": str(c.document_id),
                "source_doc":  str(c.document_id),
                "metadata":    c.metadata or {},
            }
            for c in raw_chunks
        ]

        # ── 5. Vector search ───────────────────────────────────────────────────
        vector_store = get_vector_store(session=self._session)

        vector_results: List[Dict] = await vector_store.search(
            session_id=session_id,
            query_vector=query_vector,
            model_name=self._embedding_model,
            top_k=20,
        )

        # ── Cross-language strategy override ───────────────────────────────────
        effective_strategy = request.strategy
        cross_lang_selector = get_cross_language_selector()

        if cross_lang_selector.should_force_semantic(request.question, chunks):
            if request.strategy != RetrievalStrategy.SEMANTIC:
                logger.info(
                    f"[STRATEGY] Cross-language override: "
                    f"'{request.strategy}' → 'semantic' for query: {request.question[:50]}..."
                )
            effective_strategy = RetrievalStrategy.SEMANTIC

        retrieval_latency_ms = int((time.perf_counter() - retrieval_start) * 1000)
        logger.info(
            f"[CHAT] retrieval done | strategy={effective_strategy.value} "
            f"| chunks={len(chunks)} | vector_hits={len(vector_results)} "
            f"| latency={retrieval_latency_ms}ms"
        )

        # ── 6. Retrieval pipeline ──────────────────────────────────────────────
        ranked_chunks: List[Dict] = await self._retrieval.run(
            query=rewritten_query,
            query_vector=query_vector,
            chunks=chunks,
            vector_results=vector_results,
            strategy=effective_strategy.value,
        )

        raw_chunks_by_id = {str(c.get("chunk_id")): c for c in chunks}

        normalized_chunks = []
        for rc in ranked_chunks:
            cid = str(rc.get("chunk_id", ""))
            base = raw_chunks_by_id.get(cid, {})
            normalized_chunks.append({
                **base,
                "chunk_id": cid,
                "score": float(rc.get("rrf_score") or rc.get("score") or 0.0),
                "document_name": (
                    base.get("document_name")
                    or base.get("source_doc")
                    or base.get("document_id")
                    or "Unknown"
                ),
            })

        # ── 7. Generation pipeline ─────────────────────────────────────────────
        gen_output = await self._generation.run(
            GenerationInput(
                original_query=request.question,
                query_vector=query_vector,
                chunks=normalized_chunks,
                vector_results=vector_results,
                strategy=effective_strategy,
                history=history_messages,
            )
        )

        # ── 8. Persist assistant message ───────────────────────────────────────
        source_chunks_for_db = [
            {
                "chunk_id":    str(c.get("chunk_id", "")),
                "content":     c.get("content", ""),
                "score":       float(c.get("score") or 0.0),
                "source_doc":  str(c.get("source_doc") or c.get("document_name") or ""),
                "chunk_index": int(c.get("chunk_index") or 0),
            }
            for c in normalized_chunks
        ]

        assistant_message = await self.db.messages.create_assistant_message(
            session_id=session_id,
            content=gen_output.answer,
            retrieval_strategy=effective_strategy.value,
            source_chunks=source_chunks_for_db,
            llm_model=self._llm_model,
            latency_ms=int(gen_output.timing.get("total", 0) * 1000),
            embedding_model=self._embedding_model,
            retrieval_latency_ms=retrieval_latency_ms,
        )
        await self.db.session.commit()

        # ── 9. Map → ChatResponse ──────────────────────────────────────────────
        chunks_lookup = {
            int(c.get("chunk_index", -1)): c
            for c in normalized_chunks
        }
        total_latency_ms = int((time.perf_counter() - total_start) * 1000)

        return self._generation.to_chat_response(
            message_id=assistant_message.id,
            request=request,
            output=gen_output,
            retrieval_latency_ms=retrieval_latency_ms,
            total_latency_ms=total_latency_ms,
            llm_model=self._llm_model,
            embedding_model=self._embedding_model,
            chunks_lookup=chunks_lookup,
        )

    async def aclose(self) -> None:
        await self._generation.aclose()
        await self._retrieval.aclose()
        await self._query_rewriter.aclose()
        logger.info("[CHAT] ChatService closed")

    # ── History / turn helpers ─────────────────────────────────────────────────

    async def list_history(
        self,
        session_id: UUID,
        user_id: UUID,
        page: int = 1,
        page_size: int = 20,
    ) -> ChatHistoryOut:
        await self._assert_session_owner(session_id, user_id)

        total = await self.db.messages.count_by_session(session_id)

        offset = (page - 1) * page_size
        messages = await self.db.messages.list_by_session(
            session_id=session_id,
            limit=page_size,
            offset=offset,
        )

        turns = [self._message_to_turn_out(m) for m in messages]

        return ChatHistoryOut(
            session_id=session_id,
            total=total,
            page=page,
            page_size=page_size,
            turns=turns,
        )

    async def get_turn(
        self,
        session_id: UUID,
        turn_id: UUID,
        user_id: UUID,
    ) -> ChatTurnOut:
        await self._assert_session_owner(session_id, user_id)

        message = await self.db.messages.get_by_id(turn_id)

        if message is None:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Turn {turn_id} not found.",
            )

        if str(message.session_id) != str(session_id):
            raise HTTPException(
                status_code=http_status.HTTP_403_FORBIDDEN,
                detail="Turn does not belong to this session.",
            )

        return self._message_to_turn_out(message)

    async def clear_history(
        self,
        session_id: UUID,
        user_id: UUID,
    ) -> None:
        await self._assert_session_owner(session_id, user_id)

        messages = await self.db.messages.list_by_session(session_id=session_id)

        for message in messages:
            await self.db.messages.delete(message.id)

        logger.info(f"[CHAT] cleared {len(messages)} messages | session={session_id}")

    # ── Private helpers ────────────────────────────────────────────────────────

    async def _assert_session_owner(self, session_id: UUID, user_id: UUID) -> None:
        session = await self.db.sessions.get_by_id(session_id)

        if session is None:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found.",
            )

        if str(session.user_id) != str(user_id):
            raise HTTPException(
                status_code=http_status.HTTP_403_FORBIDDEN,
                detail="Access to this session is forbidden.",
            )

        if not session.is_active:
            raise HTTPException(
                status_code=http_status.HTTP_403_FORBIDDEN,
                detail="This session has been deactivated.",
            )

    @staticmethod
    def _message_to_turn_out(message) -> ChatTurnOut:
        return ChatTurnOut(
            id=message.id,
            session_id=message.session_id,
            role=message.role,
            content=message.content,
            created_at=message.created_at,
            retrieval_strategy=message.retrieval_strategy,
            source_chunks=message.source_chunks or [],
            llm_model=message.llm_model,
            latency_ms=message.latency_ms,
        )