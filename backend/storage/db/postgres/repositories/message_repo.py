from __future__  import annotations
from .base_repo import BaseRepository
from backend.storage.db.postgres.schemas import Message
from backend.models.enums.message_role import MessageRole 
from typing import List
from sqlalchemy import func, select, update,asc, desc
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from backend.storage.db.postgres.schemas.evaluation import Evaluation
from backend.storage.db.postgres.schemas.session import Session


class MessageRepository(BaseRepository[Message]):
    def __init__(self,sessionConn:AsyncSession):
        self._s = sessionConn

    async def create_user_message(
        self,
        session_id: UUID,
        content: str,
    ) -> Message:
        """Convenience method for creating a user turn message."""
        return await self.create(
            session_id=session_id,
            role=MessageRole.USER,
            content=content,
        )

    async def create_assistant_message(
        self,
        session_id: UUID,
        content: str,
        retrieval_strategy: str | None = None,
        source_chunks: list[dict] | None = None,
        llm_model: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        latency_ms: int | None = None,
        embedding_model: str | None = None,
        retrieval_latency_ms: int | None = None,
    ) -> Message:
        """
        Convenience method for creating an assistant turn message.
        source_chunks is a JSONB snapshot:
        [
            {
                "chunk_id":    "uuid-string",
                "content":     "the chunk text",
                "score":       0.94,
                "source_doc":  "paper.pdf",
                "chunk_index": 12,
            },
            ...
        ]
        """
        return await self.create(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=content,
            retrieval_strategy=retrieval_strategy,
            source_chunks=source_chunks or [],
            llm_model=llm_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            embedding_model=embedding_model,
            retrieval_latency_ms=retrieval_latency_ms,
        )

    async def create(
        self,
        *,
        session_id: UUID,
        role: MessageRole | str,
        content: str,
        retrieval_strategy: str | None = None,
        source_chunks: List[dict] | None = None,
        llm_model: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        latency_ms: int | None = None,
        embedding_model: str | None = None,
        retrieval_latency_ms: int | None = None,
    ) -> Message:
        
        obj = Message(
            session_id=session_id,
            role=role if isinstance(role, str) else role.value,
            content=content,
            retrieval_strategy=retrieval_strategy,
            source_chunks=source_chunks or [],
            llm_model=llm_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            embedding_model=embedding_model,
            retrieval_latency_ms=retrieval_latency_ms,
        )
        self._s.add(obj)
        await self._s.flush()
        await self._s.refresh(obj)
        return obj

    async def update(self, record_id: UUID, **kwargs) -> Message | None:
        obj = await self.get_by_id(record_id)
        if obj is None:
            return None
        for field, value in kwargs.items():
            setattr(obj, field, value)
        await self._s.flush()
        await self._s.refresh(obj)
        return obj

    async def delete(self, record_id: UUID) -> bool:
        obj = await self.get_by_id(record_id)
        if obj is None:
            return False
        await self._s.delete(obj)
        await self._s.flush()
        return True


    async def get_by_id(self, record_id: UUID) -> Message | None:
        result = await self._s.execute(
            select(Message).where(Message.id == record_id)
        )
        return result.scalar_one_or_none()

    async def list_by_session(
        self,
        session_id: UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Message]:
        """Return messages in chronological order (oldest first). Supports pagination."""
        result = await self._s.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.asc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def get_recent_history(
        self, session_id: UUID, n_exchanges: int = 3
    ) -> list[Message]:
        """
        Return the last N user+assistant pairs for prompt context injection.

        We fetch newest-first (n_exchanges * 2 rows) then reverse to get
        chronological order. This is what the prompt builder needs.

        n_exchanges=3 → up to 6 messages (3 user + 3 assistant).
        """
        result = await self._s.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.desc())
            .limit(n_exchanges * 2)
        )
        msgs = list(result.scalars().all())
        return list(reversed(msgs))  # restore chronological order

    async def count_by_session(self, session_id: UUID) -> int:
        result = await self._s.execute(
            select(func.count(Message.id)).where(Message.session_id == session_id)
        )
        return result.scalar_one()

    async def count_by_role(self, session_id: UUID, role: MessageRole) -> int:
        result = await self._s.execute(
            select(func.count(Message.id)).where(
                Message.session_id == session_id,
                Message.role == role,
            )
        )
        return result.scalar_one()
    
    async def get_message_for_evaluation(self,session_id:UUID,
                                message_id:UUID,role:MessageRole=MessageRole.ASSISTANT)->Message:
        """to use in evlautor service"""
        stmt = select(Message).where(
            Message.session_id == session_id,
            Message.id == message_id,
            Message.role == role
        )
        
        result = await self._s.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_preceding_user_message(
        self, session_id: UUID, before_message_id: UUID
    ) -> Message | None:
        """
        Return the most recent USER message that was created
        before the message identified by before_message_id.
        """
        # First get the reference message's timestamp
        ref = await self.get_by_id(before_message_id)
        if ref is None:
            return None

        result = await self._s.execute(
            select(Message)
            .where(
                Message.session_id == session_id,
                Message.role == MessageRole.USER.value,
                Message.created_at < ref.created_at,
            )
            .order_by(Message.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def count_all(self) -> int:
        """Total number of messages across all sessions."""
        result = await self._s.execute(select(func.count()).select_from(Message))
        return result.scalar_one()
    
    async def get_all_with_eval(
        self,
        user_id=None,           # UUID | None  — filters via session.user_id
        session_id=None,        # UUID | None
        is_active=None,         # bool | None  — filters on session.is_active
        role=None,              # MessageRole | None
        order_by: str = "created_at",
        order_dir: str = "desc",
    ) -> list:
        """
        Returns messages joined with their evaluation scores and session.user_id.
    
        order_by options:
            created_at, faithfulness, answer_relevancy,
            context_precision, context_recall
        """
        stmt = (
            select(
                Message,
                Session.user_id.label("user_id"),
                Evaluation.faithfulness.label("faithfulness"),
                Evaluation.answer_relevancy.label("answer_relevancy"),
                Evaluation.context_precision.label("context_precision"),
                Evaluation.context_recall.label("context_recall"),
            )
            .join(Session, Message.session_id == Session.id)
            .outerjoin(Evaluation, Evaluation.message_id == Message.id)
        )
    
        if user_id is not None:
            stmt = stmt.where(Session.user_id == user_id)
        if session_id is not None:
            stmt = stmt.where(Message.session_id == session_id)
        if is_active is not None:
            stmt = stmt.where(Session.is_active == is_active)
        if role is not None:
            stmt = stmt.where(Message.role == role)
    
        # ── ordering ──────────────────────────────────────────────────────────
        _dir = desc if order_dir == "desc" else asc
    
        _eval_col = {
            "faithfulness": Evaluation.faithfulness,
            "answer_relevancy": Evaluation.answer_relevancy,
            "context_precision": Evaluation.context_precision,
            "context_recall": Evaluation.context_recall,
        }
    
        if order_by in _eval_col:
            # NULLs last when ascending, NULLs first when descending is the
            # default PostgreSQL behaviour — acceptable for admin views.
            stmt = stmt.order_by(_dir(_eval_col[order_by]))
        else:
            stmt = stmt.order_by(_dir(Message.created_at))
    
        result = await self._s.execute(stmt)
        return result.all()