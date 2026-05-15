from __future__ import annotations

from uuid import UUID

from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func

from backend.storage.db.postgres.repositories.base_repo import BaseRepository
from backend.storage.db.postgres.schemas import User
from backend.storage.db.postgres.schemas.session import Session
from backend.storage.db.postgres.schemas.message import Message
from backend.storage.db.postgres.schemas.document import Document
from backend.storage.db.postgres.schemas.evaluation import Evaluation


class UserRepository(BaseRepository[User]):
    def __init__(self, sessionConn: AsyncSession):
        self._s = sessionConn

    # ── CRUD ───────────────────────────────────────────────────────────────

    async def create(
        self,
        email: str,
        username: str,
        hashed_password: str,
        is_active: bool = True,
        is_superuser: bool = False,
    ) -> User:
        user = User(
            email=email,
            username=username,
            hashed_password=hashed_password,
            is_active=is_active,
            is_superuser=is_superuser,
        )
        self._s.add(user)
        await self._s.flush()
        await self._s.refresh(user)
        return user

    async def get_by_id(self, record_id: UUID) -> User | None:
        result = await self._s.execute(select(User).where(User.id == record_id))
        return result.scalar_one_or_none()

    async def get_by_username(self, username: str) -> User | None:
        result = await self._s.execute(
            select(User).where(User.username == username)
        )
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> User | None:
        result = await self._s.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    async def update(self, record_id: UUID, **kwargs) -> User | None:
        user = await self.get_by_id(record_id)
        if user is None:
            return None
        for key, value in kwargs.items():
            setattr(user, key, value)
        await self._s.flush()
        await self._s.refresh(user)
        return user

    async def deactivate(self, record_id: UUID) -> User | None:
        """Soft-delete: marks the user inactive instead of removing the row."""
        return await self.update(record_id, is_active=False)

    async def delete(self, record_id: UUID) -> bool:
        user = await self.get_by_id(record_id)
        if user is None:
            return False
        await self._s.delete(user)
        await self._s.flush()
        return True

    # ── Lookups ────────────────────────────────────────────────────────────

    async def is_email_exists(self, email: str) -> bool:
        result = await self._s.execute(
            select(User.id).where(User.email == email)
        )
        return result.scalar_one_or_none() is not None

    async def is_username_exists(self, username: str) -> bool:
        result = await self._s.execute(
            select(User.id).where(User.username == username)
        )
        return result.scalar_one_or_none() is not None

    # ── Admin: global counts ───────────────────────────────────────────────
    async def count_all_superusers(self) -> int:
        """Total number of users with is_superuser = True."""
        result = await self._s.execute(
            select(func.count()).select_from(User).where(User.is_superuser.is_(True))
        )
        return result.scalar_one()
    async def count_all(self) -> int:
        """Total number of registered users."""
        result = await self._s.execute(select(func.count()).select_from(User))
        return result.scalar_one()

    # ── Admin: per-user stats (single user) 

    async def get_user_stats(self, user_id: UUID) -> dict:
        """
        Returns aggregated stats for one user:
            session_count, message_count, document_count,
            avg_faithfulness, avg_answer_relevancy,
            avg_context_precision, avg_context_recall
        All avg_* fields are None if no evaluations exist yet.
        """
        # ── counts via correlated subqueries ─────────────────────────────
        session_count_sq = (
            select(func.count(Session.id))
            .where(Session.user_id == user_id)
            .scalar_subquery()
        )
        message_count_sq = (
            select(func.count(Message.id))
            .join(Session, Message.session_id == Session.id)
            .where(Session.user_id == user_id)
            .scalar_subquery()
        )
        document_count_sq = (
            select(func.count(Document.id))
            .join(Session, Document.session_id == Session.id)
            .where(Session.user_id == user_id)
            .scalar_subquery()
        )

        # ── avg evaluation scores ─────────────────────────────────────────
        # Evaluations are linked to messages → sessions → user
        eval_base = (
            select(Evaluation)
            .join(Message, Evaluation.message_id == Message.id)
            .join(Session, Message.session_id == Session.id)
            .where(Session.user_id == user_id)
            .subquery()
        )
        avg_sq = select(
            func.avg(eval_base.c.faithfulness).label("avg_faithfulness"),
            func.avg(eval_base.c.answer_relevancy).label("avg_answer_relevancy"),
            func.avg(eval_base.c.context_precision).label("avg_context_precision"),
            func.avg(eval_base.c.context_recall).label("avg_context_recall"),
        )

        # ── execute both in one round-trip ────────────────────────────────
        counts_result = await self._s.execute(
            select(
                session_count_sq.label("session_count"),
                message_count_sq.label("message_count"),
                document_count_sq.label("document_count"),
            )
        )
        counts_row = counts_result.one()

        avg_result = await self._s.execute(avg_sq)
        avg_row = avg_result.one()

        return {
            "session_count": counts_row.session_count,
            "message_count": counts_row.message_count,
            "document_count": counts_row.document_count,
            "avg_faithfulness": (
                float(avg_row.avg_faithfulness)
                if avg_row.avg_faithfulness is not None
                else None
            ),
            "avg_answer_relevancy": (
                float(avg_row.avg_answer_relevancy)
                if avg_row.avg_answer_relevancy is not None
                else None
            ),
            "avg_context_precision": (
                float(avg_row.avg_context_precision)
                if avg_row.avg_context_precision is not None
                else None
            ),
            "avg_context_recall": (
                float(avg_row.avg_context_recall)
                if avg_row.avg_context_recall is not None
                else None
            ),
        }

    # ── Admin: all users with stats 

    async def get_all_with_stats(self) -> list[dict]:
        """
        Returns every user row joined with aggregated stats.
        Uses a single GROUP BY query instead of N+1 per-user calls.

        Returns a list of dicts; each dict contains all User scalar fields
        plus the same stat keys as get_user_stats().
        """
        # Sub-select: session counts per user
        session_counts = (
            select(
                Session.user_id.label("uid"),
                func.count(Session.id).label("session_count"),
            )
            .group_by(Session.user_id)
            .subquery()
        )

        # Sub-select: message counts per user (via sessions)
        message_counts = (
            select(
                Session.user_id.label("uid"),
                func.count(Message.id).label("message_count"),
            )
            .join(Message, Message.session_id == Session.id)
            .group_by(Session.user_id)
            .subquery()
        )

        # Sub-select: document counts per user (via sessions)
        document_counts = (
            select(
                Session.user_id.label("uid"),
                func.count(Document.id).label("document_count"),
            )
            .join(Document, Document.session_id == Session.id)
            .group_by(Session.user_id)
            .subquery()
        )

        # Sub-select: avg evaluation scores per user
        eval_scores = (
            select(
                Session.user_id.label("uid"),
                func.avg(Evaluation.faithfulness).label("avg_faithfulness"),
                func.avg(Evaluation.answer_relevancy).label("avg_answer_relevancy"),
                func.avg(Evaluation.context_precision).label("avg_context_precision"),
                func.avg(Evaluation.context_recall).label("avg_context_recall"),
            )
            .join(Message, Evaluation.message_id == Message.id)
            .join(Session, Message.session_id == Session.id)
            .group_by(Session.user_id)
            .subquery()
        )

        stmt = (
            select(
                User,
                func.coalesce(session_counts.c.session_count, 0).label("session_count"),
                func.coalesce(message_counts.c.message_count, 0).label("message_count"),
                func.coalesce(document_counts.c.document_count, 0).label("document_count"),
                eval_scores.c.avg_faithfulness,
                eval_scores.c.avg_answer_relevancy,
                eval_scores.c.avg_context_precision,
                eval_scores.c.avg_context_recall,
            )
            .outerjoin(session_counts, session_counts.c.uid == User.id)
            .outerjoin(message_counts, message_counts.c.uid == User.id)
            .outerjoin(document_counts, document_counts.c.uid == User.id)
            .outerjoin(eval_scores, eval_scores.c.uid == User.id)
            .order_by(User.created_at.desc())
        )

        result = await self._s.execute(stmt)
        rows = result.all()

        def _to_float(val) -> float | None:
            return float(val) if val is not None else None

        return [
            {
                # ORM object lives at index 0 of the Row tuple
                "user": row[0],
                "session_count": row.session_count,
                "message_count": row.message_count,
                "document_count": row.document_count,
                "avg_faithfulness": _to_float(row.avg_faithfulness),
                "avg_answer_relevancy": _to_float(row.avg_answer_relevancy),
                "avg_context_precision": _to_float(row.avg_context_precision),
                "avg_context_recall": _to_float(row.avg_context_recall),
            }
            for row in rows
        ]