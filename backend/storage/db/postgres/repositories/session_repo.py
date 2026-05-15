from __future__ import annotations

from typing import List
from uuid import UUID

from sqlalchemy import asc, desc, func, text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from backend.storage.db.postgres.repositories.base_repo import BaseRepository
from backend.storage.db.postgres.schemas import Session
from backend.storage.db.postgres.schemas.document import Document
from backend.storage.db.postgres.schemas.message import Message


class SessionRepository(BaseRepository[Session]):
    def __init__(self, sessionConn: AsyncSession):
        self._s = sessionConn

    # ── CRUD ───────────────────────────────────────────────────────────────

    async def create(self, user_id: UUID, name: str, description: str) -> Session | None:
        session = Session(user_id=user_id, name=name, description=description)
        self._s.add(session)
        await self._s.flush()
        await self._s.refresh(session)
        return session

    async def get_by_id(self, record_id) -> Session | None:
        result = await self._s.execute(
            select(Session).where(Session.id == record_id)
        )
        return result.scalar_one_or_none()

    async def get_by_id_and_user(self, session_id: UUID, user_id: UUID) -> Session | None:
        result = await self._s.execute(
            select(Session).where(
                Session.id == session_id,
                Session.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_list_sessions_by_user(
        self, user_id: UUID, active_only: bool = False
    ) -> List[Session]:
        stmt = select(Session).where(Session.user_id == user_id)
        if active_only:
            stmt = stmt.where(Session.is_active.is_(True))
        stmt = stmt.order_by(Session.created_at.desc())
        result = await self._s.execute(stmt)
        return list(result.scalars().all())

    async def update(self, record_id: UUID, **kwargs) -> Session | None:
        session = await self.get_by_id(record_id)
        if session is None:
            return None
        for key, value in kwargs.items():
            setattr(session, key, value)
        await self._s.flush()
        await self._s.refresh(session)
        return session

    async def delete(self, record_id) -> bool:
        session = await self.get_by_id(record_id)
        if session is None:
            return False
        await self._s.delete(session)
        await self._s.flush()
        return True

    async def deactivate(self, record_id: UUID) -> Session | None:
        return await self.update(record_id, is_active=False)

    # ── Admin ──────────────────────────────────────────────────────────────

    async def count_all(self) -> int:
        result = await self._s.execute(select(func.count()).select_from(Session))
        return result.scalar_one()

    async def get_all_with_stats(
        self,
        user_id=None,
        is_active=None,
        order_by: str = "created_at",
        order_dir: str = "desc",
    ) -> list:
        """
        Returns all sessions with document_count, message_count,
        and a list of (doc_id, filename, status) tuples per session.

        FIX: order_by on subquery columns now uses the labeled column
        from the outer SELECT — not the subquery column reference — so
        PostgreSQL keeps NULL rows instead of filtering them out.
        """
        _dir = desc if order_dir == "desc" else asc

        # ── aggregation subqueries ─────────────────────────────────────────
        doc_counts = (
            select(
                Document.session_id.label("sid"),
                func.count(Document.id).label("document_count"),
            )
            .group_by(Document.session_id)
            .subquery()
        )
        msg_counts = (
            select(
                Message.session_id.label("sid"),
                func.count(Message.id).label("message_count"),
            )
            .group_by(Message.session_id)
            .subquery()
        )

        # ── main query ────────────────────────────────────────────────────
        doc_count_col = func.coalesce(doc_counts.c.document_count, 0).label("document_count")
        msg_count_col = func.coalesce(msg_counts.c.message_count, 0).label("message_count")

        stmt = (
            select(Session, doc_count_col, msg_count_col)
            .outerjoin(doc_counts, doc_counts.c.sid == Session.id)
            .outerjoin(msg_counts, msg_counts.c.sid == Session.id)
        )

        if user_id is not None:
            stmt = stmt.where(Session.user_id == user_id)
        if is_active is not None:
            stmt = stmt.where(Session.is_active == is_active)

        # ── ordering — use text() label reference to avoid NULL filtering ──
        # Referencing the label name via text() tells PostgreSQL to reuse
        # the already-computed SELECT expression instead of re-evaluating
        # the subquery column, which strips NULLs.
        if order_by == "document_count":
            stmt = stmt.order_by(_dir(text("document_count")))
        elif order_by == "message_count":
            stmt = stmt.order_by(_dir(text("message_count")))
        else:
            stmt = stmt.order_by(_dir(Session.created_at))

        sessions_result = await self._s.execute(stmt)
        rows = sessions_result.all()

        if not rows:
            return []

        # ── fetch documents for all returned sessions in one query ─────────
        session_ids = [row[0].id for row in rows]
        docs_result = await self._s.execute(
            select(Document.session_id, Document.id, Document.filename, Document.status)
            .where(Document.session_id.in_(session_ids))
            .order_by(Document.session_id, Document.uploaded_at)
        )
        docs_by_session: dict[UUID, list[dict]] = {}
        for doc_row in docs_result.all():
            docs_by_session.setdefault(doc_row.session_id, []).append(
                {
                    "id": doc_row.id,
                    "filename": doc_row.filename,
                    "status": doc_row.status,
                }
            )

        # ── attach documents to each row ───────────────────────────────────
        # Return a list of dicts so SessionAdminOut.from_row() can stay simple
        return [
            {
                "session": row[0],
                "document_count": row.document_count,
                "message_count": row.message_count,
                "documents": docs_by_session.get(row[0].id, []),
            }
            for row in rows
        ]