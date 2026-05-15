from __future__ import annotations

from uuid import UUID
from typing import Optional

from backend.storage.db.db_dispatcher import DBDispatcher
from backend.models.admin import (
    AdminDashboardOut,
    AvgScores,
    MessageAdminOut,
    SessionAdminOut,
    ToggleStatusOut,
    UserAdminOut,
)
from backend.models.enums.admin_enum import MessageOrderBy, OrderDir, SessionOrderBy
from backend.models.enums.message_role import MessageRole
from backend.services.base_service import BaseService
from backend.services.exceptions import NotFoundError, ValidationError


class AdminService(BaseService):
    def __init__(self, db: DBDispatcher) -> None:
        super().__init__(db)

    # ── Dashboard ──────────────────────────────────────────────────────────

    async def get_dashboard(self) -> AdminDashboardOut:
        total_users = await self.db.users.count_all()
        total_superusers = await self.db.users.count_all_superusers()   # ← new
        total_sessions = await self.db.sessions.count_all()
        total_messages = await self.db.messages.count_all()
        total_documents = await self.db.documents.count_all()
        global_scores = await self.db.evaluations.get_global_avg_scores()

        return AdminDashboardOut(
            total_users=total_users,
            total_superusers=total_superusers,
            total_sessions=total_sessions,
            total_messages=total_messages,
            total_documents=total_documents,
            global_avg_scores=AvgScores(
                faithfulness=global_scores.get("avg_faithfulness"),
                answer_relevancy=global_scores.get("avg_answer_relevancy"),
                context_precision=global_scores.get("avg_context_precision"),
                context_recall=global_scores.get("avg_context_recall"),
            ),
        )

    # ── Users ──────────────────────────────────────────────────────────────

    async def get_all_users(self) -> list[UserAdminOut]:
        rows = await self.db.users.get_all_with_stats()
        return [UserAdminOut.from_stats_dict(row) for row in rows]

    async def get_user_detail(self, user_id: UUID) -> UserAdminOut:
        user = await self.db.users.get_by_id(user_id)
        if user is None:
            raise NotFoundError("User", str(user_id))
        stats = await self.db.users.get_user_stats(user_id)
        stats["user"] = user
        return UserAdminOut.from_stats_dict(stats)

    # ── Sessions ───────────────────────────────────────────────────────────

    async def get_all_sessions(
        self,
        user_id: Optional[UUID],
        is_active: Optional[bool],
        order_by: SessionOrderBy,
        order_dir: OrderDir,
    ) -> list[SessionAdminOut]:
        rows = await self.db.sessions.get_all_with_stats(
            user_id=user_id,
            is_active=is_active,
            order_by=order_by.value,
            order_dir=order_dir.value,
        )
        return [SessionAdminOut.from_row(row) for row in rows]

    # ── Messages ───────────────────────────────────────────────────────────

    async def get_all_messages(
        self,
        user_id: Optional[UUID],
        session_id: Optional[UUID],
        is_active: Optional[bool],
        role: Optional[MessageRole],
        order_by: MessageOrderBy,
        order_dir: OrderDir,
    ) -> list[MessageAdminOut]:
        rows = await self.db.messages.get_all_with_eval(
            user_id=user_id,
            session_id=session_id,
            is_active=is_active,
            role=role,
            order_by=order_by.value,
            order_dir=order_dir.value,
        )
        return [MessageAdminOut.from_row(row) for row in rows]

    # ── User control ───────────────────────────────────────────────────────

    async def activate_user(self, user_id: UUID) -> ToggleStatusOut:
        user = await self.db.users.update(user_id, is_active=True)
        if user is None:
            raise NotFoundError("User", str(user_id))
        return ToggleStatusOut(id=user.id, is_active=True,
                               detail=f"User '{user.username}' activated.")

    async def deactivate_user(self, user_id: UUID) -> ToggleStatusOut:
        user = await self.db.users.update(user_id, is_active=False)
        if user is None:
            raise NotFoundError("User", str(user_id))
        return ToggleStatusOut(id=user.id, is_active=False,
                               detail=f"User '{user.username}' deactivated.")

    async def promote_to_superuser(self, user_id: UUID) -> ToggleStatusOut:
        user = await self.db.users.update(user_id, is_superuser=True)
        if user is None:
            raise NotFoundError("User", str(user_id))
        return ToggleStatusOut(id=user.id, is_active=user.is_active,
                               detail=f"User '{user.username}' promoted to superuser.")

    async def demote_from_superuser(
        self,
        user_id: UUID,
        requesting_admin_id: UUID,
    ) -> ToggleStatusOut:
        if user_id == requesting_admin_id:
            raise ValidationError("Cannot revoke your own superuser privileges.")
        user = await self.db.users.update(user_id, is_superuser=False)
        if user is None:
            raise NotFoundError("User", str(user_id))
        return ToggleStatusOut(id=user.id, is_active=user.is_active,
                               detail=f"User '{user.username}' demoted from superuser.")

    async def delete_user(self, user_id: UUID) -> dict:
        user = await self.db.users.get_by_id(user_id)
        if user is None:
            raise NotFoundError("User", str(user_id))
        username = user.username
        await self.db.users.delete(user_id)
        return {"detail": f"User '{username}' and all associated data permanently deleted."}

    # ── Session control ────────────────────────────────────────────────────

    async def activate_session(self, session_id: UUID) -> ToggleStatusOut:
        session = await self.db.sessions.update(session_id, is_active=True)
        if session is None:
            raise NotFoundError("Session", str(session_id))
        return ToggleStatusOut(id=session.id, is_active=True,
                               detail=f"Session '{session.name}' activated.")

    async def deactivate_session(self, session_id: UUID) -> ToggleStatusOut:
        session = await self.db.sessions.update(session_id, is_active=False)
        if session is None:
            raise NotFoundError("Session", str(session_id))
        return ToggleStatusOut(id=session.id, is_active=False,
                               detail=f"Session '{session.name}' deactivated.")