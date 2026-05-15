"""
Admin Router  (/api/v1/admin)
─────────────────────────────
All routes require a valid JWT **and** is_superuser = True.
The `get_current_superuser` dependency enforces both via dependency chaining.

Routes:
  GET    /admin/dashboard
  GET    /admin/users
  GET    /admin/users/{user_id}
  PATCH  /admin/users/{user_id}/activate
  PATCH  /admin/users/{user_id}/deactivate
  DELETE /admin/users/{user_id}
  PATCH  /admin/sessions/{session_id}/activate
  PATCH  /admin/sessions/{session_id}/deactivate
"""

from __future__ import annotations

from uuid import UUID
from typing import Optional

from fastapi import APIRouter, Depends, status,Query

from backend.core.dependencies import get_db
from backend.core.security.jwt_deps import get_current_superuser
from backend.storage.db.db_dispatcher import DBDispatcher
from backend.models.admin import (
    AdminDashboardOut,
    MessageAdminOut,
    SessionAdminOut,
    ToggleStatusOut,
    UserAdminOut,
)
from backend.models.enums.message_role import MessageRole
from backend.models.enums.admin_enum import MessageOrderBy, OrderDir, SessionOrderBy

from backend.models.auth import UserOut
from backend.services.admin_service import AdminService

router = APIRouter(
    dependencies=[Depends(get_current_superuser)],
)

def get_admin_service(db: DBDispatcher = Depends(get_db)) -> AdminService:
    """
    Per-request AdminService factory.
    `get_db` is the existing dependency that yields a DBDispatcher.
    """
    return AdminService(db)


# ── Dashboard ──────────────────────────────────────────────────────────────────
 
@router.get("/dashboard", response_model=AdminDashboardOut)
async def get_dashboard(
    service: AdminService = Depends(get_admin_service),
) -> AdminDashboardOut:
    return await service.get_dashboard()
 
 
# ── Users ──────────────────────────────────────────────────────────────────────
 
@router.get("/users", response_model=list[UserAdminOut])
async def list_users(
    service: AdminService = Depends(get_admin_service),
) -> list[UserAdminOut]:
    return await service.get_all_users()
 
 
@router.get("/users/{user_id}", response_model=UserAdminOut)
async def get_user(
    user_id: UUID,
    service: AdminService = Depends(get_admin_service),
) -> UserAdminOut:
    return await service.get_user_detail(user_id)
 
 
@router.patch("/users/{user_id}/activate", response_model=ToggleStatusOut)
async def activate_user(
    user_id: UUID,
    service: AdminService = Depends(get_admin_service),
) -> ToggleStatusOut:
    return await service.activate_user(user_id)
 
 
@router.patch("/users/{user_id}/deactivate", response_model=ToggleStatusOut)
async def deactivate_user(
    user_id: UUID,
    service: AdminService = Depends(get_admin_service),
) -> ToggleStatusOut:
    return await service.deactivate_user(user_id)
 
 
@router.patch("/users/{user_id}/promote", response_model=ToggleStatusOut)
async def promote_user(
    user_id: UUID,
    service: AdminService = Depends(get_admin_service),
) -> ToggleStatusOut:
    return await service.promote_to_superuser(user_id)
 
 
@router.patch("/users/{user_id}/demote", response_model=ToggleStatusOut)
async def demote_user(
    user_id: UUID,
    current_admin: UserOut = Depends(get_current_superuser),
    service: AdminService = Depends(get_admin_service),
) -> ToggleStatusOut:
    return await service.demote_from_superuser(user_id, current_admin.id)
 
 
@router.delete("/users/{user_id}", status_code=status.HTTP_200_OK)
async def delete_user(
    user_id: UUID,
    service: AdminService = Depends(get_admin_service),
) -> dict:
    return await service.delete_user(user_id)
 
 
# ── Sessions ───────────────────────────────────────────────────────────────────

@router.get("/sessions", response_model=list[SessionAdminOut])
async def list_sessions(
    user_id: Optional[UUID] = Query(default=None, description="Filter by user"),
    is_active: Optional[bool] = Query(default=None, description="Filter by active status"),
    order_by: SessionOrderBy = Query(default=SessionOrderBy.created_at),
    order_dir: OrderDir = Query(default=OrderDir.desc),
    service: AdminService = Depends(get_admin_service),
) -> list[SessionAdminOut]:
    return await service.get_all_sessions(
        user_id=user_id,
        is_active=is_active,
        order_by=order_by,
        order_dir=order_dir,
    )
 
 
@router.patch("/sessions/{session_id}/activate", response_model=ToggleStatusOut)
async def activate_session(
    session_id: UUID,
    service: AdminService = Depends(get_admin_service),
) -> ToggleStatusOut:
    return await service.activate_session(session_id)
 
 
@router.patch("/sessions/{session_id}/deactivate", response_model=ToggleStatusOut)
async def deactivate_session(
    session_id: UUID,
    service: AdminService = Depends(get_admin_service),
) -> ToggleStatusOut:
    return await service.deactivate_session(session_id)
 
 
# ── Messages ───────────────────────────────────────────────────────────────────
@router.get("/messages", response_model=list[MessageAdminOut])
async def list_messages(
    user_id: Optional[UUID] = Query(default=None, description="Filter by user"),
    session_id: Optional[UUID] = Query(default=None, description="Filter by session"),
    is_active: Optional[bool] = Query(default=None, description="Filter by session active status"),
    role: Optional[MessageRole] = Query(default=None, description="Filter by role: user / assistant"),
    order_by: MessageOrderBy = Query(default=MessageOrderBy.created_at),
    order_dir: OrderDir = Query(default=OrderDir.desc),
    service: AdminService = Depends(get_admin_service),
) -> list[MessageAdminOut]:
    return await service.get_all_messages(
        user_id=user_id,
        session_id=session_id,
        is_active=is_active,
        role=role,
        order_by=order_by,
        order_dir=order_dir,
    )