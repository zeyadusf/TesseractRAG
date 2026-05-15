"""
Sessions router — RAG session lifecycle management.

A "session" is a named workspace that owns documents and chat history.

Routes
------
POST   /api/v1/sessions                       → create session
GET    /api/v1/sessions                       → list user's sessions
GET    /api/v1/sessions/{session_id}          → get session detail
PATCH  /api/v1/sessions/{session_id}          → rename / update session
DELETE /api/v1/sessions/{session_id}          → delete session + cascade
"""
# from __future__ import annotations
from uuid import UUID
from typing import List

from fastapi import APIRouter, Depends, Query, status, Request

from backend.models.sessions import SessionCreate, SessionOut, SessionUpdate
from backend.core.dependencies import get_db
from backend.core.security.jwt_deps import get_current_active_user
from backend.core.limiter import limiter  
from backend.models.auth import UserOut
from backend.services.session_service import SessionService
from backend.storage.db.db_dispatcher import DBDispatcher

router = APIRouter()


def get_session_service(db: DBDispatcher = Depends(get_db)) -> SessionService:
    return SessionService(db)

# ── Create 
@router.post(
    "",
    response_model=SessionOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new RAG session",
)
@limiter.limit("10/hour")
async def create_session(
    request: Request,
    payload: SessionCreate,
    current_user: UserOut = Depends(get_current_active_user),
    session_service: SessionService = Depends(get_session_service),
) -> SessionOut:
    return await session_service.create_session(data=payload, user_id=current_user.id)

# ── List 
@router.get(
    "",
    response_model=List[SessionOut],
    summary="List the authenticated user's sessions",
)
@limiter.limit("60/minute")
async def list_sessions(
    request: Request,
    active_only: bool = Query(False, description="Filter active sessions only"),
    current_user: UserOut = Depends(get_current_active_user),
    session_service: SessionService = Depends(get_session_service),
) -> List[SessionOut]:
    return await session_service.list_sessions(user_id=current_user.id, active_only=active_only)

# ── Get 
@router.get(
    "/{session_id}",
    response_model=SessionOut,
    summary="Get session detail",
)
@limiter.limit("60/minute")
async def get_session(
    request: Request,
    session_id: UUID,
    current_user: UserOut = Depends(get_current_active_user),
    session_service: SessionService = Depends(get_session_service),
) -> SessionOut:
    return await session_service.get_session(session_id=session_id, user_id=current_user.id)

# ── Update ────────────────────────────────────────────────────────────────────
@router.patch(
    "/{session_id}",
    response_model=SessionOut,
    summary="Rename or update session metadata",
)
@limiter.limit("20/minute")
async def update_session(
    request: Request,
    session_id: UUID,
    payload: SessionUpdate,
    current_user: UserOut = Depends(get_current_active_user),
    session_service: SessionService = Depends(get_session_service),
) -> SessionOut:
    return await session_service.update_session(session_id=session_id, user_id=current_user.id, data=payload)

# ── Delete 
@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a session and all its documents / chat history",
)
@limiter.limit("10/minute")
async def delete_session(
    request: Request,
    session_id: UUID,
    current_user: UserOut = Depends(get_current_active_user),
    session_service: SessionService = Depends(get_session_service),
) -> None:
    await session_service.delete_session(session_id=session_id, user_id=current_user.id)
    return None

# ── Admin: Toggle Session Status deactivate 