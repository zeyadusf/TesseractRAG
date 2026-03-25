"""
sessions.py
-----------
Session CRUD endpoints for TesseractRAG API v1.

All endpoints require the X-Owner-ID header (enforced via get_owner_id
dependency). Sessions are created, listed, and deleted in the context of
the requesting owner — cross-owner access is rejected at the manager level.
"""

from fastapi import APIRouter, Depends
from app.models.session import SessionCreate, SessionResponse
from app.dependencies import get_session_manager, get_owner_id
from app.core.session_manager import SessionManager

router = APIRouter()


@router.post("/", response_model=SessionResponse, status_code=201)
async def create_session(
    body: SessionCreate,
    manager: SessionManager = Depends(get_session_manager),
    owner_id: str = Depends(get_owner_id),              # ← NEW
):
    """
    Create a new session owned by the requesting browser.

    The owner_id from X-Owner-ID is stored with the session in R2.
    Future requests to this session must present the same owner_id.
    """
    return manager.create_session(body.name, body.description, owner_id)


@router.get("/", response_model=list[SessionResponse])
async def list_sessions(
    manager: SessionManager = Depends(get_session_manager),
    owner_id: str = Depends(get_owner_id),              # ← NEW
):
    """
    List all sessions belonging to the requesting browser.

    Sessions owned by other browsers are not returned — they are completely
    invisible. An empty list is returned if the owner has no sessions yet.
    """
    return manager.list_sessions(owner_id)


@router.delete("/{session_id}", status_code=204)
async def delete_session(
    session_id: str,
    manager: SessionManager = Depends(get_session_manager),
    owner_id: str = Depends(get_owner_id),              # ← NEW
):
    """
    Delete a session and all its R2 data.

    Returns 403 if the requesting owner does not own this session.
    Returns 404 if the session does not exist.
    """
    manager.delete_session(session_id=session_id, owner_id=owner_id)
