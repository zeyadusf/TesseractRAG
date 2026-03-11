from fastapi import APIRouter, Depends
from app.models.session import SessionCreate, SessionResponse
from app.dependencies import get_session_manager
from app.core.session_manager import SessionManager

router = APIRouter()

@router.post("/", response_model=SessionResponse,status_code=201)
async def create_session(
    body : SessionCreate ,
    manager : SessionManager = Depends(get_session_manager)):

    return manager.create_session(body.name, body.description)

@router.get("/", response_model=list[SessionResponse])
async def list_sessions( manager : SessionManager = Depends(get_session_manager)):
    return manager.list_sessions()

@router.delete("/{session_id}", status_code=204)
async def delete_session(session_id: str,manager : SessionManager = Depends(get_session_manager)):
    manager.delete_session(session_id=session_id)