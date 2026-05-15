from __future__ import annotations
from backend.storage.db.db_dispatcher import DBDispatcher
from .base_service import BaseService
from .exceptions import NotFoundError
from backend.models.sessions import SessionCreate,SessionOut,SessionUpdate
from backend.storage.db.postgres.schemas.session import Session
from typing import List
from uuid import UUID
class SessionService(BaseService):

    def __init__(self,db:DBDispatcher):
        super().__init__(db)

    
    async def _to_out(self, session: Session) -> SessionOut:
        """
        Convert ORM model → Pydantic output, enriching with live counts.
        Counts are fetched here rather than stored on the model to avoid
        stale denormalized values during heavy concurrent ingestion.
        """
        doc_count = await self.db.documents.count_by_session(session.id)
        msg_count = await self.db.messages.count_by_session(session.id)

        return SessionOut(
            id=session.id,
            user_id=session.user_id,
            name=session.name,
            description=session.description,
            is_active=session.is_active,
            document_count=doc_count,
            message_count=msg_count,
        )


    async def get_session(self, session_id: UUID, user_id: UUID) -> SessionOut:
        """
        Fetch a single session. Returns 404 if it doesn't exist
        """
        session =await self._verify_ownership(session_id,user_id)
        return await self._to_out(session)

    async def list_sessions(self,user_id:UUID,active_only=False)->List[SessionOut]:
        """
        Fetch all session by user_id 
        """
        sessions = await self.db.sessions.get_list_sessions_by_user(user_id,active_only)
        return [await self._to_out(s)
                for s in sessions if s is not None]
    
    async def create_session(self,data:SessionCreate,user_id:UUID)->SessionOut:
        """
        Create New session
        """
        session = await self.db.sessions.create(
            user_id=user_id,
            name=data.name,
            description= data.description)
        return await self._to_out(session)
    
    async def update_session(self, session_id: UUID,user_id: UUID,data: SessionUpdate,) -> SessionOut:
        session = await self._verify_ownership(session_id,user_id)

        updates = data.model_dump(exclude_none=True)
        if updates:
            session = await self.db.sessions.update(session_id,**updates)
        return await self._to_out(session)

    async def delete_session(self,session_id:UUID,user_id:UUID)->bool:

        session =await self._verify_ownership(session_id,user_id)
        return await self.db.sessions.delete(session_id)

    async def de_activate_session(self, session_id: UUID, user_id: UUID,is_active=False) -> SessionOut:
        """
        Soft delete — keeps all data but marks the session inactive.   
        and also can update to active 
        """
        session = await self._verify_ownership(session_id,user_id)
        if is_active :
            update  = await self.db.sessions.update(session_id, is_active=True)
        else:
            update  = await self.db.sessions.update(session_id, is_active=False)
        return await self._to_out(update)

    async def _verify_ownership(self, session_id: UUID, user_id: UUID) -> Session:
        """
        Shared guard used by other services (DocumentService, ChatService)
        to verify a session belongs to the requesting user before operating on it.
        and use with session service to avoid duplicate code written
        Raises:
            NotFoundError
        """
        session = await self.db.sessions.get_by_id_and_user(session_id, user_id)
        if session is None:
            raise NotFoundError("Session", str(session_id))
        return session
