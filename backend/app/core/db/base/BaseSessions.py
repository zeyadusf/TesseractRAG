from abc import abstractmethod
from typing import Optional
from uuid import UUID
from .BaseDataModel import BaseDataModel


class BaseSessions(BaseDataModel):
    """
    Interface for Sessions table.

    Schema reminder:
        id          UUID  PK
        name        String
        description String nullable
        owner_id    String
        created_at  DateTime
    """

    @classmethod
    @abstractmethod
    async def create_instance(cls, db_client: object) -> "BaseSessions":
        pass

    @abstractmethod
    async def create_session(
        self,
        name: str,
        owner_id: str,
        description: Optional[str] = None,
    ) -> dict:
        """Create and return a new session."""
        pass

    @abstractmethod
    async def get_session(self, session_id: UUID, owner_id: str) -> Optional[dict]:
        """Get a single session by id, scoped to owner."""
        pass

    @abstractmethod
    async def get_all_sessions(
        self,
        owner_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> list[dict]:
        """Get all sessions for an owner, ordered by created_at DESC."""
        pass

    @abstractmethod
    async def update_session(
        self,
        session_id: UUID,
        owner_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Optional[dict]:
        """Update session fields. Returns updated session or None if not found."""
        pass

    @abstractmethod
    async def delete_session(self, session_id: UUID, owner_id: str) -> bool:
        """Delete a session. Returns True if deleted."""
        pass

    @abstractmethod
    async def is_session_exists(self, session_id: UUID, owner_id: str) -> bool:
        """Check if a session belongs to this owner."""
        pass