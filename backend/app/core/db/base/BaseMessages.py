from abc import abstractmethod
from typing import Optional
from uuid import UUID
from .BaseDataModel import BaseDataModel


class BaseMessages(BaseDataModel):
    """
    Interface for Messages table.

    Schema reminder:
        id          UUID  PK
        session_id  UUID  FK → sessions.id CASCADE
        role        String          ('user' | 'assistant' | 'system')
        content     Text
        timestamp   DateTime
    """

    @classmethod
    @abstractmethod
    async def create_instance(cls, db_client: object) -> "BaseMessages":
        pass

    @abstractmethod
    async def insert_message(
        self,
        session_id: UUID,
        role: str,
        content: str,
    ) -> dict:
        """Insert a single message and return it."""
        pass

    @abstractmethod
    async def insert_messages_batch(
        self,
        session_id: UUID,
        messages: list[dict],   # [{"role": ..., "content": ...}]
    ) -> list[dict]:
        """Bulk insert messages. Returns inserted rows."""
        pass

    @abstractmethod
    async def get_message(self, message_id: UUID, session_id: UUID) -> Optional[dict]:
        """Get a single message by id, scoped to session."""
        pass

    @abstractmethod
    async def get_last_n_messages(
        self,
        session_id: UUID,
        n: int = 20,
    ) -> list[dict]:
        """
        Get the last N messages in a session ordered by timestamp ASC
        (so the result is ready to pass directly to an LLM).
        """
        pass

    @abstractmethod
    async def get_all_messages(
        self,
        session_id: UUID,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> list[dict]:
        """Get all messages in a session ordered by timestamp ASC."""
        pass

    @abstractmethod
    async def count_messages(self, session_id: UUID) -> int:
        """Return total message count for a session."""
        pass

    @abstractmethod
    async def delete_message(self, message_id: UUID, session_id: UUID) -> bool:
        """Delete a single message. Returns True if deleted."""
        pass

    @abstractmethod
    async def delete_all_messages(self, session_id: UUID) -> int:
        """Delete all messages in a session. Returns count deleted."""
        pass