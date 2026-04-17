from abc import abstractmethod
from typing import Optional
from uuid import UUID
from .BaseDataModel import BaseDataModel


class BaseEvaluation(BaseDataModel):
    """
    Interface for Evaluations table.

    Schema reminder:
        id                  UUID  PK
        session_id          UUID  FK → sessions.id  CASCADE
        message_id          UUID  FK → messages.id  CASCADE
        faithfulness        Float nullable
        answer_relevancy    Float nullable
        context_precision   Float nullable
        created_at          DateTime
    """

    @classmethod
    @abstractmethod
    async def create_instance(cls, db_client: object) -> "BaseEvaluation":
        pass

    @abstractmethod
    async def insert_evaluation(
        self,
        session_id: UUID,
        message_id: UUID,
        faithfulness: Optional[float] = None,
        answer_relevancy: Optional[float] = None,
        context_precision: Optional[float] = None,
    ) -> dict:
        """Insert a single evaluation record and return it."""
        pass

    @abstractmethod
    async def get_evaluation(self, evaluation_id: UUID) -> Optional[dict]:
        """Get a single evaluation by its PK."""
        pass

    @abstractmethod
    async def get_evaluation_by_message(
        self, message_id: UUID, session_id: UUID
    ) -> Optional[dict]:
        """
        Get the evaluation for a specific message.
        One message → one evaluation (1-to-1 relation).
        """
        pass

    @abstractmethod
    async def get_last_n_evaluations(
        self,
        session_id: UUID,
        n: int = 20,
    ) -> list[dict]:
        """Get the last N evaluations for a session, ordered by created_at DESC."""
        pass

    @abstractmethod
    async def get_all_evaluations(
        self,
        session_id: UUID,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> list[dict]:
        """Get all evaluations for a session, ordered by created_at DESC."""
        pass

    @abstractmethod
    async def get_average_scores(self, session_id: UUID) -> dict:
        """
        Return aggregate scores for a session.
        Example return:
        {
            "faithfulness": 0.87,
            "answer_relevancy": 0.91,
            "context_precision": 0.78,
            "count": 42
        }
        """
        pass

    @abstractmethod
    async def delete_evaluation(self, evaluation_id: UUID) -> bool:
        """Delete a single evaluation. Returns True if deleted."""
        pass

    @abstractmethod
    async def delete_evaluations_by_session(self, session_id: UUID) -> int:
        """Delete all evaluations for a session. Returns count deleted."""
        pass