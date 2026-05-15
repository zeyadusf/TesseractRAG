"""
Abstract vector store interface.
Any vector backend (pgvector, FAISS, Qdrant, etc.) must implement this.
The service layer imports only this — never a concrete class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID


class BaseVectorStore(ABC):

    @abstractmethod
    async def upsert(
        self,
        session_id: UUID,
        chunk_id: UUID,
        vector: list[float],
        model_name: str,
        dimensions: int,
        token_usage: int | None = None,
    ) -> None:
        """Insert or update an embedding for a chunk."""
        ...

    @abstractmethod
    async def search(
        self,
        session_id: UUID,
        query_vector: list[float],
        model_name: str,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Return top_k nearest chunks by cosine similarity.
        Shape: [{chunk_id: UUID, score: float}]
        """
        ...

    @abstractmethod
    async def delete_by_session(self, session_id: UUID) -> int:
        """Delete all embeddings for a session. Returns count."""
        ...

    @abstractmethod
    async def delete_by_chunk(self, chunk_id: UUID) -> int:
        """Delete all embeddings for a single chunk. Returns count."""
        ...
