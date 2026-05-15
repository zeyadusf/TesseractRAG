"""
pgvector implementation of BaseVectorStore.
Uses the EmbeddingRepository for all DB operations.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.storage.db.postgres.repositories.embedding_repo import EmbeddingRepository
from backend.storage.vector_db.base import BaseVectorStore


class PgVectorStore(BaseVectorStore):
    """
    Stores and searches embeddings using PostgreSQL + pgvector HNSW index.
    Requires an AsyncSession injected at construction — follows the
    unit-of-work pattern so all writes share the caller's transaction.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._repo = EmbeddingRepository(session)

    async def upsert(self,session_id: UUID,chunk_id: UUID,vector: list[float],model_name: str,
                    dimensions: int,token_usage: int | None = None,) -> None:
        await self._repo.upsert_by_chunk(chunk_id=chunk_id,session_id=session_id,model_name= model_name,
                                            dimensions=dimensions,vector=vector,token_usage=token_usage,)

    async def search(
        self,
        session_id: UUID,
        query_vector: list[float],
        model_name: str,
        top_k: int = 10,
    ) -> list[dict]:
        return await self._repo.similarity_search(
            query_vector=query_vector,
            session_id=session_id,
            model_name=model_name,
            top_k=top_k,
        )

    async def delete_by_session(self, session_id: UUID) -> int:
        return await self._repo.delete_by_session(session_id)

    async def delete_by_chunk(self, chunk_id: UUID) -> int:
        return await self._repo.delete_by_chunk(chunk_id)
