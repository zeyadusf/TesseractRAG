from abc import abstractmethod
from typing import Optional
from uuid import UUID
from .BaseDataModel import BaseDataModel


class BaseChunks(BaseDataModel):
    """
    Interface for Chunks table.

    Schema reminder:
        id              Integer  PK autoincrement
        content         Text
        embedding       Vector(DIM_EMBEDDING)
        metadata        JSONB nullable
        chunk_index     Integer
        session_id      UUID  FK → sessions.id CASCADE
        document_name   String
        created_at      DateTime

    Indexes: HNSW on embedding, composite (session_id, document_name),
             composite (session_id, chunk_index), created_at.
    """

    @classmethod
    @abstractmethod
    async def create_instance(cls, db_client: object) -> "BaseChunks":
        pass

    # ─── Single Operations ────────────────────────────────────────────────────

    @abstractmethod
    async def insert_chunk(
        self,
        session_id: UUID,
        document_name: str,
        content: str,
        embedding: list[float],
        chunk_index: int,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Insert a single chunk and return it."""
        pass

    @abstractmethod
    async def get_chunk(self, chunk_id: int) -> Optional[dict]:
        """Get a single chunk by its PK."""
        pass

    @abstractmethod
    async def delete_chunk(self, chunk_id: int) -> bool:
        """Delete a single chunk. Returns True if deleted."""
        pass

    # ─── Batch Operations ─────────────────────────────────────────────────────

    @abstractmethod
    async def insert_chunks_batch(
        self,
        session_id: UUID,
        document_name: str,
        chunks: list[dict],     # [{"content", "embedding", "chunk_index", "metadata"?}]
    ) -> list[dict]:
        """Bulk insert chunks. Returns inserted rows."""
        pass

    @abstractmethod
    async def delete_chunks_by_document(
        self, session_id: UUID, document_name: str
    ) -> int:
        """Delete all chunks for a document. Returns count deleted."""
        pass

    @abstractmethod
    async def delete_chunks_by_session(self, session_id: UUID) -> int:
        """Delete ALL chunks for a session. Returns count deleted."""
        pass

    # ─── Query / Retrieval ────────────────────────────────────────────────────

    @abstractmethod
    async def get_chunks_by_session(
        self,
        session_id: UUID,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> list[dict]:
        """Get all chunks for a session ordered by chunk_index."""
        pass

    @abstractmethod
    async def get_chunks_by_document(
        self,
        session_id: UUID,
        document_name: str,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> list[dict]:
        """Get chunks for a specific document ordered by chunk_index."""
        pass

    @abstractmethod
    async def similarity_search(
        self,
        session_id: UUID,
        query_embedding: list[float],
        top_k: int = 5,
        document_name: Optional[str] = None,
    ) -> list[dict]:
        """
        Vector cosine similarity search within a session.
        Returns chunks ordered by similarity DESC with a `similarity_score` field.
        Optionally filter by document_name.
        """
        pass

    # ─── Utility ─────────────────────────────────────────────────────────────

    @abstractmethod
    async def count_chunks(
        self,
        session_id: UUID,
        document_name: Optional[str] = None,
    ) -> int:
        """Count chunks for a session, optionally filtered by document."""
        pass

    @abstractmethod
    async def document_exists(self, session_id: UUID, document_name: str) -> bool:
        """Check if any chunks exist for a document in this session."""
        pass

    @abstractmethod
    async def list_documents(self, session_id: UUID) -> list[str]:
        """Return distinct document names that have chunks in this session."""
        pass