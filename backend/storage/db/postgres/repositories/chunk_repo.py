from __future__  import annotations
from .base_repo import BaseRepository
from backend.storage.db.postgres.schemas import Chunk
from typing import List
from sqlalchemy import  select,delete
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession



class ChunkRepository(BaseRepository[Chunk]):
    def __init__(self,sessionConn:AsyncSession):
        self._s = sessionConn
    
    async def create(
        self,
        *,
        document_id: UUID,
        session_id: UUID,
        content: str,
        chunk_index: int,
        chunk_size: int,
        word_count: int,
        chunker_type: str = "recursive",
        chunk_metadata: dict | None = None,
    ) -> Chunk:
        obj = Chunk(
            document_id=document_id,
            session_id=session_id,
            content=content,
            chunk_index=chunk_index,
            chunk_size=chunk_size,
            word_count=word_count,
            chunker_type=chunker_type,
            chunk_metadata=chunk_metadata or {},
        )
        self._s.add(obj)
        await self._s.flush()
        await self._s.refresh(obj)
        return obj

    async def bulk_create(self, chunks: List[dict]) -> List[Chunk]:
        """
        Insert multiple chunks in a single flush.
        `chunks` is a list of dicts matching Chunk column names.

        This is the primary insertion method — always use this during ingestion.
        Single-row create() is only for one-off inserts (tests, etc.).

        Example chunk dict:
        {
            "document_id": UUID(...),
            "session_id":  UUID(...),
            "content":     "text here",
            "chunk_index": 0,
            "chunk_size":  480,
            "word_count":  91,
            "content_hash":"abc123...",
            "chunker_type":"recursive",
            "metadata":    {"language": "en", "page_num": 1},
        }
        """
        if not chunks:
            return []
        objs = [Chunk(**c) for c in chunks]
        self._s.add_all(objs)
        await self._s.flush()
        for obj in objs:
            await self._s.refresh(obj)
        return objs


    async def get_by_id(self, record_id: UUID) -> Chunk | None:
        result = await self._s.execute(
        select(Chunk).where(Chunk.id == record_id)
        )
        return result.scalar_one_or_none()

    async def get_by_ids(self, chunk_ids: List[UUID]) -> List[Chunk]:
        if not chunk_ids:
            return []
        
        result = await self._s.execute(
            select(Chunk).where(Chunk.id.in_(chunk_ids))
        )
        # Preserve caller's requested order (score order from vector search)
        chunks_map = {c.id: c for c in result.scalars().all()}
        return [chunks_map[cid] for cid in chunk_ids if cid in chunks_map]

    async def list_by_document(self, document_id: UUID) -> List[Chunk]:
        """Return all chunks for a document in reading order."""
        result = await self._s.execute(
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index.asc())
        )
        return list(result.scalars().all())

    async def list_by_session(self, session_id: UUID) -> List[Chunk]:
        result = await self._s.execute(
            select(Chunk)
            .where(Chunk.session_id == session_id)
            .order_by(Chunk.chunk_index.asc())
        )
        return list(result.scalars().all())


    async def update(self, record_id: UUID, **kwargs) -> Chunk | None:
        """Chunks are generally immutable after ingestion — use sparingly."""
        obj = await self.get_by_id(record_id)
        if obj is None:
            return None
        for field, value in kwargs.items():
            setattr(obj, field, value)
        await self._s.flush()
        await self._s.refresh(obj)
        return obj

    async def delete(self, record_id: UUID) -> bool:
        obj = await self.get_by_id(record_id)
        if obj is None:
            return False
        await self._s.delete(obj)
        await self._s.flush()
        return True

    async def delete_by_document(self, document_id: UUID) -> int:
        """
        Delete all chunks for a document in one query.
        Returns the count of deleted rows.
        Used when re-indexing or removing a document.
        """
        result = await self._s.execute(
            delete(Chunk)
            .where(Chunk.document_id == document_id)
            .returning(Chunk.id)
        )
        await self._s.flush()
        return len(result.fetchall())
    
    async def delete_by_session(self, session_id: UUID) -> int:
        """Delete all chunks across all documents in a session."""
        result = await self._s.execute(
            delete(Chunk)
            .where(Chunk.session_id == session_id)
            .returning(Chunk.id)
        )
        await self._s.flush()
        return len(result.fetchall())

    # ── Lookups ────────────────────────────────────────────────────────────

    async def count_by_session(self, session_id: UUID) -> int:
        from sqlalchemy import func
        result = await self._s.execute(
            select(func.count(Chunk.id)).where(Chunk.session_id == session_id)
        )
        return result.scalar_one()
