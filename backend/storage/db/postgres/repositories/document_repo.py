from __future__  import annotations
from .base_repo import BaseRepository
from backend.storage.db.postgres.schemas import Document
from backend.models.enums.doc_status import DocumentStatus
from typing import List
from datetime import datetime, timezone
from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession



class DocumentRepository(BaseRepository[Document]):
    def __init__(self,sessionConn:AsyncSession):
        self._s = sessionConn

    # ── CRUD ───────────────────────────────────────────────────────────────

    async def create( self, 
        session_id: UUID,
        filename: str,
        file_extension: str,
        file_size_bytes: int,
        blob_path: str | None = None,) -> Document:
        
        doc = Document(
            session_id=session_id,
            filename = filename,
            file_extension = file_extension,
            file_size_bytes = file_size_bytes,
            blob_path = blob_path,
            status=DocumentStatus.PENDING,
        )

        self._s.add(doc)
        await self._s.flush()
        await self._s.refresh(doc)
        return doc
    
    async def get_by_id(self, record_id: UUID) -> Document | None:
        result = await self._s.execute(
            select(Document).where(Document.id == record_id)
        )
        return result.scalar_one_or_none()

    async def get_by_id_and_session(self, document_id: UUID, session_id: UUID) -> Document | None:
        result = await self._s.execute(
            select(Document).where(
                Document.id == document_id,
                Document.session_id == session_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_by_session(self, session_id: UUID) -> List[Document]:
        result = await self._s.execute(
            select(Document)
            .where(Document.session_id == session_id)
            .order_by(Document.uploaded_at.desc())
        )
        return list(result.scalars().all())

    async def list_by_status(self, session_id: UUID, status: DocumentStatus) -> List[Document]:
        result = await self._s.execute(
            select(Document).where(
                Document.session_id == session_id,
                Document.status == status,
            )
        )
        return list(result.scalars().all())

    async def update(self, record_id: UUID, **kwargs) -> Document | None:
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
    # ── Lookups ────────────────────────────────────────────────────────────

    async def count_by_session(self, session_id: UUID) -> int:
        result = await self._s.execute(
            select(func.count(Document.id)).where(Document.session_id == session_id)
        )
        return result.scalar_one()

    
    # ── Pipeline-specific helpers ──────────────────────────────────────────

    async def set_status(self,document_id: UUID,status: DocumentStatus,) -> None:
        await self._s.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(status=status)
        )
        await self._s.flush()

    async def increment_chunk_count(self, document_id: UUID, amount: int) -> None:
        await self._s.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(chunk_count=Document.chunk_count + amount)
        )
        await self._s.flush()

    async def mark_indexed(self,document_id: UUID,language: str | None = None) -> None:
        """
        Mark the document as fully indexed.
        Called at the end of the ingestion pipeline when all chunks
        and embeddings are persisted successfully.
        """
        await self._s.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(
                status=DocumentStatus.INDEXED,
                indexed_at=datetime.now(timezone.utc),
                language=language,
            )
        )
        await self._s.flush()

    async def count_all(self) -> int:
        """Total number of documents across all sessions."""
        result = await self._s.execute(select(func.count()).select_from(Document))
        return result.scalar_one()