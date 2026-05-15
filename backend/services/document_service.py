from __future__ import annotations

from pathlib import Path
from typing import List
from uuid import UUID

from backend.core import get_logger
from backend.models.documents import DocumentOut
from backend.models.enums.doc_status import DocumentStatus
from backend.rag.pipelines.ingestion_pipeline import IngestionPipeline
from backend.services.base_service import BaseService
from backend.services.exceptions import IngestionError, NotFoundError, ValidationError
from backend.storage.db.db_dispatcher import DBDispatcher
from backend.storage.db.postgres.schemas.document import Document

logger = get_logger(__name__)


class DocumentService(BaseService):
    def __init__(self, db: DBDispatcher):
        super().__init__(db)
        self._pipeline: IngestionPipeline = IngestionPipeline()

    async def get_document(self, session_id: UUID, document_id: UUID) -> DocumentOut:
        doc = await self.db.documents.get_by_id_and_session(document_id, session_id)
        if doc is None:
            raise NotFoundError("Document", str(document_id))
        return DocumentOut.model_validate(doc)

    async def list_documents(self, session_id: UUID) -> List[DocumentOut]:
        docs = await self.db.documents.list_by_session(session_id)
        return [DocumentOut.model_validate(d) for d in docs]

    async def upload_and_index(
        self, session_id: UUID, filename: str, file_bytes: bytes,blob_path=None,
    ) -> DocumentOut:
        """
        Full ingestion pipeline for a single document.
        """
        doc = None
        try:
            # Verify session exists
            session = await self.db.sessions.get_by_id(session_id)
            if session is None:
                raise NotFoundError("Session", str(session_id))

            # Validate file size
            if len(file_bytes) > self._config.MAX_FILE_SIZE_BYTES:
                raise ValidationError(
                    f"File too large: {len(file_bytes) / 1024 / 1024:.1f} MB. "
                    f"Max: {self._config.MAX_FILE_SIZE_BYTES / 1024 / 1024:.1f} MB"
                )

            doc = await self.db.documents.create(
                session_id=session_id,
                filename=filename,
                file_extension=Path(filename).suffix.lower(),
                file_size_bytes=len(file_bytes),
                blob_path=blob_path,
            )

            # Run ingestion pipeline
            result = await self._pipeline.run(file_bytes=file_bytes, filename=filename)
            chunks = result.get("chunks", [])
            embeddings = result.get("embeddings", [])
            metadata = result.get("metadata", {})

            if not chunks:
                raise ValidationError("No chunks extracted from document")

            # Prepare chunk rows for bulk insert
            chunk_rows = [
                {
                    "document_id": doc.id,
                    "session_id": session_id,
                    "content": c["text"] if isinstance(c, dict) else c,
                    "chunk_index": i,
                    "chunk_size": len(c["text"] if isinstance(c, dict) else c),
                    "word_count": len(
                        (c["text"] if isinstance(c, dict) else c).split()
                    ),
                    "chunker_type": self._config.DEFAULT_CHUNK,
                    "chunk_metadata": (
                        c["chunk_metadata"].model_dump()
                        if isinstance(c, dict) and c.get("chunk_metadata") is not None
                        else {}
                    ),
                }
                for i, c in enumerate(chunks)
            ]
            inserted_chunks = await self.db.chunks.bulk_create(chunk_rows)
            await self.db.documents.increment_chunk_count(doc.id, len(inserted_chunks))

            # Prepare embedding rows for bulk insert
            model_name = self._config.DEFAULT_EMBED
            dimensions = self._config.EMBED_DIM

            embedding_rows = [
                {
                    "chunk_id": inserted_chunks[i].id,
                    "session_id": session_id,
                    "model_name": model_name,
                    "dimensions": dimensions,
                    "vector": (
                        embeddings[i]["vector"]
                        if isinstance(embeddings[i], dict)
                        else embeddings[i]
                    ),
                    "token_usage": (
                        embeddings[i].get("tokens")
                        if isinstance(embeddings[i], dict)
                        else None
                    ),
                }
                for i in range(len(inserted_chunks))
            ]
            await self.db.embeddings.bulk_create(embedding_rows)

            # Mark document as indexed
            language = metadata.get("language", "en") if metadata else "en"
            await self.db.documents.mark_indexed(doc.id, language=language)
            logger.info("Document fully indexed: %s", doc.id)

            # Return final document state
            final_doc = await self.db.documents.get_by_id(doc.id)
            return DocumentOut.model_validate(final_doc)

        except IngestionError:
            if doc and getattr(doc, "id", None):
                await self.db.documents.set_status(
                    doc.id, DocumentStatus.FAILED, error_message="Ingestion pipeline failed"
                )
            raise

        except Exception as exc:
            if doc and getattr(doc, "id", None):
                try:
                    await self.db.documents.set_status(
                        doc.id, DocumentStatus.FAILED, error_message=str(exc)[:500]
                    )
                except Exception as update_err:
                    logger.error("Failed to update doc %s status: %s", doc.id, update_err)
            
            logger.error(
                "Ingestion failed for session=%s file_name=%s",
                session_id,
                filename,
                exc_info=True,
                extra={
                    "session_id": str(session_id),
                    "doc_filename": filename,  
                    "file_size_mb": len(file_bytes) / 1024 / 1024,
                    "error_type": type(exc).__name__,
                },
            )
            raise IngestionError(str(doc.id if doc else "unknown"), str(exc)) from exc
    
    async def delete_document(self, document_id: UUID, session_id: UUID) -> None:
        """
        Delete a document and all its chunks + embeddings.
        Chunk/embedding rows are removed by DB CASCADE.
        IF EXISTS: Blob file is deleted from storage.

        Raises:
            NotFoundError: if document does not exist or doesn't belong to session
        """
        # Verify document exists and belongs to the session
        doc = await self.db.documents.get_by_id_and_session(document_id, session_id)
        if doc is None:
            raise NotFoundError("Document", str(document_id))

        # Delete blob file from storage if it exists
        if doc.blob_path:
            try:
                blob_key = f"{session_id}/{doc.filename}"
                await self._blob.delete(blob_key)
                logger.info("Blob file deleted: %s", blob_key)
            except Exception as blob_err:
                # Log warning but don't fail the whole operation
                logger.warning(
                    "Failed to delete blob %s: %s", 
                    blob_key, 
                    blob_err,
                    extra={"doc_id": str(document_id), "session_id": str(session_id)}
                )

        await self.db.documents.delete(document_id)
        logger.info(
            "Document deleted successfully: %s", 
            document_id,
            extra={"doc_id": str(document_id), "session_id": str(session_id)}
        )