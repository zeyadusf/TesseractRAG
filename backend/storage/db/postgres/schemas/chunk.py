from .sqlalchemy_base import SqlAlchemyBase

from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import (
    String, Integer, Text, DateTime, ForeignKey,
    UniqueConstraint, Index, func,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from datetime import datetime
from uuid import uuid4


class Chunk(SqlAlchemyBase):
    __tablename__ = "chunks"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    document_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    session_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )

    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_size: Mapped[int] = mapped_column(Integer, nullable=False)
    word_count: Mapped[int] = mapped_column(Integer, nullable=False)
    chunker_type: Mapped[str] = mapped_column(String(64), nullable=False) 
    chunk_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    created_at: Mapped[datetime] = mapped_column(         
        DateTime(timezone=True), server_default=func.now(), nullable=False 
    )

    # Relationships
    # Fix: no cascade on the many→one back-ref
    document: Mapped["Document"] = relationship("Document", back_populates="chunks") # type: ignore 
    session: Mapped["Session"] = relationship("Session", back_populates="chunks") # type: ignore
    embeddings: Mapped[list["Embedding"]] = relationship( # type: ignore
        "Embedding", back_populates="chunk", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunk_document_index"),
        Index("idx_chunks_document_id", "document_id"),
        Index("idx_chunks_session_id", "session_id"),
        # GIN index for JSONB metadata filtering (language, file_ext, etc.)
        # Defined here with postgresql_using; Alembic will emit the correct DDL
        Index(
            "idx_chunks_metadata",
            "chunk_metadata",
            postgresql_using="gin",
            postgresql_ops={"chunk_metadata": "jsonb_path_ops"},
        ),
    )