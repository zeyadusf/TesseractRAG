from .sqlalchemy_base import SqlAlchemyBase

from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, DateTime, Text, ForeignKey, BigInteger, Integer, Index, func
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
from typing import Optional
import uuid


class Document(SqlAlchemyBase):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )

    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    file_extension: Mapped[str] = mapped_column(String(20), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)

    language: Mapped[Optional[str]] = mapped_column(String(10))
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="pending"
        # swap for DocumentStatus.PENDING if you import the enum
    )

    blob_path: Mapped[Optional[str]] = mapped_column(Text)

    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    indexed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Relationships
    session: Mapped["Session"] = relationship("Session", back_populates="documents") # type: ignore
    chunks: Mapped[list["Chunk"]] = relationship( # type: ignore
        "Chunk", back_populates="document", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_documents_session_id", "session_id"),
        Index("idx_documents_status", "status"),
    )