from .sqlalchemy_base import SqlAlchemyBase

from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, DateTime, Text, Boolean, ForeignKey, Index, func
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
from typing import Optional
import uuid


class Session(SqlAlchemyBase):
    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)          # fix: was Mapped[str] but nullable
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="sessions") # type: ignore

    documents: Mapped[list["Document"]] = relationship( # type: ignore
        "Document", back_populates="session", cascade="all, delete-orphan"
    )
    chunks: Mapped[list["Chunk"]] = relationship( # type: ignore
        "Chunk", back_populates="session", cascade="all, delete-orphan"
    )
    embeddings: Mapped[list["Embedding"]] = relationship( # type: ignore
        "Embedding", back_populates="session", cascade="all, delete-orphan"
    )
    messages: Mapped[list["Message"]] = relationship( # type: ignore
        "Message", back_populates="session", cascade="all, delete-orphan"
    )
    evaluations: Mapped[list["Evaluation"]] = relationship( # type: ignore
        "Evaluation", back_populates="session", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_sessions_user_id", "user_id"),
        Index("idx_sessions_created", "created_at"),
    )