from .sqlalchemy_base import SqlAlchemyBase

from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import (
    String, Integer, DateTime, Text, ForeignKey,
    CheckConstraint, Index, func,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from datetime import datetime
from typing import Optional
from uuid import uuid4


class Message(SqlAlchemyBase):
    __tablename__ = "messages"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )

    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    retrieval_strategy: Mapped[Optional[str]] = mapped_column(String(50))

    source_chunks: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    # Generation metadata
    llm_model: Mapped[Optional[str]] = mapped_column(String(200))
    prompt_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    completion_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer)
    embedding_model: Mapped[Optional[str]] = mapped_column(String(200))
    retrieval_latency_ms: Mapped[Optional[int]] = mapped_column(Integer)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    session: Mapped["Session"] = relationship("Session", back_populates="messages")  # type: ignore
    evaluation: Mapped[Optional["Evaluation"]] = relationship(  # type: ignore
        "Evaluation",
        back_populates="message",
        uselist=False,
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant', 'system')", name="ck_message_role"),
        Index("idx_messages_session_id", "session_id"),
        Index("idx_messages_session_created", "session_id", "created_at"),
        Index("idx_messages_session_role", "session_id", "role"),
    )