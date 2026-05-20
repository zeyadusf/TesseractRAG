from .sqlalchemy_base import SqlAlchemyBase

from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, Float, DateTime, ForeignKey, Index, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from datetime import datetime
from typing import Optional
from uuid import uuid4


class Evaluation(SqlAlchemyBase):
    __tablename__ = "evaluations"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    message_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    session_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )

    # RAGAS metric scores — NULL until evaluated
    faithfulness: Mapped[Optional[float]] = mapped_column(Float)
    answer_relevancy: Mapped[Optional[float]] = mapped_column(Float)
    context_precision: Mapped[Optional[float]] = mapped_column(Float)
    context_recall: Mapped[Optional[float]] = mapped_column(Float)

    evaluator: Mapped[str] = mapped_column(String(100), nullable=False, default="ragas")
    eval_model: Mapped[Optional[str]] = mapped_column(String(200))
    raw_result: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    message: Mapped["Message"] = relationship(  # type: ignore
        "Message",
        back_populates="evaluation",
        passive_deletes=True,
    )
    session: Mapped["Session"] = relationship(  # type: ignore
        "Session",
        back_populates="evaluations",
        passive_deletes=True,
    )

    __table_args__ = (
        Index("idx_evaluations_message_id", "message_id"),
        Index("idx_evaluations_session_id", "session_id"),
    )