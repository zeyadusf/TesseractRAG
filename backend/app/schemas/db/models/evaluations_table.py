from sqlalchemy import Column, Float, DateTime, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import uuid
from backend.app.schemas.db.models.tesseractrag_base import SQLAlchemyBase


class Evaluation(SQLAlchemyBase):
    __tablename__ = "evaluations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # ForeignKeys: Use UUID type to match their parent tables
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), 
                        nullable=False, index=True)
    message_id = Column(UUID(as_uuid=True), ForeignKey("messages.id", ondelete="CASCADE"), 
                        nullable=False, index=True)

    # Evaluation scores
    faithfulness = Column(Float, nullable=True)
    answer_relevancy = Column(Float, nullable=True)
    context_precision = Column(Float, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships: back_populates must match Sessions.evaluations and Message.evaluations
    session = relationship("Sessions", back_populates="evaluations")
    message = relationship("Message", back_populates="evaluations")

    # Table-level indexes
    __table_args__ = (
        Index('idx_session_id', 'session_id'),  # ForeignKey lookup
        Index('idx_message_id', 'message_id'),  # ForeignKey lookup
        Index('idx_session_created', 'session_id', 'created_at'),  # Recent evaluations for a session
        Index('idx_message_created', 'message_id', 'created_at'),  # Recent evaluations for a message
    )