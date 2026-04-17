from sqlalchemy import Column, String, ForeignKey, Text, DateTime, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import uuid
from backend.app.schemas.db.models.tesseractrag_base import SQLAlchemyBase


class Message(SQLAlchemyBase):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # ForeignKey: Use UUID type to match sessions.id
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), 
                        nullable=False, index=True)

    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)

    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationship: back_populates must match Sessions.messages
    session = relationship("Sessions", back_populates="messages")

    # Table-level indexes
    __table_args__ = (
        Index('idx_session_id', 'session_id'),  # ForeignKey lookup
        Index('idx_session_timestamp', 'session_id', 'timestamp'),  # Fetch conversation chronologically
    )