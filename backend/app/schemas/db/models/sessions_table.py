from sqlalchemy import Column, String, DateTime, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import uuid
from .tesseractrag_base import SQLAlchemyBase


class Sessions(SQLAlchemyBase):
    __tablename__ = "sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    name = Column(String, nullable=False)
    description = Column(String, nullable=True)

    owner_id = Column(String, nullable=False, index=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")
    chunks = relationship("Chunk", back_populates="session", cascade="all, delete-orphan")
    evaluations = relationship("Evaluation", back_populates="session", cascade="all, delete-orphan")

    # Table-level indexes
    __table_args__ = (
        Index('idx_owner_id', 'owner_id'),
        Index('idx_owner_created', 'owner_id', 'created_at'),  # For fetching user sessions chronologically
    )