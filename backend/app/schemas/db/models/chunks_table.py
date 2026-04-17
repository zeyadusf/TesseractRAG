from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from app.config import get_settings
from backend.app.schemas.db.models.tesseractrag_base import SQLAlchemyBase

settings = get_settings()


class Chunk(SQLAlchemyBase):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)

    content = Column(Text, nullable=False)

    # Vector embedding - pgvector handles this
    embedding = Column(Vector(settings.DIM_EMBEDDING), nullable=False)

    metadata = Column(JSONB, nullable=True)

    chunk_index = Column(Integer, nullable=False)

    # ForeignKey: Use UUID type to match sessions.id
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), 
                        nullable=False, index=True)

    document_name = Column(String, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationship
    session = relationship("Sessions", back_populates="chunks")

    # Table-level indexes
    __table_args__ = (
        # HNSW: Best for <1M vectors (faster, lower memory)
        Index('idx_embedding_hnsw', 'embedding', postgresql_using='hnsw', 
                postgresql_with={'m': 16, 'ef_construction': 64}),
        
        # Index('idx_embedding_ivfflat', 'embedding', postgresql_using='ivfflat', 
        #       postgresql_with={'lists': 100}),

        # Composite indexes for filtering
        Index('idx_session_id', 'session_id'),  # ForeignKey lookup
        Index('idx_session_document', 'session_id', 'document_name'),  # Filter chunks by document
        Index('idx_session_chunk_index', 'session_id', 'chunk_index'),  # Order chunks
        Index('idx_created_at', 'created_at'),  # Time-based queries
    )