from .sqlalchemy_base import SqlAlchemyBase

from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import (
    String, Integer, DateTime, ForeignKey,
    UniqueConstraint, Index, func, text,
)
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from datetime import datetime
from typing import Optional
from uuid import uuid4


class Embedding(SqlAlchemyBase):                       
    __tablename__ = "embeddings"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    chunk_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chunks.id", ondelete="CASCADE"),    
        nullable=False,
    )
    session_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )

    model_name: Mapped[str] = mapped_column(
        String(200), nullable=False                     
    )
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    token_usage: Mapped[Optional[int]] = mapped_column(Integer)  
    vector: Mapped[Vector] = mapped_column(Vector(dim=512), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    chunk: Mapped["Chunk"] = relationship("Chunk", back_populates="embeddings") # type: ignore
    session: Mapped["Session"] = relationship("Session", back_populates="embeddings") # type: ignore

    __table_args__ = (
        UniqueConstraint("chunk_id", "model_name", name="uq_embedding_chunk_model"),
        Index("idx_embeddings_session_id", "session_id"),
        Index("idx_embeddings_model", "model_name"),
        # HNSW index for approximate cosine similarity search
        # postgresql_with maps to WITH (m=16, ef_construction=64)
        Index(
            "idx_embeddings_hnsw",
            "vector",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"vector": "vector_cosine_ops"},
        ),
    )