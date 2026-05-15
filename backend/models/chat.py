from __future__ import annotations

from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, field_validator, Field
from typing import Any

from .enums.retrieval_strategy import RetrievalStrategy

class ChatRequest(BaseModel):
    question: str
    strategy: RetrievalStrategy = RetrievalStrategy.AUTO
    show_context: bool = True
    metadata_filter: dict | None = None   

class ChatResponse(BaseModel):
    message_id: UUID
    question: str
    answer: str
    strategy_used: str
    sources: list[SourceChunkOut]
    retrieval_latency_ms: int
    total_latency_ms: int
    llm_model: str
    embedding_model: str

class ChatHistoryOut(BaseModel):
    session_id: UUID
    total: int                                
    page: int
    page_size: int
    turns: list[ChatTurnOut]


class SourceChunkOut(BaseModel):
    chunk_id: str | None = None
    content: str = ""                         
    score: float = 0.0                        
    source_doc: str = ""                       
    chunk_index: int = 0

    @classmethod
    def from_db(cls, data: dict) -> "SourceChunkOut":
        """Handle both DB format and API format."""
        return cls(
            chunk_id=str(data.get("chunk_id") or data.get("id") or ""),
            content=data.get("content", ""),
            score=float(data.get("score") or 0.0),
            source_doc=str(
                data.get("source_doc")
                or data.get("document_name")
                or ""
            ),
            chunk_index=int(data.get("chunk_index") or 0),
        )


class ChatTurnOut(BaseModel):
    id: UUID
    session_id: UUID
    role: str
    content: str
    created_at: datetime
    retrieval_strategy: str | None = None
    source_chunks: list[SourceChunkOut] = Field(default_factory=list)
    llm_model: str | None = None
    latency_ms: int | None = None

    @field_validator("source_chunks", mode="before")
    @classmethod
    def normalize_source_chunks(cls, v: Any) -> list:
        """Convert raw JSONB dicts from DB → SourceChunkOut."""
        if not v:
            return []
        return [
            SourceChunkOut.from_db(item) if isinstance(item, dict) else item
            for item in v
        ]

    class Config:
        from_attributes = True