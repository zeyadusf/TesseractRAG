from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class RetrievalStrategy(str, Enum):
    """Retrieval strategy options for the RAG pipeline."""
    AUTO = "auto"
    HYBRID = "hybrid"
    SEMANTIC = "semantic"
    LEXICAL = "lexical"


class ChatRequest(BaseModel):
    """
    What the user sends when asking a question.
    """
    question: str = Field(
        ...,
        min_length=1,
        description="The user's question about their documents"
    )
    strategy: RetrievalStrategy = Field(
        RetrievalStrategy.AUTO,
        description="Retrieval strategy to use"
    )
    show_context: bool = Field(
        True,
        description="Whether to return source chunks with the answer"
    )


class SourceChunk(BaseModel):
    """
    One retrieved chunk shown in the sources panel.
    """
    document_name: str
    content: str
    chunk_index: int = Field(0, ge=0)
    relevance_score: float = Field(0.0, ge=0.0, le=1.0)


class ChatResponse(BaseModel):
    """
    What the server returns after answering a question.
    """
    answer: str
    sources: list[SourceChunk] = Field(default_factory=list)
    strategy_used: RetrievalStrategy = RetrievalStrategy.AUTO
    retrieval_ms: int = Field(0, ge=0)