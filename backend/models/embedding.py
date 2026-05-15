from pydantic import BaseModel
from typing import List

class EmbeddingChunk(BaseModel):
    index: int 
    text: str
    embedding: List[float]
    tokens: int
    is_estimate:bool


class EmbeddingMeta(BaseModel):
    model: str
    total_tokens: int
    total_chunks: int
    dimensions: int

