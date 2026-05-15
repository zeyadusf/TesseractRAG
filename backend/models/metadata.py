from pydantic import BaseModel
from typing import Optional
class Metadata(BaseModel):
    source: str
    ext: str
    language: str
    chars: int
    pages: Optional[int] = None
    document_id: Optional[str] = None  

class ChunkMetadata(Metadata):
    chunk_index: int
    word_count: int
    chunk_size: int  
    chunker:str