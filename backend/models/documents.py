from pydantic import BaseModel
from datetime import datetime
from uuid import UUID
class DocumentOut(BaseModel):
    id: UUID
    session_id: UUID
    filename: str
    file_extension: str
    file_size_bytes: int
    chunk_count: int
    language: str | None
    status: str
    uploaded_at: datetime
    indexed_at: datetime | None
    chunk_count:int
    blob_path:str|None

    model_config = {"from_attributes": True}

class DeleteResponse(BaseModel):
    message: str
    deleted_id: UUID