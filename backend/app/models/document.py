from pydantic import BaseModel, Field
from datetime import datetime, timezone


class DocumentInfo(BaseModel):
    """
    Returned after a document is uploaded and indexed.
    Represents one document within a session.
    """
    filename: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Original uploaded filename"
    )
    file_size_bytes: int = Field(
        0,
        description="File size in bytes"
    )
    chunk_count: int = Field(
        0,
        description="Number of text chunks created from this document"
    )
    uploaded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of when document was uploaded"
    )