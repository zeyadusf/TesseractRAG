from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class SessionCreate(BaseModel):
    """
    What the user sends when creating a new session.
    Only name and description — server generates everything else.
    """
    name: str = Field(
        ...,                   
        min_length=3,
        max_length=50,
        description="Session name"
    )
    description: Optional[str] = Field(
        None,                   
        description="Session description"
    )


class SessionResponse(BaseModel):
    """
    What the server returns about a session.
    Contains all fields including server-generated ones.
    """
    id: str
    name: str
    description: Optional[str] = None
    created_at: datetime
    document_count: int = 0
    message_count: int = 0