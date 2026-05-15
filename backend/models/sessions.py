from __future__ import annotations 
from pydantic import BaseModel, Field
from uuid import UUID
from typing import List, Optional  

class SessionCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(default=None, max_length=500)

class SessionUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    description: Optional[str] = Field(default=None, max_length=500)

class SessionOut(BaseModel):
    id: UUID
    user_id: UUID
    name: str
    description: Optional[str]
    is_active: bool
    document_count: int = 0
    message_count: int = 0

    model_config = {"from_attributes": True}
