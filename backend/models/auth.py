
from __future__ import annotations
from pydantic import BaseModel,EmailStr,Field,ConfigDict
from uuid import UUID

class RegisterInput(BaseModel):
    email: EmailStr
    username: str
    password: str 

class LoginInput(BaseModel):
    email: EmailStr = Field(
        ..., 
        validation_alias="username",  
        alias="username"            
    )
    password: str
    
    model_config = ConfigDict(
        populate_by_name=True,     
        extra="ignore"               
)

class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class UserOut(BaseModel):
    id: UUID
    email: str
    username: str
    is_active: bool
    is_superuser: bool

    model_config = {"from_attributes": True}

# api models
class RefreshInput(BaseModel):
    refresh_token: str
class ChangePasswordInput(BaseModel):
    current_password: str
    new_password: str