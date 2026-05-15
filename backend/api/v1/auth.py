from fastapi import APIRouter, Depends, status, Request
from fastapi.security import OAuth2PasswordRequestForm

from backend.models.auth import (
    RegisterInput, LoginInput, TokenPair,
    UserOut, ChangePasswordInput, RefreshInput
)
from backend.core.limiter import limiter
from backend.services.auth_service import AuthService
from backend.core.security.jwt_deps import get_current_active_user
from backend.core.dependencies import get_db


def get_auth_service(db=Depends(get_db)):
    return AuthService(db)

router = APIRouter()
"""
Auth router — public + authenticated endpoints.

Routes
------
POST /api/v1/auth/register        → create account, return tokens
POST /api/v1/auth/login           → OAuth2 form login, return tokens
POST /api/v1/auth/refresh         → exchange refresh token for new pair
GET  /api/v1/auth/me              → current user profile
PUT  /api/v1/auth/me/password     → change password
POST /api/v1/auth/logout          → (stateless) client-side token discard
"""

@router.post("/register", response_model=TokenPair, status_code=status.HTTP_201_CREATED)
@limiter.limit("5/hour")
async def register(
        request: Request,
        data: RegisterInput,
        service: AuthService = Depends(get_auth_service),
    ):
    user, tokens = await service.register(data)
    return {
        "user": user,
        "access_token": tokens.access_token,
        "refresh_token": tokens.refresh_token,
        "token_type": tokens.token_type,
    }

@router.post("/login", response_model=TokenPair)
@limiter.limit("3/minute")
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    service: AuthService = Depends(get_auth_service),
):
    # return await service.authenticate_and_issue_tokens(payload) # when admin endpoint done}

    data = LoginInput(email=form_data.username, password=form_data.password)
    user, tokens = await service.login(data)
    return {
        "user": user,
        "access_token": tokens.access_token,
        "refresh_token": tokens.refresh_token,
        "token_type": tokens.token_type,
    }

@router.post("/refresh", response_model=TokenPair)
@limiter.limit("30/minute")
async def refresh_token(
    request: Request,
    refresh_token: RefreshInput,
    service: AuthService = Depends(get_auth_service),
):
    return await service.refresh(refresh_token.refresh_token)

@router.get("/me", response_model=UserOut)
@limiter.limit("60/minute")
async def get_me(
    request: Request,
    user: UserOut = Depends(get_current_active_user)
):
    return user

@router.post("/me/password")
@limiter.limit("5/minute")
async def change_password(
    request: Request,
    data: ChangePasswordInput,
    user: UserOut = Depends(get_current_active_user),
    service: AuthService = Depends(get_auth_service),
):
    await service.change_password(
        user_id=user.id,
        current_password=data.current_password,
        new_password=data.new_password,
    )
    return {"message": "Password updated successfully"}

@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit("30/minute")
async def logout(
    request: Request,
    _: UserOut = Depends(get_current_active_user),
) -> None:
    """
    JWT is stateless; the server cannot invalidate a token.
    The client is responsible for discarding both tokens.
    A token-blocklist / Redis approach can be layered in here later.
    """
    return

@router.post("/guest", response_model=TokenPair)
@limiter.limit("5/minute")
async def create_guest(
    request: Request,
    # form_data: OAuth2PasswordRequestForm = Depends(),
    service: AuthService = Depends(get_auth_service),
):
    # return await service.authenticate_and_issue_tokens(payload) # when admin endpoint done}

    # data = LoginInput(email=form_data.username, password=form_data.password)
    user, tokens = await service.create_guest_user()
    return {
        "user": user,
        "access_token": tokens.access_token,
        "refresh_token": tokens.refresh_token,
        "token_type": tokens.token_type,
    }
