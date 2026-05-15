"""
FastAPI dependency functions for authentication.

Dependency chain (innermost → outermost):
    get_db
        └─ get_current_user          ← validates token, fetches user
               └─ get_current_active_user   ← rejects deactivated accounts
                      └─ get_current_superuser  ← admin-only routes

Usage in routes:
    from backend.core.security.dependencies import (
        get_current_active_user,
        get_current_superuser,
    )

    @router.get("/me")
    async def me(user: UserOut = Depends(get_current_active_user)):
        return user

    @router.delete("/admin/users/{user_id}")
    async def delete_user(user: UserOut = Depends(get_current_superuser)):
        ...
"""

from __future__ import annotations
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from backend.core import get_config
from backend.core.dependencies import get_db
from backend.core.security.jwt import decode_token
from backend.models.auth import  UserOut
from backend.services.exceptions import NotFoundError
from backend.storage.db.db_dispatcher import DBDispatcher

# Tells FastAPI where tokens come from.
# tokenUrl must match your login route path exactly —
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")



# ── Token → User ───────────────────────────────────────────────────

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: DBDispatcher = Depends(get_db),
) -> UserOut:
    """
    Extract and validate the Bearer token, then return the matching user.

    Steps:
      1. OAuth2PasswordBearer pulls the token from the Authorization header.
      2. decode_token verifies signature and expiry.
      3. Payload type must be "access" (not "refresh") — prevents refresh
         tokens from being used as access tokens.
      4. AuthService.get_current_user fetches the user row.

    Raises:
        401 Unauthorized: token missing, invalid, expired, wrong type,
                          or user no longer exists.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    settings = get_config()
    payload = decode_token(
        token,
        settings.SECRET_KEY,
        settings.JWT_ALGORITHM,
    )

    # Reject missing/invalid tokens and any non-access token type
    # (guards against a refresh token being used as an access token)
    if payload is None or payload.get("type") != "access":
        raise credentials_exception

    try:
        user_id = UUID(payload["sub"])
    except (KeyError, ValueError):
        # "sub" missing or not a valid UUID
        raise credentials_exception

    try:
        from backend.services.auth_service import AuthService
        return await AuthService(db).get_current_user(user_id)
    except NotFoundError:
        # User was deleted after the token was issued
        raise credentials_exception


# ── Active user guard ──────────────────────────────────────────────

async def get_current_active_user(
    current_user: UserOut = Depends(get_current_user),
) -> UserOut:
    """
    Extends get_current_user by also rejecting deactivated accounts.

    Use this (not get_current_user) for every route that acts on
    behalf of a normal user.

    Raises:
        403 Forbidden: account exists but is_active is False.
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )
    return current_user


# ── Superuser guard ────────────────────────────────────────────────

async def get_current_superuser(
    current_user: UserOut = Depends(get_current_active_user),
) -> UserOut:
    """
    Restricts a route to superusers only.

    Raises:
        403 Forbidden: user is active but is_superuser is False.
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    return current_user