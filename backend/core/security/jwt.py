"""
JWT utilities — token creation and decoding.

Rules:
  - create_access_token  → short-lived (minutes), type="access"
  - create_refresh_token → long-lived  (days),    type="refresh"
  - decode_token         → used for BOTH types; callers check payload["type"]
"""

from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from jose import JWTError, jwt


def create_access_token(
    user_id: UUID,
    secret_key: str,
    algorithm: str,
    expires_minutes: int,
) -> str:
    """Create a short-lived JWT for API authentication."""
    expire = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)
    payload = {
        "sub": str(user_id),
        "type": "access",
        "iat": datetime.now(timezone.utc),
        "exp": expire,
    }
    return jwt.encode(payload, secret_key, algorithm=algorithm)


def create_refresh_token(
    user_id: UUID,
    secret_key: str,
    algorithm: str,
    expires_days: int,
) -> str:
    """Create a long-lived JWT for token renewal."""
    expire = datetime.now(timezone.utc) + timedelta(days=expires_days)
    payload = {
        "sub": str(user_id),
        "type": "refresh",
        "iat": datetime.now(timezone.utc),
        "exp": expire,
    }
    return jwt.encode(payload, secret_key, algorithm=algorithm)


def decode_token(
    token: str,
    secret_key: str,
    algorithm: str,
) -> Optional[dict]:
    """
    Decode and verify a JWT (access OR refresh).

    Returns the payload dict on success, or None if the token is
    invalid, expired, or tampered with.

    Callers are responsible for checking payload["type"] to ensure
    they received the correct token kind.

    Previously named decode_access_token — renamed because this
    function is also used to decode refresh tokens.
    """
    try:
        return jwt.decode(token, secret_key, algorithms=[algorithm])
    except JWTError:
        return None