"""
Flow:
  Register: validate uniqueness → hash password → create User ORM → return tokens
  Login:    fetch user by email → verify password → return tokens
  Refresh:  decode refresh token → return new access token

This service owns all authentication logic. No route should call
passlib or jose directly — that all lives here.
"""

from __future__ import annotations
from uuid import UUID,uuid4
from datetime import datetime

from backend.core.security.jwt import create_access_token, create_refresh_token, decode_token
from backend.core.security.password import hash_password, verify_password
from backend.storage.db.db_dispatcher import DBDispatcher
from backend.storage.db.postgres.schemas.user import User
from backend.models.auth import RegisterInput,UserOut,LoginInput,TokenPair

from .base_service import BaseService
from .exceptions import AuthenticationError, ConflictError, NotFoundError, ValidationError

class AuthService(BaseService):
    def __init__(self, db: DBDispatcher) -> None:
        super().__init__(db)

    # ── Register 
    async def register(self, data: RegisterInput) -> tuple[UserOut, TokenPair]:
        """
        Create a new user account and return the user + token pair.
        Raises:
            ConflictError: if email or username is already taken.
        """
        if await self.db.users.is_email_exists(data.email):
            raise ConflictError(f"Email already registered: {data.email}")

        if await self.db.users.is_username_exists(data.username):
            raise ConflictError(f"Username already taken: {data.username}")

        user: User = await self.db.users.create(
            email=data.email,
            username=data.username,
            hashed_password=hash_password(data.password),
        )

        tokens = self._issue_tokens(user.id)
        return UserOut.model_validate(user), tokens # cur , refreshed

    # ── Login 
    async def login(self, data: LoginInput) -> tuple[UserOut, TokenPair]:
        """
        Authenticate a user by email + password.
        Raises:
            AuthenticationError: if credentials are wrong or account is inactive.
        """
        user = await self.db.users.get_by_email(data.email)

        # Use the same error message for wrong email AND wrong password.
        # Different messages would allow user enumeration attacks.
        if user is None or not verify_password(data.password, user.hashed_password):
            raise AuthenticationError("Invalid email or password")

        if not user.is_active:
            raise AuthenticationError("Account is deactivated")

        tokens = self._issue_tokens(user.id)
        return UserOut.model_validate(user), tokens

    # ── Refresh token 
    async def refresh(self, refresh_token: str) -> TokenPair:
        """
        Decode a valid refresh token and issue a new token pair.
        Raises:
            AuthenticationError: if the token is invalid, expired, or belongs to a deactivated user.
        """
        payload = decode_token(
            refresh_token,
            self._config.SECRET_KEY,
            self._config.JWT_ALGORITHM,
        )
        if payload is None or payload.get("type") != "refresh":
            raise AuthenticationError("Invalid or expired refresh token")

        user_id = UUID(payload["sub"])
        user = await self.db.users.get_by_id(user_id)

        if user is None or not user.is_active:
            raise AuthenticationError("User not found or deactivated")

        return self._issue_tokens(user_id)

    # ── Profile 
    async def get_current_user(self, user_id: UUID) -> UserOut:
        """
        Fetch the authenticated user's profile.
        Called by the get_current_user FastAPI dependency in
        core/security/dependencies.py — not directly by routes.

        Raises:
            NotFoundError: if the user was deleted after token issuance.
        """
        user = await self.db.users.get_by_id(user_id)
        if user is None:
            raise NotFoundError("User", str(user_id))
        return UserOut.model_validate(user)

    async def change_password(
        self,
        user_id: UUID,
        current_password: str,
        new_password: str,
    ) -> None:
        """
        Verify current password then update to new hash.

        Raises:
            AuthenticationError: if current_password is wrong.
            NotFoundError:        if user does not exist.
            ValidationError:      if new password is the same as the current one.
        """
        user = await self.db.users.get_by_id(user_id)
        if user is None:
            raise NotFoundError("User", str(user_id))

        if not verify_password(current_password, user.hashed_password):
            raise AuthenticationError("Current password is incorrect")

        # FIX: guard against no-op password changes
        if verify_password(new_password, user.hashed_password):
            raise ValidationError("New password must differ from current password")

        await self.db.users.update(
            user_id,
            hashed_password=hash_password(new_password),
        )

    async def create_guest_user(self):
        guest_id = uuid4()
        guest_email = f"guest-{guest_id}@tesseract.ai.com"
        guest_username = f"guest_{str(guest_id)[:8]}"


        try:
            # Create guest user in database
            # user: User = await self.db.users.create(
            #     email=guest_email,
            #     username=guest_username,
            #     hashed_password=hash_password(str(guest_id)),  # Random hash, unused
            #     is_active=True,
            #     is_superuser=False,
                
            # )
            _ = await self.register(RegisterInput(
                email=guest_email,
                password=str(guest_id),
                username=guest_username
            ))
            
            return await self.login(LoginInput(
                username=guest_email,
                password=str(guest_id),
                    )
                )
        except ConflictError:
            # Email/username collision (should be extremely rare)
            raise ConflictError("Failed to create guest user (collision). Try again.")


# TODO: in next Vx.1.X change name and email 

    # ─────────────────────────────────────────── Internal helpers 

    def _issue_tokens(self, user_id: UUID) -> TokenPair:
        return TokenPair(
            access_token=create_access_token(
                user_id=user_id,
                secret_key=self._config.SECRET_KEY,
                algorithm=self._config.JWT_ALGORITHM,
                expires_minutes=self._config.ACCESS_TOKEN_EXPIRE_MINUTES,
            ),
            refresh_token=create_refresh_token(
                user_id=user_id,
                secret_key=self._config.SECRET_KEY,
                algorithm=self._config.JWT_ALGORITHM,
                expires_days=self._config.REFRESH_TOKEN_EXPIRE_DAYS,
            ),
        )