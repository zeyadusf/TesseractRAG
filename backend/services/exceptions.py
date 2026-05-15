"""
Domain Exceptions
Services raise these; routes catch them and return the correct HTTP status.

This keeps HTTP concerns (status codes) out of the service layer entirely.
"""

from __future__ import annotations


class NotFoundError(Exception):
    """Resource does not exist or the caller is not authorised to see it."""
    def __init__(self, resource: str, identifier: str | None = None) -> None:
        msg = f"{resource} not found"
        if identifier:
            msg += f": {identifier}"
        super().__init__(msg)
        self.resource = resource


class ConflictError(Exception):
    """Resource already exists (e.g. duplicate email, username)."""
    def __init__(self, message: str) -> None:
        super().__init__(message)


class ValidationError(Exception):
    """Business-rule validation failed (distinct from Pydantic schema validation)."""
    def __init__(self, message: str) -> None:
        super().__init__(message)


class AuthenticationError(Exception):
    """Credentials are wrong or token is invalid."""
    def __init__(self, message: str = "Invalid credentials") -> None:
        super().__init__(message)


class PermissionError(Exception):
    """Authenticated user lacks permission for this operation."""
    def __init__(self, message: str = "Permission denied") -> None:
        super().__init__(message)


class IngestionError(Exception):
    """Document ingestion pipeline failed."""
    def __init__(self, document_id: str, reason: str) -> None:
        super().__init__(f"Ingestion failed for {document_id}: {reason}")
        self.document_id = document_id
        self.reason = reason
