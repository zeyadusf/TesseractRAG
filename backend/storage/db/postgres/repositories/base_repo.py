from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from uuid import UUID

ModelT = TypeVar("ModelT")


class BaseRepository(ABC, Generic[ModelT]):

    @abstractmethod
    async def get_by_id(self, record_id: UUID) -> ModelT | None:
        """Return a single record by primary key, or None if not found."""
        ...

    @abstractmethod
    async def create(self, **kwargs) -> ModelT:
        """Insert a new record. Returns the ORM model with all DB-generated fields."""
        ...

    @abstractmethod
    async def update(self, record_id: UUID, **kwargs) -> ModelT | None:
        """Update fields on an existing record. Returns None if not found."""
        ...

    @abstractmethod
    async def delete(self, record_id: UUID) -> bool:
        """Delete a record by PK. Returns True if it existed, False if not."""
        ...
