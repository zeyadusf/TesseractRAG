from __future__ import annotations
from sqlalchemy.ext.asyncio import AsyncSession
from typing import AsyncGenerator
from backend.core.config import get_config
from backend.storage.db.postgres.postgres_provider import get_postgres_session




_DB_PROVIDERS = {
    "postgres": get_postgres_session,
}

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    settings = get_config()
    provider = _DB_PROVIDERS.get(settings.DEFAULT_DB)
    if provider is None:
        raise NotImplementedError(f"DB backend '{settings.DEFAULT_DB}' is not supported")
    async for session in provider():
        yield session
