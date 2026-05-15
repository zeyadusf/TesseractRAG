from sqlalchemy.ext.asyncio import AsyncSession
from backend.storage.db.connections.connection import SessionLocal
from typing import AsyncGenerator


async def get_postgres_session() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise