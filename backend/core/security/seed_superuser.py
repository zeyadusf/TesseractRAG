from backend.core.config import get_config
from backend.core.logger import get_logger
from sqlalchemy.ext.asyncio import AsyncSession

logger=get_logger(__name__)
settings = get_config()
async def seed_superuser(session: AsyncSession):
    from backend.storage.db.postgres.schemas.user import User
    from backend.core.security.password import hash_password
    from sqlalchemy.future import select

    if not settings.SUPERUSER_EMAIL:
        return  

    result = await session.execute(
        select(User).where(User.email == settings.SUPERUSER_EMAIL)
    )
    if result.scalar_one_or_none():
        return  
    user = User(
        email=settings.SUPERUSER_EMAIL,
        username=settings.SUPERUSER_USERNAME,
        hashed_password=hash_password(settings.SUPERUSER_PASSWORD),
        is_active=True,
        is_superuser=True,
    )
    session.add(user)
    await session.commit()
    logger.info(f"Superuser '{user.username}' seeded from environment.")