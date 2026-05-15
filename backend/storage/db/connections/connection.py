"""
connection DataBase :
here make engine and sessions for db providers :
    - postgres -_^ -> used lifespan in main.py 

    
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from backend.core import get_config

config = get_config()

postgres_conn = (
    f'postgresql+asyncpg://{config.POSTGRES_USERNAME}:'
    f'{config.POSTGRES_PASSWORD}@{config.POSTGRES_HOST}:'
    f'{config.POSTGRES_PORT}/{config.POSTGRES_DATABASE_NAME}'
)

engine = create_async_engine(
    postgres_conn,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

SessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)