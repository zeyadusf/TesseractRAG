from __future__ import annotations
from backend.core import get_config
from sqlalchemy.ext.asyncio import AsyncSession

settings = get_config()


class DBDispatcher:

    def __init__(self, session: AsyncSession):
        self.session = session
        self._load_repos(session)

    def _load_repos(self, session: AsyncSession):
        if settings.DEFAULT_DB == "postgres":
            from backend.storage.db.postgres.repo_factory import init_postgres_repos
            repos = init_postgres_repos(session)

        else:
            raise ValueError(f"Unsupported DB: {settings.DEFAULT_DB}")

        # attach dynamically
        for name, repo in repos.items():
            setattr(self, name, repo)