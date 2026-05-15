
from __future__ import annotations
from backend.storage.db.db_dispatcher import DBDispatcher
from backend.core import get_config
class BaseService:

    def __init__(self, db: DBDispatcher) -> None:
        self.db = db
        self._config = get_config()

