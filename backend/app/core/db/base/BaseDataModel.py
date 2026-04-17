from abc import ABC, abstractmethod
from config import get_settings


class BaseDataModel(ABC):
    def __init__(self, db_client: object):
        self.db_client = db_client
        self.app_settings = get_settings()

    @classmethod
    @abstractmethod
    async def create_instance(cls, db_client: object) -> "BaseDataModel":
        pass