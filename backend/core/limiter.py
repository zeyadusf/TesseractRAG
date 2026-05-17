from slowapi import Limiter
from slowapi.util import get_remote_address
from backend.core.config import get_config
config = get_config()
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=config.REDIS_URL
)