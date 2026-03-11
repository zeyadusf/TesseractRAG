from app.core.session_manager import SessionManager
from functools import lru_cache


@lru_cache
def get_session_manager() -> SessionManager:
    """
    Return the singleton SessionManager instance.
    
    Uses lru_cache to ensure only one SessionManager is created
    for the entire lifetime of the backend process.
    """
    manager = SessionManager()
    return manager
