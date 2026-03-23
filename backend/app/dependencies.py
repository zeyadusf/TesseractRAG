from app.core.retrieval.reranker import CrossEncoderReranker
from app.core.ingestion.embedder import Embedder
from app.core.generation.llm_client import HuggingFaceClient
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

@lru_cache
def get_embedder():
    _embedder = Embedder()
    return _embedder

@lru_cache
def get_reranker():
    _reranker = CrossEncoderReranker()
    return _reranker

@lru_cache
def get_llm_client():
    _llm_client = HuggingFaceClient()
    return _llm_client