"""
dependencies.py
---------------
FastAPI dependency injection for TesseractRAG.

All singleton services are created here via lru_cache and injected into
route handlers using FastAPI's Depends() system.

Owner Identity (v1.1):
    get_owner_id() is a new FastAPI dependency that reads the X-Owner-ID
    header from every incoming request. This header is set by the browser
    on first visit using a randomly generated UUID stored in localStorage.

    It is NOT authentication — there is no signature, no secret, no server-
    side user record. It is anonymous identity: a stable browser-scoped UUID
    that lets the backend filter sessions by "owner" without requiring login.

    If the header is missing the request is rejected with HTTP 400.
    The frontend must always send this header — it is set once on page load
    and attached to every fetch() call.
"""

from fastapi import Header, HTTPException
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
    return SessionManager()


@lru_cache
def get_embedder() -> Embedder:
    return Embedder()


@lru_cache
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()


@lru_cache
def get_llm_client() -> HuggingFaceClient:
    return HuggingFaceClient()


async def get_owner_id(x_owner_id: str = Header(...)) -> str:

    owner_id = x_owner_id.strip()
    # Minimal format guard — must be non-empty after stripping whitespace.
    # We intentionally do NOT validate UUID format here: strict UUID
    # validation would break clients that use other ID formats, and the
    # security model does not depend on UUID unpredictability (there is no
    # server-side secret). The only invariant we need is non-empty.
    if not owner_id:
        raise HTTPException(
            status_code=400,
            detail="X-Owner-ID header must not be empty.",
        )

    return owner_id
