"""Claude chat help me here -_-"""
from __future__ import annotations

import threading
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core import get_config
from backend.storage.vector_db.base import BaseVectorStore

_settings = get_config()

# ------------------------------------------------------------------
# Singleton container — avoids module-level `global` mutation
# ------------------------------------------------------------------
_faiss_lock = threading.Lock()
_registry: dict[str, BaseVectorStore] = {}
#
# Why a lock here but not in the old version?
# FAISSStore.__init__ creates directories and loads config — not free.
# Two threads hitting a cold start simultaneously would both pass the
# `if _faiss_instance is None` check and construct two instances.
# The lock makes construction atomic; after the first build every
# subsequent call just reads from the dict (fast path, no lock needed
# because dict reads are GIL-safe and we never delete keys).
# ------------------------------------------------------------------


def get_vector_store(session: AsyncSession | None = None) -> BaseVectorStore:
    """
    Returns the configured vector store backend.

    FAISSStore  → singleton (thread-safe, lazy init).
                  FAISSStore manages its own per-session indexes internally,
                  so one process-level instance is correct.

    PgVectorStore → fresh instance per call.
                    It owns an AsyncSession whose lifetime is one HTTP request;
                    caching it would leak a stale session across requests.
    """
    backend = _settings.DEFAULT_VECTOR_STORE

    if backend == "pgvector":
        if session is None:
            raise ValueError(
                "pgvector backend requires an AsyncSession — "
                "pass the current request session via dependency injection."
            )
        from backend.storage.vector_db.pgvector.pgvector_store import PgVectorStore
        return PgVectorStore(session)  # never cached; session is request-scoped

    if backend == "faiss":
        # Fast path — already initialised (no lock needed; dict read is GIL-safe)
        if "faiss" in _registry:
            return _registry["faiss"]

        # Slow path — first call, construct under lock
        with _faiss_lock:
            # Double-checked locking: another thread may have built it
            # between our read above and acquiring the lock
            if "faiss" not in _registry:
                from backend.storage.vector_db.faiss.faiss_store import FAISSStore
                _registry["faiss"] = FAISSStore()

        return _registry["faiss"]

    raise NotImplementedError(
        f"Vector store backend '{backend}' is not supported. "
        "Valid options: 'faiss', 'pgvector'."
    )