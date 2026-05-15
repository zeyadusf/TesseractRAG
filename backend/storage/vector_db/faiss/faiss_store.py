"Cloud's chat isn't helping me here, but he almost wrote it x_x"
from __future__ import annotations

import faiss
import numpy as np
import os
import threading
from collections import defaultdict, OrderedDict
from uuid import UUID

from backend.core.logger import get_logger
from backend.core.config import get_settings
from backend.storage.vector_db.base import BaseVectorStore

logger = get_logger(__name__)
settings = get_settings()

_MAX_CACHED_SESSIONS = 100  # evict LRU sessions beyond this threshold


class FAISSStore(BaseVectorStore):
    """
    Production-ready FAISS store with:
    - per-session index & per-session lock (no global bottleneck)
    - atomic disk writes (crash-safe)
    - LRU in-memory cache (bounded RAM)
    - model_name guard (prevents mixed-model search garbage)
    - cosine similarity via L2-normalised IndexFlatIP

    Single-node only. For multi-node or large-scale workloads use pgvector/Qdrant.
    """

    def __init__(self) -> None:
        self.dim = settings.EMBED_DIM
        self.base_path = settings.FAISS_PATH or "./faiss_indexes"

        os.makedirs(self.base_path, exist_ok=True)

        # OrderedDict used as LRU cache: most-recently-used moves to end
        self._indexes: OrderedDict[str, dict] = OrderedDict()

        # defaultdict gives us per-key lock creation without a hand-rolled helper.
        # defaultdict.__missing__ uses the GIL for the insertion itself — safe.
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)

        # Separate coarse lock only for _indexes structure mutations
        # (eviction, key insertion) — NOT held during FAISS I/O
        self._cache_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _key(self, session_id: UUID) -> str:
        return str(session_id)

    def _index_path(self, key: str) -> str:
        return os.path.join(self.base_path, f"{key}.index")

    def _meta_path(self, key: str) -> str:
        return os.path.join(self.base_path, f"{key}.meta.npy")

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------

    def _create_index(self) -> faiss.IndexFlatIP:
        """
        IndexFlatIP: exact inner-product search.
        Vectors are L2-normalised before insertion → IP == cosine similarity.
        No training required; reconstruct(i) works out of the box.
        """
        return faiss.IndexFlatIP(self.dim)

    def _load_or_create(self, key: str) -> dict:
        """
        Returns the in-memory store for *key*, loading from disk if needed.
        Caller must already hold self._locks[key].
        LRU eviction is applied when cache exceeds _MAX_CACHED_SESSIONS.
        """
        with self._cache_lock:
            if key in self._indexes:
                self._indexes.move_to_end(key)  # mark as recently used
                return self._indexes[key]

        # --- load / create outside cache_lock (FAISS I/O can be slow) ---
        index_path = self._index_path(key)
        meta_path = self._meta_path(key)

        if os.path.exists(index_path):
            index = faiss.read_index(index_path)
            meta = np.load(meta_path, allow_pickle=True)
            chunk_ids: list = list(meta[0])
            model_name: str = str(meta[1])
        else:
            index = self._create_index()
            chunk_ids = []
            model_name = ""

        store = {
            "index": index,
            "chunk_ids": chunk_ids,
            "model_name": model_name,
        }

        with self._cache_lock:
            self._indexes[key] = store
            self._indexes.move_to_end(key)
            # evict LRU entry if over limit
            while len(self._indexes) > _MAX_CACHED_SESSIONS:
                evicted_key, _ = self._indexes.popitem(last=False)
                logger.debug("FAISSStore: evicted session %s from cache", evicted_key)

        return store

    def _save(self, key: str, store: dict) -> None:
        """
        Atomic write: write to .tmp then os.replace → crash-safe.
        os.replace is atomic on POSIX; on Windows it's best-effort.
        """
        index_tmp = self._index_path(key) + ".tmp"
        meta_tmp = self._meta_path(key) + ".tmp"

        faiss.write_index(store["index"], index_tmp)
        os.replace(index_tmp, self._index_path(key))

        np.save(
            meta_tmp,
            np.array([store["chunk_ids"], store["model_name"]], dtype=object),
        )
        os.replace(meta_tmp, self._meta_path(key))

    # ------------------------------------------------------------------
    # Vector reconstruction
    # ------------------------------------------------------------------

    def _reconstruct(self, store: dict) -> np.ndarray:
        """IndexFlatIP supports reconstruct() natively — no make_direct_map needed."""
        index = store["index"]
        total = index.ntotal

        if total == 0:
            return np.empty((0, self.dim), dtype=np.float32)

        vectors = np.zeros((total, self.dim), dtype=np.float32)
        for i in range(total):
            vectors[i] = index.reconstruct(i)

        return vectors

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def upsert(
        self,
        session_id: UUID,
        chunk_id: UUID,
        vector: list[float],
        model_name: str,
        dimensions: int,
        token_usage: int | None = None,
    ) -> None:
        if dimensions != self.dim:
            raise ValueError(
                f"Dimension mismatch: expected {self.dim}, got {dimensions}."
            )

        key = self._key(session_id)

        with self._locks[key]:
            store = self._load_or_create(key)

            # Enforce model consistency within a session
            if store["model_name"] and store["model_name"] != model_name:
                raise ValueError(
                    f"Embedding model mismatch for session {session_id}: "
                    f"stored='{store['model_name']}', got='{model_name}'. "
                    "Re-embed the entire session or create a new one."
                )

            if not store["model_name"]:
                store["model_name"] = model_name

            vec = np.array([vector], dtype=np.float32)
            faiss.normalize_L2(vec)

            store["index"].add(vec)
            store["chunk_ids"].append(chunk_id)

            self._save(key, store)

    async def search(
        self,
        session_id: UUID,
        query_vector: list[float],
        model_name: str,
        top_k: int = 10,
    ) -> list[dict]:
        key = self._key(session_id)

        with self._locks[key]:
            store = self._load_or_create(key)
            index = store["index"]

            if index.ntotal == 0:
                return []

            if store["model_name"] and store["model_name"] != model_name:
                raise ValueError(
                    f"Embedding model mismatch for session {session_id}: "
                    f"stored='{store['model_name']}', search model='{model_name}'."
                )

            effective_k = min(top_k, index.ntotal)

            query = np.array([query_vector], dtype=np.float32)
            faiss.normalize_L2(query)

            scores, indices = index.search(query, effective_k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                results.append({
                    "chunk_id": store["chunk_ids"][idx],
                    "score": float(score),
                })

            return results

    async def delete_by_session(self, session_id: UUID) -> int:
        key = self._key(session_id)

        with self._locks[key]:
            with self._cache_lock:
                store = self._indexes.pop(key, None)

            for path in (self._index_path(key), self._meta_path(key)):
                if os.path.exists(path):
                    os.remove(path)

            return len(store["chunk_ids"]) if store else 0

    async def delete_by_chunk(self, chunk_id: UUID) -> int:
        """
        FAISS has no native delete — rebuild the index without the target vector.
        O(n·sessions) — acceptable for current scale; known limitation.
        """
        deleted = 0

        # Snapshot keys under cache_lock to avoid iterating a mutating dict
        with self._cache_lock:
            keys_snapshot = list(self._indexes.keys())

        for key in keys_snapshot:
            with self._locks[key]:
                # Re-fetch: session may have been evicted between snapshot and lock
                store = self._indexes.get(key)
                if store is None or chunk_id not in store["chunk_ids"]:
                    continue

                idx = store["chunk_ids"].index(chunk_id)
                vectors = self._reconstruct(store)

                vectors = np.delete(vectors, idx, axis=0)
                store["chunk_ids"].pop(idx)

                new_index = self._create_index()
                if len(vectors) > 0:
                    new_index.add(vectors)

                store["index"] = new_index
                self._save(key, store)

                deleted += 1

        return deleted