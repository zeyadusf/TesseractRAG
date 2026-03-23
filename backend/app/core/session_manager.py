"""
session_manager.py
------------------
Core session management for TesseractRAG.

Contains:
    - SessionState  : In-memory representation of a single RAG session
    - SessionManager: Registry that creates, retrieves, lists, and deletes sessions

Storage backend: Cloudflare R2 (replaces local disk).
All persistence goes through R2Storage — no direct filesystem I/O here.
FAISS indexes are serialized to/from bytes via io.BytesIO for R2 transport.
"""

import io
import json
import faiss
from typing import Optional, Any
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import HTTPException

from app.models.session import SessionResponse
from app.config import get_settings as _config
from app.utils.logger import get_logger
from app.core.storage.r2_storage import R2Storage

from app.core.ingestion.parser import DocumentParser
from app.core.ingestion.chunker import RecursiveChunker
from app.core.ingestion.indexer import FAISSIndexer
from app.core.ingestion.embedder import Embedder
from app.core.retrieval.bm25_retriever import BM25Retriever

logger = get_logger(__name__)


# ── SessionState ───────────────────────────────────────────────────────────────

class SessionState:
    """
    In-memory representation of a single chat session.

    Holds all state for one isolated RAG environment:
    - Core identity  : id, name, description, created_at
    - Document data  : document_names, chunks
    - ML indexes     : faiss_index, bm25_retriever
    - Conversation   : messages
    - R2 keys        : computed properties used as object keys in R2
    """

    def __init__(
        self,
        id: str,
        name: str,
        description: Optional[str],
        created_at: datetime,
    ):
        # Core identity — set at creation, never changed
        self.id = id
        self.name = name
        self.description = description
        self.created_at = created_at

        # Document data — populated during ingestion
        self.document_names: list[str] = []
        self.chunks: list[dict] = []

        # ML indexes — not JSON-serializable, rebuilt from R2 on startup
        self.faiss_index: Any = None
        self.bm25_retriever: Any = None

        # Conversation history
        self.messages: list[dict] = []

    # ── R2 key properties ──────────────────────────────────────────────────────
    # These replace the old disk path properties.
    # The key structure mirrors the old folder structure for consistency.
    # "sessions/abc-123/metadata.json" in R2 == "./data/sessions/abc-123/metadata.json" on disk

    @property
    def r2_prefix(self) -> str:
        """Root prefix for all R2 objects belonging to this session."""
        return f"sessions/{self.id}"

    @property
    def r2_metadata_key(self) -> str:
        return f"{self.r2_prefix}/metadata.json"

    @property
    def r2_chunks_key(self) -> str:
        return f"{self.r2_prefix}/chunks.json"

    @property
    def r2_index_key(self) -> str:
        return f"{self.r2_prefix}/faiss.index"

    def __repr__(self) -> str:
        return (
            f"SessionState(id={self.id!r}, name={self.name!r}, "
            f"docs={len(self.document_names)}, chunks={len(self.chunks)})"
        )


# ── SessionManager ─────────────────────────────────────────────────────────────

class SessionManager:
    """
    Registry for all active sessions.

    Responsibilities:
    - Create sessions and persist metadata to R2
    - Load persisted sessions from R2 on startup
    - Retrieve, list, and delete sessions
    - Enforce session isolation (each session has its own indexes)

    Used as a singleton via get_session_manager() in dependencies.py.
    """

    def __init__(self):
        # In-memory store: session_id → SessionState
        self._sessions: dict[str, SessionState] = {}

        # R2 client — single instance shared across all operations
        self._r2 = R2Storage()

        # Reload all sessions persisted from previous backend runs
        self._load_persisted_sessions()

    # ── Private helpers ────────────────────────────────────────────────────────

    def _save_metadata(self, session: SessionState) -> None:
        """
        Persist session metadata to R2 as JSON.

        Saves: id, name, description, created_at, document_names, messages.
        Does NOT save faiss_index or bm25_retriever — they are binary ML
        objects. FAISS is saved separately as bytes; BM25 is rebuilt from chunks.

        Args:
            session: The SessionState object to persist.
        """
        data_dict = {
            "id": session.id,
            "name": session.name,
            "description": session.description,
            "created_at": session.created_at.isoformat(),
            "document_names": session.document_names,
            "messages": session.messages,
        }
        self._r2.put_json(session.r2_metadata_key, data_dict)

    def _save_faiss_index(self, session: SessionState) -> None:
        """
        Serialize FAISS index to bytes and upload to R2.

        Why io.BytesIO?
            FAISS's save API writes to a file path on disk.
            We can't give it a file path for R2 — we need raw bytes.
            BytesIO acts as an in-memory file: FAISS writes into it,
            we read the bytes out and send them to R2.

        Args:
            session: SessionState containing a non-None faiss_index.
        """
        buf = io.BytesIO()
        faiss.write_index(
            session.faiss_index.index,
            faiss.PyCallbackIOWriter(buf.write),
            )
        buf.seek(0)
        self._r2.put(session.r2_index_key, buf.read())

    def _load_faiss_index(self, session: SessionState, dimension: int) -> Optional[FAISSIndexer]:
        """
        Download FAISS bytes from R2 and deserialize into a FAISSIndexer.

        Args:
            session:   SessionState to load the index for.
            dimension: Embedding dimension (must match what was used at index time).

        Returns:
            Populated FAISSIndexer, or None if no index exists in R2 yet.
        """
        raw = self._r2.get(session.r2_index_key)
        if raw is None:
            return None

        buf = io.BytesIO(raw)
        raw_index = faiss.read_index(
            faiss.PyCallbackIOReader(buf.read)   # read from the BytesIO buffer
        )

        indexer = FAISSIndexer(dimension)
        indexer.index = raw_index
        return indexer

    def _load_persisted_sessions(self) -> None:
        """
        Reload all sessions persisted to R2 on startup.

        For each session ID found in R2:
          1. Load metadata.json → rebuild SessionState
          2. Load chunks.json   → restore session.chunks
          3. Load faiss.index   → rebuild FAISSIndexer from bytes
          4. Rebuild BM25 from chunks (BM25 is not stored — it's fast to rebuild)
          5. Add to self._sessions

        Any session with a missing or corrupt metadata.json is skipped with
        a warning rather than crashing the entire startup.
        """
        session_ids = self._r2.list_session_ids()

        loaded = 0
        for session_id in session_ids:
            try:
                # 1. Load metadata
                meta = self._r2.get_json(f"sessions/{session_id}/metadata.json")
                if meta is None:
                    logger.warning(f"Session {session_id[:8]}... has no metadata, skipping")
                    continue

                session = SessionState(
                    id=meta["id"],
                    name=meta["name"],
                    description=meta["description"],
                    created_at=datetime.fromisoformat(meta["created_at"]),
                )
                session.document_names = meta.get("document_names", [])
                session.messages = meta.get("messages", [])

                # 2. Load chunks
                chunks = self._r2.get_json(session.r2_chunks_key)
                if chunks:
                    session.chunks = chunks

                    # 3. Load FAISS index from bytes
                    session.faiss_index = self._load_faiss_index(
                        session, _config().DIM_FAISS
                    )

                    # 4. Rebuild BM25 (fast, no storage needed)
                    session.bm25_retriever = BM25Retriever()
                    session.bm25_retriever.build(session.chunks)

                self._sessions[session.id] = session
                loaded += 1

            except Exception as e:
                logger.error(f"Failed to reload session {session_id[:8]}...: {e}")
                continue

        logger.info(f"Reloaded {loaded} session(s) from R2")

    # ── Public CRUD methods ────────────────────────────────────────────────────

    def create_session(self, name: str, description: Optional[str]) -> SessionResponse:
        """
        Create a new session, persist it to R2, and return its response model.

        Args:
            name: Display name for the session.
            description: Optional description.

        Returns:
            SessionResponse with all session metadata.
        """
        session_id = str(uuid4())

        session = SessionState(
            id=session_id,
            name=name,
            description=description,
            created_at=datetime.now(timezone.utc),
        )

        self._sessions[session_id] = session
        self._save_metadata(session)

        logger.info(f"Created session {session_id[:8]}... | name='{name}'")

        return SessionResponse(
            id=session.id,
            name=session.name,
            description=session.description,
            created_at=session.created_at,
            document_count=len(session.document_names),
            message_count=len(session.messages),
        )

    def get_session(self, session_id: str) -> SessionState:
        """
        Retrieve a session by ID.

        Raises:
            HTTPException 404: If session_id is not found in memory.
        """
        if session_id not in self._sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        return self._sessions[session_id]

    def list_sessions(self) -> list[SessionResponse]:
        """Return a SessionResponse for every active session."""
        return [
            SessionResponse(
                id=s.id,
                name=s.name,
                description=s.description,
                created_at=s.created_at,
                document_count=len(s.document_names),
                message_count=len(s.messages),
            )
            for s in self._sessions.values()
        ]

    def delete_session(self, session_id: str) -> None:
        """
        Delete a session from memory and remove all its R2 objects.

        Order: remove from R2 first, then from memory.
        If R2 deletion fails, the session remains in memory (no orphaned state).

        Raises:
            HTTPException 404: If session_id is not found.
        """
        session = self.get_session(session_id)
        self._r2.delete_prefix(f"{session.r2_prefix}/")
        del self._sessions[session_id]
        logger.info(f"Deleted session {session_id[:8]}...")

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """
        Append a message to conversation history and persist to R2.

        Args:
            session_id: UUID of the session.
            role:       "user" or "assistant".
            content:    The message text.

        Raises:
            HTTPException 404: If session_id is not found.
        """
        session = self.get_session(session_id)

        session.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        self._save_metadata(session)
        logger.info(f"Message saved | session={session_id[:8]}... | role={role}")

    # ── Ingestion pipeline ─────────────────────────────────────────────────────

    def ingest_document(self, session_id: str, file_bytes: bytes, filename: str) -> int:
        """
        Run the full ingestion pipeline for one document.

        Steps:
            1. Parse document → raw text
            2. Chunk text → list of chunk dicts
            3. Embed chunks → float32 vectors
            4. Add vectors to FAISS index (create if first document)
            5. Extend session.chunks
            6. Rebuild BM25 over all session chunks
            7. Append filename to session.document_names
            8. Persist: FAISS bytes → R2, chunks JSON → R2, metadata → R2

        Args:
            session_id: UUID of the target session.
            file_bytes: Raw file content as bytes.
            filename:   Original filename (used for chunk attribution).

        Returns:
            Number of chunks created from this document.

        Raises:
            HTTPException 400: If the file type is unsupported.
            HTTPException 404: If the session is not found.
        """
        session = self.get_session(session_id)

        # 1. Parse
        try:
            doc = DocumentParser().parse(file_bytes, filename)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # 2. Chunk
        chunks = RecursiveChunker(
            chunk_size=_config().CHUNK_SIZE,
            chunk_overlap=_config().CHUNK_OVERLAP,
        ).split(doc, filename)

        # 3. Embed
        vectors = Embedder().embed_chunks(chunks=chunks)

        # 4. Add to FAISS (create index on first document)
        if session.faiss_index is None:
            session.faiss_index = FAISSIndexer(vectors.shape[1])
        session.faiss_index.add(vectors)

        # 5. Extend chunk list
        session.chunks.extend(chunks)

        # 6. Rebuild BM25 over all chunks
        session.bm25_retriever = BM25Retriever()
        session.bm25_retriever.build(session.chunks)

        # 7. Record filename
        session.document_names.append(filename)

        # 8. Persist everything to R2
        self._save_faiss_index(session)
        self._r2.put_json(session.r2_chunks_key, session.chunks)
        self._save_metadata(session)

        logger.info(
            f"Ingested '{filename}' | session={session_id[:8]}... "
            f"| chunks={len(chunks)} | total={len(session.chunks)}"
        )
        return len(chunks)