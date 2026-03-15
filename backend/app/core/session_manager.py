"""
session_manager.py
------------------
Core session management for TesseractRAG.

Contains:
    - SessionState  : In-memory representation of a single RAG session
    - SessionManager: Registry that creates, retrieves, lists, and deletes sessions
"""

from typing import Optional, Any
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
import json
import shutil

from fastapi import HTTPException

from app.models.session import SessionResponse
from app.config import get_settings as _config
from app.utils.logger import get_logger

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
    - Document data  : document_names, chunks         (populated in Phase 3)
    - ML indexes     : faiss_index, bm25_retriever    (populated in Phase 3)
    - Conversation   : messages                       (populated in Phase 5)
    - Disk paths     : computed properties via data_dir + id
    """

    def __init__(
        self,
        id: str,
        name: str,
        description: Optional[str],
        created_at: datetime,
        data_dir: str,
    ):
        # Core identity — set at creation, never changed
        self.id = id
        self.name = name
        self.description = description
        self.created_at = created_at

        # Document data — populated in Phase 3: Document Ingestion Pipeline
        self.document_names: list[str] = list()
        self.chunks: list[dict] = list()

        # ML indexes — populated in Phase 3: Document Ingestion Pipeline
        # Not JSON-serializable — restored by reloading chunks in Phase 3
        self.faiss_index: Any = None
        self.bm25_retriever: Any = None

        # Conversation history — populated in Phase 5: Generation Layer
        self.messages: list[dict] = list()

        # Root data directory — used to compute all disk paths below
        self.data_dir = data_dir

    # ── Disk path properties ───────────────────────────────────────────────────
    # Computed dynamically so they always reflect the current data_dir and id.
    # Never stored as instance variables — derived values waste memory.

    @property
    def session_dir(self) -> Path:
        """Root directory for this session: {data_dir}/sessions/{id}"""
        return Path(self.data_dir) / "sessions" / self.id

    @property
    def metadata_path(self) -> Path:
        """Path to session metadata JSON file."""
        return self.session_dir / "metadata.json"

    @property
    def chunks_path(self) -> Path:
        """Path to chunks JSON file (written in Phase 3)."""
        return self.session_dir / "chunks.json"

    @property
    def index_path(self) -> Path:
        """Path to FAISS binary index file (written in Phase 3)."""
        return self.session_dir / "faiss.index"

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
    - Create sessions and persist metadata to disk
    - Load persisted sessions from disk on startup
    - Retrieve, list, and delete sessions
    - Enforce session isolation (each session has its own indexes)

    Used as a singleton via get_session_manager() in dependencies.py.
    """

    def __init__(self):
        # In-memory store: session_id (str) → SessionState
        # Dict chosen over list for O(1) lookup by ID
        self._sessions: dict[str, SessionState] = {}
        self.data_dir: str = _config().DATA_DIR

        # Reload any sessions persisted from previous backend runs
        self._load_persisted_sessions()

    # ── Private helpers ────────────────────────────────────────────────────────

    def _save_metadata(self, session: SessionState) -> None:
        """
        Persist session metadata to disk as JSON.

        Saves: id, name, description, created_at, document_names, messages.
        Does NOT save faiss_index or bm25_retriever — they are binary ML
        objects that cannot be JSON-serialized. They are rebuilt in Phase 3.

        Args:
            session: The SessionState object to persist.
        """
        data_dict = {
            "id": session.id,
            "name": session.name,
            "description": session.description,
            "created_at": session.created_at.isoformat(),   # datetime → string
            "document_names": session.document_names,
            "messages": session.messages,
        }
        with open(session.metadata_path, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, indent=2)

    def _load_persisted_sessions(self) -> None:
        """
        Reload all sessions persisted to disk on startup.

        Iterates over subdirectories in {data_dir}/sessions/.
        For each directory that contains a valid metadata.json,
        reconstructs a SessionState, restores FAISS index, chunks,
        and BM25 retriever, then adds it to self._sessions.
        """
        root_path = Path(self.data_dir) / "sessions"

        if not root_path.exists():
            return

        for session_path in root_path.iterdir():
            if not session_path.is_dir():
                continue

            file_path = session_path / "metadata.json"

            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                session = SessionState(
                    id=data["id"],
                    name=data["name"],
                    description=data["description"],
                    created_at=datetime.fromisoformat(data["created_at"]),
                    data_dir=self.data_dir,
                )

                session.document_names = data["document_names"]
                session.messages = data["messages"]

                if session.index_path.exists():
                    session.faiss_index = FAISSIndexer(_config().DIM_FAISS)
                    session.faiss_index.load(session.index_path)

                if session.chunks_path.exists():
                    with open(session.chunks_path, 'r') as f:
                        session.chunks = json.load(f)

                if session.chunks:
                    session.bm25_retriever = BM25Retriever()
                    session.bm25_retriever.build(session.chunks)

                self._sessions[session.id] = session

        logger.info(f"Reloaded {len(self._sessions)} session(s) from disk")

        # ── Public CRUD methods ────────────────────────────────────────────────────

    def create_session(self, name: str, description: Optional[str]) -> SessionResponse:
        """
        Create a new session, persist it to disk, and return its response model.

        Steps:
            1. Generate a unique UUID
            2. Instantiate SessionState
            3. Store in _sessions dict
            4. Create session directory on disk
            5. Save metadata.json
            6. Return SessionResponse

        Args:
            name: Display name for the session (3-50 chars).
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
            data_dir=self.data_dir,
        )

        self._sessions[session_id] = session
        session.session_dir.mkdir(parents=True, exist_ok=True)
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

        HTTPException is raised here (not in the endpoint) so all callers
        get consistent 404 behavior without duplicating the check everywhere.

        Args:
            session_id: UUID string of the session to retrieve.

        Returns:
            SessionState object for the requested session.

        Raises:
            HTTPException 404: If session_id is not found.
        """
        if session_id not in self._sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        return self._sessions[session_id]

    def list_sessions(self) -> list[SessionResponse]:
        """
        Return a SessionResponse for every active session.

        Returns:
            List of SessionResponse objects, one per session.
        """
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
        Delete a session from memory and remove its directory from disk.

        Order matters: get the object first (to access session_dir),
        then delete disk, then delete from memory. If rmtree fails,
        the session remains in memory — no orphaned state.

        Args:
            session_id: UUID string of the session to delete.

        Raises:
            HTTPException 404: If session_id is not found.
        """
        session = self.get_session(session_id)       # raises 404 if not found
        shutil.rmtree(session.session_dir)            # delete from disk first
        del self._sessions[session_id]                # then remove from memory
        logger.info(f"Deleted session {session_id[:8]}...")

    # ــــــــ ingestion pipeline ─────────────────────────────────────────────

    def ingest_document(self, session_id: str, file_bytes: bytes, filename: str) -> int:
    # 1. get_session(session_id)           — raises 404 if not found
        session = self.get_session(session_id)
    # 2. DocumentParser().parse()          — extract text
        try:
            doc = DocumentParser().parse(file_bytes,filename)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    # 3. RecursiveChunker().split()        — split into chunk dicts
        chunks = RecursiveChunker(chunk_size=_config().CHUNK_SIZE,
                                    chunk_overlap=_config().CHUNK_OVERLAP).split(doc,filename)
    # 4. Embedder().embed_chunks()         — get vectors
        vector_doc = Embedder().embed_chunks(chunks=chunks)
    # 5. If session.faiss_index is None:
        if session.faiss_index is None:
        # create FAISSIndexer(dimension)
            session.faiss_index= FAISSIndexer(vector_doc.shape[1])
    # 6. session.faiss_index.add(vectors)  — add to FAISS
        session.faiss_index.add(vector_doc)
    # 7. session.chunks.extend(new_chunks) — add to master chunk list
        session.chunks.extend(chunks)
    # 8. session.bm25_retriever = BM25Retriever()
        session.bm25_retriever = BM25Retriever()
    #    session.bm25_retriever.build(session.chunks) — rebuild BM25
        session.bm25_retriever.build(session.chunks)
    # 9. session.document_names.append(filename)
        session.document_names.append(filename)
    # 10. persist: save faiss index + save chunks json
        session.faiss_index.save(session.index_path)
        self._save_metadata(session)
        with open(session.chunks_path, 'w') as f:
            json.dump(session.chunks, f)
    # 11. return len(new_chunks)
        return len(chunks)

# ── Manual test blocks ─────────────────────────────────────────────────────────
# Run with: python -m app.core.session_manager (from backend/ directory)

if __name__ == "__main__":

    # ── P2-02: Disk path properties ───────────────────────────────────────────
    print("\n=== P2-02: SessionState path properties ===")
    s = SessionState(
        id="abc-123",
        name="Test",
        description=None,
        created_at=datetime.now(timezone.utc),
        data_dir="./data",
    )
    print(s.session_dir)     # data\sessions\abc-123
    print(s.metadata_path)   # data\sessions\abc-123\metadata.json
    print(s.chunks_path)     # data\sessions\abc-123\chunks.json
    print(s.index_path)      # data\sessions\abc-123\faiss.index
    print(s.id)              # abc-123
    print(s.chunks)          # []
    print(s.faiss_index)     # None

    # ── P2-03: create_session + _save_metadata ────────────────────────────────
    print("\n=== P2-03: create_session ===")
    manager = SessionManager()
    response = manager.create_session("Test Session", "My first session")
    print(response.id)
    print(response.name)
    print(response.document_count)
    session_dir = Path("./data/sessions") / response.id
    print(f"Dir exists:  {session_dir.exists()}")
    print(f"JSON exists: {(session_dir / 'metadata.json').exists()}")
    with open(session_dir / "metadata.json") as f:
        print(json.load(f))

    # ── P2-04: get_session, list_sessions, delete_session ─────────────────────
    print("\n=== P2-04: CRUD operations ===")
    s1 = manager.create_session("Research Papers", "ML papers")
    s2 = manager.create_session("Legal Docs", None)
    print("After creating 2 sessions:")
    for s in manager.list_sessions():
        print(f"  {s.id[:8]}... | {s.name}")
    fetched = manager.get_session(s1.id)
    print(f"Fetched: {fetched.name}")
    manager.delete_session(s1.id)
    print(f"After deleting '{s1.name}':")
    for s in manager.list_sessions():
        print(f"  {s.id[:8]}... | {s.name}")
    print(f"Folder exists: {(Path('./data/sessions') / s1.id).exists()}")
    try:
        manager.get_session(s1.id)
    except HTTPException as e:
        print(f"Expected 404: {e.detail}")

    # ── P2-05: _load_persisted_sessions ───────────────────────────────────────
    print("\n=== P2-05: persistence across restarts ===")
    manager_a = SessionManager()
    new_session = manager_a.create_session("Persisted Session", "testing reload")
    print(f"Created: {new_session.name} | {new_session.id[:8]}...")
    manager_b = SessionManager()
    reloaded = manager_b.list_sessions()
    print(f"Reloaded {len(reloaded)} session(s)")
    for s in reloaded:
        print(f"  {s.name} | {s.id[:8]}...")

    # cleanup — remove all test sessions so next run starts clean
    for s in manager_b.list_sessions():
        manager_b.delete_session(s.id)
    print("Cleaned up all test sessions")