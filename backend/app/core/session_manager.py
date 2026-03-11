from typing  import Optional,Any
from pathlib import Path
from datetime import datetime, timezone
from app.models.session import SessionResponse
from app.config import get_settings
from uuid import uuid4
import json
import shutil
from fastapi import HTTPException
from app.utils.logger import get_logger

logger = get_logger(__name__)
class SessionState:
  def __init__(
    self,
    id: str,
    name: str,
    description: Optional[str],
    created_at: datetime,
    data_dir :str
  ):
    # Core identity — set at creation, never change
    self.id = id
    self.name = name
    self.description = description
    self.created_at = created_at
  
    # Document data — populated in Phase 3 :Document Ingestion Pipeline
    self.document_names: list[str] = list()
    self.chunks: list[dict] = []
  
    # ML indexes — populated in Phase 3 : Document Ingestion Pipeline
    self.faiss_index: Any = None
    self.bm25_retriever: Any = None
  
    # Conversation — populated in Phase 5 : Generation Layer & Chat Endpoint
    self.messages: list[dict] = list()
  
    # Data Path 
    self.data_dir = data_dir
  
  @property
  def session_dir(self) -> Path: return Path(self.data_dir) / "sessions" / self.id
  
  @property
  def metadata_path(self) -> Path: return self.session_dir / "metadata.json"
  
  @property
  def chunks_path(self) -> Path: return self.session_dir / "chunks.json"
  
  @property
  def index_path(self) -> Path: return self.session_dir / "faiss.index"


class SessionManager:
  def __init__(self):
    self._sessions : dict = dict()
    self.data_dir : str = get_settings().DATA_DIR
  
  def create_session(self, name: str, description: Optional[str]) -> SessionResponse:
    # Step 1 — generate ID
    session_id = str(uuid4())

    # Step 2 — create the SessionState object
    session = SessionState(
        id=session_id,
        name=name,
        description=description,
        created_at=datetime.now(timezone.utc),
        data_dir=self.data_dir
    )

    # Step 3 — store in dict
    self._sessions[session_id] = session

    # Step 4 — create directory on disk
    session.session_dir.mkdir(parents=True, exist_ok=True)

    # Step 5 — save metadata to disk
    self._save_metadata(session)

    # Step 6 — return SessionResponse
    return SessionResponse(
        id=session.id,
        name=session.name,
        description=session.description,
        created_at=session.created_at,
        document_count=len(session.document_names),
        message_count=len(session.messages)
    )

  def _save_metadata(self, session : SessionState)-> None:
    
    data_dict = {
      'id': session.id,
      'name' : session.name,
      'description' : session.description,
      'created_at' : session.created_at.isoformat(),
      'document_names' : session.document_names,
      'messages' : session.messages
    }
    with open(session.metadata_path, "w") as f:
      json.dump(data_dict, f, indent=2)
  
  def get_session(self, session_id: str) -> SessionState:
    if session_id not in self._sessions:
      # HTTPException raised here so all callers (endpoints, other methods)
      # get consistent 404 behavior without duplicating the check everywhere
      raise HTTPException(status_code=404,detail="Session not found")
    return self._sessions[session_id]

  def list_sessions(self) -> list[SessionResponse]:
    return [
    SessionResponse(
        id=session.id,
        name=session.name,
        description=session.description,
        created_at=session.created_at,
        document_count=len(session.document_names),
        message_count=len(session.messages)
    )
    for session in self._sessions.values()
]

  def delete_session(self, session_id: str) -> None:
      session = self.get_session(session_id)
      shutil.rmtree(session.session_dir)
      del self._sessions[session_id]
      logger.info(f"{session_id[:8]}... Session deleted successfully")



if __name__ == '__main__':
    from datetime import timezone
    
    manager_1 = SessionManager()

    response = manager_1.create_session("Test Session", "My first session")
    
    print(response.id)
    print(response.name)
    print(response.document_count)
    
    # Check the folder and file were created
    from pathlib import Path
    session_dir = Path("./data/sessions") / response.id
    print(session_dir.exists())           # True
    print((session_dir / "metadata.json").exists())  # True
    
    # Print what was saved to disk
    import json
    with open(session_dir / "metadata.json") as f:
        print(json.load(f))

    print('*-*='*10)
    

    # Create two sessions
    s1 = manager_1.create_session("Research Papers", "ML papers")
    s2 = manager_1.create_session("Legal Docs", None)

    # List both
    print("--- After creating 2 sessions ---")
    for s in manager_1.list_sessions():
        print(f"  {s.id[:8]}... | {s.name}")

    # Get one by ID
    fetched = manager_1.get_session(s1.id)
    print(f"\nFetched: {fetched.name}")

    # Delete one
    manager_1.delete_session(s1.id)
    print(f"\n--- After deleting '{s1.name}' ---")
    for s in manager_1.list_sessions():
        print(f"  {s.id[:8]}... | {s.name}")

    # Verify folder is gone
    from pathlib import Path
    print(f"\nFolder exists: {(Path('./data/sessions') / response.id).exists()}")

    for s in manager_1.list_sessions():
      manager_1.delete_session(s.id)

        # Verify folder is gone
    print(f"\nFolder exists: {(Path('./data/sessions') / response.id).exists()}")

