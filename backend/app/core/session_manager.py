from typing  import Optional,Any
from pathlib import Path
from datetime import datetime, timezone
from app.models.session import SessionResponse
from app.config import get_settings
from uuid import uuid4
import json
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
    





if __name__ == '__main__':
    from datetime import timezone
    
    manager = SessionManager()
    response = manager.create_session("Test Session", "My first session")
    
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