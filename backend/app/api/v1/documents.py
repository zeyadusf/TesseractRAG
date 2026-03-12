from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from pathlib import Path
from app.dependencies import get_session_manager
from app.core.session_manager import SessionManager
from app.models.document import DocumentInfo

router = APIRouter()

@router.post("/{session_id}/documents", status_code=201)
async def upload_document(
    session_id: str,
    file: UploadFile = File(...),
    manager: SessionManager = Depends(get_session_manager)
):
    # 1. Validate file type — allowed: pdf, txt, md
    allowed = [".pdf", ".md", ".txt"]
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed: raise HTTPException(detail="Unsupported file type, Only[pdf, md, txt]",status_code=400)
    # 2. Validate file size — max 10MB
    file_bytes = await file.read()
    #    len(file_bytes) > 10 * 1024 * 1024 → raise HTTPException 413
    if len(file_bytes) > 10 * 1024 * 1024 : raise HTTPException(detail="File is greater than 10MB",status_code=413)
    # 3. Call manager.ingest_document(session_id, file_bytes, file.filename)
    result = manager.ingest_document(session_id,file_bytes,file.filename)
    return {"filename": file.filename, "chunks_created": result}

@router.get("/{session_id}/documents")
async def list_documents(
    session_id: str,
    manager: SessionManager = Depends(get_session_manager)
):
    # get_session → return document_names as a list
    session = manager.get_session(session_id)
    result = []
    for name in session.document_names:
        result.append({
            'filename':name,
            'chunks count':len([c for c in session.chunks if c['document_name'] == name])
        })
    return result