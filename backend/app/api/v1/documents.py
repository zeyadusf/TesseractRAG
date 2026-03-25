"""
documents.py
------------
Document upload and listing endpoints for TesseractRAG API v1.

All endpoints require the X-Owner-ID header (enforced via get_owner_id
dependency). Document operations are scoped to the owning session —
a request with the wrong owner_id will receive 403 before any ingestion
work is done.
"""

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from pathlib import Path
from app.dependencies import get_session_manager, get_owner_id
from app.core.session_manager import SessionManager
from app.models.document import DocumentInfo

router = APIRouter()


@router.post("/{session_id}/documents", status_code=201)
async def upload_document(
    session_id: str,
    file: UploadFile = File(...),
    manager: SessionManager = Depends(get_session_manager),
    owner_id: str = Depends(get_owner_id),              # ← NEW
):
    """
    Upload a document to the session's knowledge base.

    Ownership is checked before any file processing occurs. If the
    requesting browser does not own this session, the request is rejected
    with 403 before any compute is spent on ingestion.

    Supported formats: PDF, TXT, MD
    Max file size: 10 MB
    """
    # 1. Validate file type — allowed: pdf, txt, md
    allowed = [".pdf", ".md", ".txt"]
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Allowed: pdf, md, txt",
        )

    # 2. Validate file size — max 10 MB
    file_bytes = await file.read()
    if len(file_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="File exceeds the 10 MB limit.",
        )

    # 3. Ingest — ownership enforced inside manager.ingest_document
    result = manager.ingest_document(
        session_id,
        file_bytes,
        file.filename,
        owner_id=owner_id,                              # ← NEW
    )

    return {"filename": file.filename, "chunks_created": result}


@router.get("/{session_id}/documents")
async def list_documents(
    session_id: str,
    manager: SessionManager = Depends(get_session_manager),
    owner_id: str = Depends(get_owner_id),              # ← NEW
):
    """
    List all documents indexed in a session.

    Returns each document's filename and its chunk count.
    Ownership is checked via get_session — returns 403 if the requesting
    browser does not own this session.
    """
    # get_session with owner_id enforces ownership before returning
    session = manager.get_session(session_id, owner_id=owner_id)   # ← NEW

    return [
        {
            "filename":     name,
            "chunks_count": len([c for c in session.chunks if c["document_name"] == name]),
        }
        for name in session.document_names
    ]
