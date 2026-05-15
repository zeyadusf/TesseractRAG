"""
Documents router — document ingestion and management within a session.

Routes
------
POST   /api/v1/sessions/{session_id}/documents            → upload & ingest document
GET    /api/v1/sessions/{session_id}/documents            → list documents in session
GET    /api/v1/sessions/{session_id}/documents/{doc_id}   → get document metadata
DELETE /api/v1/sessions/{session_id}/documents/{doc_id}   → delete document + chunks
"""

# from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, File, Query, UploadFile, status,Request

from backend.core.dependencies import get_db
from backend.core.security.jwt_deps import get_current_active_user
from backend.models.auth import UserOut
from backend.core.limiter import limiter  
from backend.models.documents import  DocumentOut, DeleteResponse
from backend.services.document_service import DocumentService
from backend.storage.db.db_dispatcher import DBDispatcher
from typing import List
router = APIRouter()


# ── Dependency 

def get_document_service(
    db: DBDispatcher = Depends(get_db),
) -> DocumentService:
    return DocumentService(db)         


# ── Upload / ingest 

@router.post("/{session_id}/documents",
    response_model=DocumentOut,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a document into the session",)
@limiter.limit("20/hour")
async def upload_document(
    session_id: UUID,
    request: Request,
    file: UploadFile = File(..., description="PDF, TXT, MD, DOCX, …"),
    current_user: UserOut = Depends(get_current_active_user),
    doc_service: DocumentService = Depends(get_document_service),
) -> DocumentOut:
    raw_bytes = await file.read()
    return await doc_service.upload_and_index(   
        session_id=session_id,
        filename=file.filename,
        file_bytes=raw_bytes,                  
    )


# ── List 

@router.get("/{session_id}/documents",
    response_model=List[DocumentOut],
    summary="List documents in a session",)
@limiter.limit("10/minute")
async def list_documents(
    session_id: UUID,
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: UserOut = Depends(get_current_active_user),
    doc_service: DocumentService = Depends(get_document_service),
) -> List[DocumentOut]:
    return await doc_service.list_documents(    
        session_id=session_id,
    )


# ── Get 

@router.get("/{session_id}/documents/{doc_id}",
    response_model=DocumentOut,
    summary="Get document metadata and ingestion status",)
@limiter.limit("10/minute")
async def get_document(
    request: Request,
    session_id: UUID,
    doc_id: UUID,
    current_user: UserOut = Depends(get_current_active_user),
    doc_service: DocumentService = Depends(get_document_service),
) -> DocumentOut:
    return await doc_service.get_document(      
        session_id=session_id,
        document_id=doc_id,                     
    )


# ── Delete 


@router.delete("/{session_id}/documents/{doc_id}",
    response_model=DeleteResponse,  
    status_code=status.HTTP_200_OK, 
    summary="Delete a document and all its vector chunks",)
@limiter.limit("10/minute")
async def delete_document(
    request: Request,               
    session_id: UUID,
    doc_id: UUID,
    current_user: UserOut = Depends(get_current_active_user),
    doc_service: DocumentService = Depends(get_document_service),
) -> DeleteResponse:                
    await doc_service.delete_document(
        document_id=doc_id,
        session_id=session_id,
    )
    return DeleteResponse(          
        message="Document deleted successfully",
        deleted_id=doc_id
    )