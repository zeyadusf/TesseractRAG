"""
Chat router — RAG query / answer flow within a session.

Routes
------
POST   /api/v1/sessions/{session_id}/chat            → ask a question, get an answer
GET    /api/v1/sessions/{session_id}/chat            → list conversation history
GET    /api/v1/sessions/{session_id}/chat/{turn_id}  → get a single turn
DELETE /api/v1/sessions/{session_id}/chat            → clear chat history
"""

from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, Query, status, Request
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.dependencies import (
    get_db,
    get_async_session,
    get_generation_pipeline,
    get_embedder,
    get_retrieval_pipeline,
    get_evaluator,
)
from backend.core.logger import get_logger
logger = get_logger(__name__)
from backend.core.limiter import limiter
from backend.core.security.jwt_deps import get_current_active_user
from backend.models.auth import UserOut
from backend.models.chat import ChatHistoryOut, ChatTurnOut
from backend.rag.pipelines.generation_pipeline import ChatRequest, ChatResponse
from backend.services.chat_service import ChatService
from backend.services.evaluation_service import EvaluationService
from backend.storage.db.db_dispatcher import DBDispatcher

router = APIRouter()


# ── Dependencies ───────────────────────────────────────────────────────────────

def get_chat_service(
    db: DBDispatcher = Depends(get_db),
    session: AsyncSession = Depends(get_async_session),
) -> ChatService:
    return ChatService(
        db=db,
        session=session,
        embedder=get_embedder(),
        generation_pipeline=get_generation_pipeline(),
        retrieval_pipeline=get_retrieval_pipeline(),
    )


def get_evaluation_service(
    db: DBDispatcher = Depends(get_db),
) -> EvaluationService:
    return EvaluationService(db, evaluator=get_evaluator())


# ── Chat ───────────────────────────────────────────────────────────────────────

@router.post(
    "/{session_id}/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ask a question; get a RAG-grounded answer",
)
@limiter.limit("10/minute")
async def ask(
    request: Request,
    session_id: UUID,
    payload: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: UserOut = Depends(get_current_active_user),
    chat_service: ChatService = Depends(get_chat_service),
    eval_service: EvaluationService = Depends(get_evaluation_service),
) -> ChatResponse:
    response = await chat_service.chat(
        session_id=session_id,
        request=payload,
    )
    background_tasks.add_task(
        eval_service.evaluate_message_background,
        session_id=session_id,
        message_id=response.message_id,
        question=payload.question,
        answer=response.answer,
        user_id=current_user.id,
    )
    logger.info("[Evaluate] run in background tasks ")

    return response


# ── History ────────────────────────────────────────────────────────────────────

@router.get(
    "/{session_id}/chat",
    response_model=ChatHistoryOut,
    summary="List all question/answer turns for a session",
)
@limiter.limit("60/minute")
async def get_history(
    request: Request,
    session_id: UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: UserOut = Depends(get_current_active_user),
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatHistoryOut:
    return await chat_service.list_history(
        session_id=session_id,
        user_id=current_user.id,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/{session_id}/chat/{turn_id}",
    response_model=ChatTurnOut,
    summary="Get a single chat turn by ID",
)
@limiter.limit("60/minute")
async def get_turn(
    request: Request,
    session_id: UUID,
    turn_id: UUID,
    current_user: UserOut = Depends(get_current_active_user),
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatTurnOut:
    return await chat_service.get_turn(
        session_id=session_id,
        turn_id=turn_id,
        user_id=current_user.id,
    )


@router.delete(
    "/{session_id}/chat",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Clear all chat history for a session",
)
@limiter.limit("5/minute")
async def clear_history(
    request: Request,
    session_id: UUID,
    current_user: UserOut = Depends(get_current_active_user),
    chat_service: ChatService = Depends(get_chat_service),
) -> None:
    await chat_service.clear_history(
        session_id=session_id,
        user_id=current_user.id,
    )