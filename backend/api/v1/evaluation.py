"""
Evaluation router — answer quality scoring for RAG turns.

Routes
------
GET    /api/v1/sessions/{session_id}/evaluation/summary     → session averages
GET    /api/v1/sessions/{session_id}/evaluation             → list evaluation results
POST   /api/v1/sessions/{session_id}/evaluation             → evaluate all turns in session
GET    /api/v1/sessions/{session_id}/evaluation/{turn_id}   → get result for one turn

Note: Single-turn evaluation is triggered automatically as a background task
      from the chat endpoint after each assistant response.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, Query, status, Request

from backend.core.dependencies import get_db, get_evaluator
from backend.core.security.jwt_deps import get_current_active_user,get_current_superuser
from backend.models.auth import UserOut
from backend.core.limiter import limiter
from backend.models.evaluation import (
    EvaluateSessionRequest,
    EvaluationOut,
    SessionEvaluationSummary,
)
from backend.services.evaluation_service import EvaluationService
from backend.storage.db.db_dispatcher import DBDispatcher

router = APIRouter()


# ── Dependency ─────────────────────────────────────────────────────────────────

def get_evaluation_service(
    db: DBDispatcher = Depends(get_db),
) -> EvaluationService:
    return EvaluationService(db, evaluator=get_evaluator())


# ── Session summary (static path — must come before {turn_id}) ────────────────

@router.get(
    "/{session_id}/evaluation/summary",
    response_model=SessionEvaluationSummary,
    summary="Get average evaluation scores for a session",
)
@limiter.limit("10/minute")
async def get_session_summary(
    request: Request,
    session_id: UUID,
    current_user: UserOut = Depends(get_current_active_user),
    eval_service: EvaluationService = Depends(get_evaluation_service),
) -> SessionEvaluationSummary:
    return await eval_service.get_session_summary(
        session_id=session_id,
        user_id=current_user.id,
    )


# ── List all evaluations ───────────────────────────────────────────────────────

@router.get(
    "/{session_id}/evaluation",
    response_model=list[EvaluationOut],
    summary="List all evaluation results for a session",
)
@limiter.limit("10/minute")
async def list_evaluations(
    request: Request,
    session_id: UUID,
    current_user: UserOut = Depends(get_current_active_user),
    eval_service: EvaluationService = Depends(get_evaluation_service),
) -> list[EvaluationOut]:
    return await eval_service.list_evaluations(
        session_id=session_id,
        user_id=current_user.id,
    )


# ── Batch evaluate session ─────────────────────────────────────────────────────

@router.post(
    "/{session_id}/evaluation",
    response_model=list[EvaluationOut],
    status_code=status.HTTP_201_CREATED,
    summary="Evaluate all un-evaluated assistant turns in a session",
)
@limiter.limit("3/minute")
async def evaluate_session(
    request: Request,
    session_id: UUID,
    current_user: UserOut = Depends(get_current_superuser),
    eval_service: EvaluationService = Depends(get_evaluation_service),
    payload: EvaluateSessionRequest | None = None,
    limit: int = Query(200, ge=1, le=500),
) -> list[EvaluationOut]:
    effective_limit = (payload.limit if payload else None) or limit
    effective_model = payload.model if payload else None

    return await eval_service.evaluate_session(
        session_id=session_id,
        user_id=current_user.id,
        limit=effective_limit,
        model=effective_model,
    )


# ── Get single turn evaluation (dynamic path — must come last) ────────────────

@router.get("/{session_id}/evaluation/{turn_id}",
    response_model=EvaluationOut,
    summary="Get the evaluation result for a specific turn",
)
@limiter.limit("10/minute")
async def get_evaluation(
    request: Request,
    session_id: UUID,
    turn_id: UUID,
    current_user: UserOut = Depends(get_current_active_user),
    eval_service: EvaluationService = Depends(get_evaluation_service),
) -> EvaluationOut:
    return await eval_service.get_evaluation(
        message_id=turn_id,
        session_id=session_id,
        user_id=current_user.id,
    )