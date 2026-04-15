"""
evaluation.py
-------------
Endpoints for RAG evaluation results per session.

Provides:
    - Full evaluation history
    - Aggregated metrics (average scores)
"""

from fastapi import APIRouter, Depends
from app.dependencies import get_session_manager, get_owner_id
from app.core.session_manager import SessionManager
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/{session_id}/evaluation")
def get_session_evaluation(
    session_id: str,
    manager: SessionManager = Depends(get_session_manager),
    owner_id: str = Depends(get_owner_id),
):
    """
    Return evaluation history + aggregated metrics for a session.
    """
    logger.info(f"GET evaluation | session_id={session_id} | owner_id={owner_id}")

    session = manager.get_session(session_id, owner_id=owner_id)

    evaluations = session.ragas_evaluation

    # ─────────────────────────────────────────────
    # Case 1: No evaluations yet
    # ─────────────────────────────────────────────
    if not evaluations:
        return {
            "count": 0,
            "avg_metrics": None,
            "history": [],
            "status": "no_evaluations_yet",
        }

    # ─────────────────────────────────────────────
    # Case 2: Compute averages
    # ─────────────────────────────────────────────
    total = 1 if len(evaluations) == 0 else len(evaluations)

    avg_faithfulness = sum(
        e["metrics"].get("faithfulness") or 0 for e in evaluations
    ) / total

    avg_relevancy = sum(
        e["metrics"].get("answer_relevancy") or 0 for e in evaluations
    ) / total

    avg_precision = sum(
        e["metrics"].get("context_precision") or 0 for e in evaluations
    ) / total

    return {
        "count": total,
        "avg_metrics": {
            "faithfulness": round(avg_faithfulness, 3),
            "answer_relevancy": round(avg_relevancy, 3),
            "context_precision": round(avg_precision, 3),
        },
        "history": evaluations,
    }