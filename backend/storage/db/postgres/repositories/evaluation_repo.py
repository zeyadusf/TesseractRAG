from __future__  import annotations
from .base_repo import BaseRepository
from backend.storage.db.postgres.schemas import Evaluation
from backend.models.enums.evaluation_enum import EvaluationTech 
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession



class EvaluationRepository(BaseRepository[Evaluation]):
    def __init__(self,sessionConn:AsyncSession):
        self._s = sessionConn



    async def get_by_id(self, record_id: UUID) -> Evaluation | None:
        result = await self._s.execute(
            select(Evaluation).where(Evaluation.id == record_id)
        )
        return result.scalar_one_or_none()

    async def get_by_message(self, message_id: UUID) -> Evaluation | None:
        """Primary lookup — one evaluation per message."""
        result = await self._s.execute(
            select(Evaluation).where(Evaluation.message_id == message_id)
        )
        return result.scalar_one_or_none()

    async def list_by_session(self, session_id: UUID) -> list[Evaluation]:
        """Return all evaluations for a session, newest first."""
        result = await self._s.execute(
            select(Evaluation)
            .where(Evaluation.session_id == session_id)
            .order_by(Evaluation.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_session_averages(self, session_id: UUID) -> dict:
        """
        Compute average RAGAS scores for a session in a single query.
        Returns a dict with avg scores (None if no evaluations exist yet).

        Used for the session quality dashboard.
        """
        result = await self._s.execute(
            select(
                func.avg(Evaluation.faithfulness).label("avg_faithfulness"),
                func.avg(Evaluation.answer_relevancy).label("avg_answer_relevancy"),
                func.avg(Evaluation.context_precision).label("avg_context_precision"),
                func.avg(Evaluation.context_recall).label("avg_context_recall"),
                func.count(Evaluation.id).label("total_evaluated"),
            ).where(Evaluation.session_id == session_id)
        )
        row = result.one()
        return {
            "avg_faithfulness":      float(row.avg_faithfulness) if row.avg_faithfulness else None,
            "avg_answer_relevancy":  float(row.avg_answer_relevancy) if row.avg_answer_relevancy else None,
            "avg_context_precision": float(row.avg_context_precision) if row.avg_context_precision else None,
            "avg_context_recall":    float(row.avg_context_recall) if row.avg_context_recall else None,
            "total_evaluated":       row.total_evaluated,
        }

    # ── CRUD ───────────────────────────────────────────────────────────────

    async def create(
        self,
        *,
        message_id: UUID,
        session_id: UUID,
        faithfulness: float | None = None,
        answer_relevancy: float | None = None,
        context_precision: float | None = None,
        context_recall: float | None = None,
        evaluator: str =EvaluationTech.DEFAULT,
        eval_model: str | None = None,
        raw_result: dict | None = None,
    ) -> Evaluation:
        obj = Evaluation(
            message_id=message_id,
            session_id=session_id,
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_precision=context_precision,
            context_recall=context_recall,
            evaluator=evaluator,
            eval_model=eval_model,
            raw_result=raw_result or {},
        )
        self._s.add(obj)
        await self._s.flush()
        await self._s.refresh(obj)
        return obj

    async def update(self, record_id: UUID, **kwargs) -> Evaluation | None:
        """
        Used to fill in metric scores after async evaluation completes.
        e.g. await repo.update(eval_id, faithfulness=0.91, answer_relevancy=0.85)
        """
        obj = await self.get_by_id(record_id)
        if obj is None:
            return None
        for field, value in kwargs.items():
            setattr(obj, field, value)
        await self._s.flush()
        await self._s.refresh(obj)
        return obj

    async def upsert_by_message(
        self,
        *,
        message_id: UUID,
        session_id: UUID,
        **metric_kwargs,
    ) -> Evaluation:
        """
        Insert or update evaluation for a message.
        Safe to call multiple times (e.g. re-running RAGAS on the same message).
        """
        existing = await self.get_by_message(message_id)
        if existing:
            return await self.update(existing.id, **metric_kwargs) or existing
        return await self.create(
            message_id=message_id,
            session_id=session_id,
            **metric_kwargs,
        )

    async def delete(self, record_id: UUID) -> bool:
        obj = await self.get_by_id(record_id)
        if obj is None:
            return False
        await self._s.delete(obj)
        await self._s.flush()
        return True


# ── admin ────────────────────────────────────────
    #  
    async def get_global_avg_scores(self) -> dict:
        """
        Global average of all four evaluation metrics across the
        entire system (all users, all sessions, all messages).
        Returns None for each metric if no evaluations exist yet.
        """
        result = await self._s.execute(
            select(
                func.avg(Evaluation.faithfulness).label("avg_faithfulness"),
                func.avg(Evaluation.answer_relevancy).label("avg_answer_relevancy"),
                func.avg(Evaluation.context_precision).label("avg_context_precision"),
                func.avg(Evaluation.context_recall).label("avg_context_recall"),
            )
        )
        row = result.one()
    
        def _f(val) -> float | None:
            return float(val) if val is not None else None
    
        return {
            "avg_faithfulness": _f(row.avg_faithfulness),
            "avg_answer_relevancy": _f(row.avg_answer_relevancy),
            "avg_context_precision": _f(row.avg_context_precision),
            "avg_context_recall": _f(row.avg_context_recall),
        }
    
