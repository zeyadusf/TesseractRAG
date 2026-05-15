# backend/services/evaluation_service.py
from __future__ import annotations

from uuid import UUID
from typing import Optional
import asyncio

from backend.core.logger import get_logger
from backend.storage.db.db_dispatcher import DBDispatcher
from backend.rag.components.evaluators.evaluation_dispatcher import EvaluatorDispatcher
from backend.models.evaluation import EvaluationOut, SessionEvaluationSummary
from backend.models.enums.message_role import MessageRole
from backend.storage.db.connections.connection import SessionLocal
from .base_service import BaseService
from .exceptions import NotFoundError, ValidationError

logger = get_logger(__name__)


class EvaluationService(BaseService):

    def __init__(self, db: DBDispatcher, evaluator: EvaluatorDispatcher = None) -> None:
        super().__init__(db)
        self._evaluator = evaluator

    # ── Background evaluation (called from chat endpoint) ──────────────────────

    async def evaluate_message_background(
        self,
        session_id: UUID,
        message_id: UUID,
        question: str,
        answer: str,
        user_id: UUID,
        model: Optional[str] = None,
    ) -> None:
        """
        Lightweight evaluation triggered as a background task from the chat endpoint.
        Opens its own DB session — independent from the request session which is
        already closed by the time this runs.
        Failures are logged and swallowed so they never affect the chat response.
        """
        
        # await asyncio.sleep(5) 
        async with SessionLocal() as session:
            db = DBDispatcher(session)
            try:
                # ── Fetch source chunks ────────────────────────────────────────
                message = await db.messages.get_message_for_evaluation(
                    session_id=session_id,
                    message_id=message_id,
                    role=MessageRole.ASSISTANT,
                )

                if not message or not message.source_chunks:
                    logger.warning(
                        f"[EVAL-BG] Skipping message {message_id} — no source chunks"
                    )
                    return

                contexts = [
                    chunk.get("content") or chunk.get("text") or chunk.get("page_content") or ""
                    for chunk in message.source_chunks[:5]
                ]
                contexts = [c for c in contexts if c]

                if not contexts:
                    logger.warning(
                        f"[EVAL-BG] Skipping message {message_id} — empty contexts"
                    )
                    return

                # ── Run evaluator ──────────────────────────────────────────────
                logger.debug(f"[EVAL-BG] Evaluating message {message_id}")
                eval_result = await self._evaluator.evaluate(
                    query=question,
                    response_text=answer,
                    contexts=contexts,
                    model=model,
                )

                # ── Persist ────────────────────────────────────────────────────
                await db.evaluations.upsert_by_message(
                    message_id=message_id,
                    session_id=session_id,
                    faithfulness=eval_result.get("faithfulness"),
                    answer_relevancy=eval_result.get("answer_relevancy"),
                    context_precision=eval_result.get("context_precision"),
                    context_recall=eval_result.get("context_recall"),
                    evaluator="cohere",
                    eval_model=eval_result.get("eval_model"),
                    raw_result={
                        "reasoning": eval_result.get("reasoning"),
                        "eval_model": eval_result.get("eval_model"),
                        **{
                            k: v for k, v in eval_result.items()
                            if k not in [
                                "faithfulness", "answer_relevancy",
                                "context_precision", "context_recall",
                                "reasoning", "eval_model",
                            ]
                        },
                    },
                )
                await session.commit()
                logger.debug(f"[EVAL-BG] Done — message {message_id}")

            except Exception as exc:
                await session.rollback()
                logger.warning(f"[EVAL-BG] Failed for message {message_id}: {exc}")

    # ── Manual evaluation on a single message ──────────────────────────────────

    async def evaluate_message(
        self,
        message_id: UUID,
        session_id: UUID,
        user_id: UUID,
        model: Optional[str] = None,
    ) -> EvaluationOut:
        """
        Run evaluation on one assistant message and persist results.
        Used by evaluate_session (batch) — reconstructs question from DB history.
        """
        session = await self.db.sessions.get_by_id_and_user(session_id, user_id)
        if session is None:
            raise NotFoundError("Session", str(session_id))

        message = await self.db.messages.get_message_for_evaluation(
            session_id=session_id,
            message_id=message_id,
            role=MessageRole.ASSISTANT,
        )
        if not message:
            raise NotFoundError("Message", str(message_id))

        if not message.source_chunks:
            raise ValidationError(
                "This message has no source chunks recorded. "
                "It cannot be evaluated for context metrics."
            )

        preceding = await self.db.messages.get_preceding_user_message(
            session_id=session_id,
            before_message_id=message_id,
        )
        if preceding is None:
            raise ValidationError(
                "Could not find the user question preceding this message"
            )

        question = preceding.content
        answer = message.content
        contexts = [
            chunk.get("content") or chunk.get("text") or chunk.get("page_content") or ""
            for chunk in message.source_chunks[:5]
        ]
        contexts = [c for c in contexts if c]

        if not contexts:
            raise ValidationError(
                "Source chunks exist but contain no extractable text."
            )

        logger.info(
            f"[EVAL] Running {self._config.DEFAULT_EVALUATOR} evaluation "
            f"on message: {message_id}"
        )
        eval_result = await self._evaluator.evaluate(
            query=question,
            response_text=answer,
            contexts=contexts,
            model=model,
        )

        evaluation = await self.db.evaluations.upsert_by_message(
            message_id=message_id,
            session_id=session_id,
            faithfulness=eval_result.get("faithfulness"),
            answer_relevancy=eval_result.get("answer_relevancy"),
            context_precision=eval_result.get("context_precision"),
            context_recall=eval_result.get("context_recall"),
            evaluator="cohere",
            eval_model=eval_result.get("eval_model"),
            raw_result={
                "reasoning": eval_result.get("reasoning"),
                "eval_model": eval_result.get("eval_model"),
                **{
                    k: v for k, v in eval_result.items()
                    if k not in [
                        "faithfulness", "answer_relevancy",
                        "context_precision", "context_recall",
                        "reasoning", "eval_model",
                    ]
                },
            },
        )

        return EvaluationOut.model_validate(evaluation)

    # ── Batch evaluate all messages in a session ───────────────────────────────

    async def evaluate_session(
        self,
        session_id: UUID,
        user_id: UUID,
        model: Optional[str] = None,
        limit: int = 200,
    ) -> list[EvaluationOut]:
        session = await self.db.sessions.get_by_id_and_user(session_id, user_id)
        if session is None:
            raise NotFoundError("Session", str(session_id))

        messages = await self.db.messages.list_by_session(session_id, limit=limit)
        assistant_messages = [
            m for m in messages
            if m.role == MessageRole.ASSISTANT and m.source_chunks
        ]

        results: list[EvaluationOut] = []
        for msg in assistant_messages:
            existing = await self.db.evaluations.get_by_message(msg.id)
            if existing:
                results.append(EvaluationOut.model_validate(existing))
                continue
            try:
                result = await self.evaluate_message(
                    msg.id, session_id, user_id, model=model
                )
                results.append(result)
            except Exception as exc:
                logger.warning(f"[EVAL] Failed for message {msg.id}: {exc}")

        return results

    # ── Queries ────────────────────────────────────────────────────────────────

    async def get_evaluation(
        self, message_id: UUID, session_id: UUID, user_id: UUID
    ) -> EvaluationOut:
        session = await self.db.sessions.get_by_id_and_user(session_id, user_id)
        if session is None:
            raise NotFoundError("Session", str(session_id))

        evaluation = await self.db.evaluations.get_by_message(message_id)
        if evaluation is None:
            raise NotFoundError("Evaluation", str(message_id))
        return EvaluationOut.model_validate(evaluation)

    async def get_session_summary(
        self, session_id: UUID, user_id: UUID
    ) -> SessionEvaluationSummary:
        session = await self.db.sessions.get_by_id_and_user(session_id, user_id)
        if session is None:
            raise NotFoundError("Session", str(session_id))

        averages = await self.db.evaluations.get_session_averages(session_id)
        return SessionEvaluationSummary(
            session_id=session_id,
            avg_faithfulness=averages["avg_faithfulness"],
            avg_answer_relevancy=averages["avg_answer_relevancy"],
            avg_context_precision=averages["avg_context_precision"],
            avg_context_recall=averages["avg_context_recall"],
            total_evaluated=averages["total_evaluated"],
        )

    async def list_evaluations(
        self, session_id: UUID, user_id: UUID
    ) -> list[EvaluationOut]:
        session = await self.db.sessions.get_by_id_and_user(session_id, user_id)
        if session is None:
            raise NotFoundError("Session", str(session_id))

        evaluations = await self.db.evaluations.list_by_session(session_id)
        return [EvaluationOut.model_validate(e) for e in evaluations]

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def aclose(self) -> None:
        await self._evaluator.aclose()
        logger.debug("[EvaluationService] closed")