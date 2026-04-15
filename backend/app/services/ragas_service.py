from app.dependencies import get_session_manager
from app.core.evaluation.ragas_evaluator import RagasEvaluator
from app.utils.logger import get_logger
from datetime import datetime, timezone
import time

logger = get_logger(__name__)

# def run_ragas_eval(sample: dict, session_id, owner_id):
#     start = time.perf_counter()

#     try:
#         result = RagasEvaluator.evaluate(**sample)
#         duration = time.perf_counter() - start

#         manager = get_session_manager()
#         session = manager.get_session(session_id, owner_id)

#         evaluation = {
#             "metrics": {
#                 "faithfulness":      result.get("faithfulness"),
#                 "answer_relevancy":  result.get("answer_relevancy"),
#                 "context_precision": result.get("context_precision"),  # ← ناقصة
#             },
#             "question":   sample.get("query"),       # ← عشان يظهر في الـ card
#             "answer":     sample.get("response_text"), # ← عشان يظهر في الـ card
#             "timestamp":  datetime.now(timezone.utc).isoformat(), # ← عشان الـ timestamp
#             "duration_sec": round(duration, 3),
#         }

#         session.ragas_evaluation.append(evaluation)
#         manager._save_metadata(session)

#         logger.info(f"RAGAS done in {duration:.2f}s : {evaluation}")

#     except Exception as e:
#         logger.error(f"RAGAS error: {e}")

def run_ragas_eval(sample: dict, session_id, owner_id):
    start = time.perf_counter()
    
    try:
        manager = get_session_manager()
        # logger.info(f"RAGAS START | session={session_id} | owner={owner_id}")
        # logger.info(f"RAGAS | sessions in memory: {list(manager._sessions.keys())}")
        
        result = RagasEvaluator.evaluate(**sample)
        duration = time.perf_counter() - start
        
        session = manager.get_session(session_id, owner_id)
        
        evaluation = {
            "metrics": {
                "faithfulness":      result.get("faithfulness"),
                "answer_relevancy":  result.get("answer_relevancy"),
                "context_precision": result.get("context_precision"),
            },
            "question":    sample.get("query"),
            "answer":      sample.get("response_text"),
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "duration_sec": round(duration, 3),
        }
        
        session.ragas_evaluation.append(evaluation)
        manager._save_metadata(session)
        logger.info(f"Evaluation SAVED | duration={duration:.2f}s")

    except Exception as e:
        logger.error(f"Evaluation error: {type(e).__name__}: {e}", exc_info=True)