from typing import Optional,AsyncGenerator
from .config import get_config
from backend.storage.db.db_dispatcher import DBDispatcher
from backend.storage.db.session_dispatcher import get_session



_embedder_instance = None
_embedder_model: Optional[str] = None
_evaluator_instance = None
_generation_pipeline_instance = None
_retrieval_pipeline_instance = None

# ── Getters 

async def get_db() -> AsyncGenerator[DBDispatcher, None]:
    try:
        async for session in get_session():
            db = DBDispatcher(session)
            yield db
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.commit()
        await session.close()

async def get_async_session() :
    try:
        async for session in get_session():
            yield session
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.commit()
        await session.close()

def get_embedder(embed_model_name: str = None) :
    from backend.rag.components.embedding.embed_dispatcher import EmbedderDispatcher

    global _embedder_instance, _embedder_model

    resolved = embed_model_name or get_config().DEFAULT_EMBED

    if _embedder_instance is None:
        _embedder_instance = EmbedderDispatcher(resolved)
        _embedder_model = resolved
        return _embedder_instance

    if resolved != _embedder_model:
        raise RuntimeError(
            f"Embedder already initialised with model '{_embedder_model}'. "
            f"Cannot switch to '{resolved}' at runtime. "
            "Call close_embedder() first if you need a different model."
        )

    return _embedder_instance

def get_evaluator():
    from backend.rag.components.evaluators.evaluation_dispatcher import EvaluatorDispatcher
    global _evaluator_instance
    if _evaluator_instance is None:
        _evaluator_instance = EvaluatorDispatcher()
    return _evaluator_instance

def get_generation_pipeline() :
    from backend.rag.pipelines.generation_pipeline import GenerationPipeline
    global _generation_pipeline_instance
    if _generation_pipeline_instance is None:
        _generation_pipeline_instance = GenerationPipeline()
    return _generation_pipeline_instance

def get_retrieval_pipeline() :
    from backend.rag.pipelines.retrieval_pipeline import RetrievalPipeline
    global _retrieval_pipeline_instance
    if _retrieval_pipeline_instance is None:
        _retrieval_pipeline_instance = RetrievalPipeline()
    return _retrieval_pipeline_instance

# ── Closers 

async def aclose_embedder() -> None:
    global _embedder_instance, _embedder_model
    if _embedder_instance is not None:
        await _embedder_instance.aclose()
        _embedder_instance = None
        _embedder_model = None

async def aclose_evaluator() -> None:
    global _evaluator_instance
    if _evaluator_instance is not None:
        await _evaluator_instance.aclose()
        _evaluator_instance = None

async def aclose_generation_pipeline() -> None:
    global _generation_pipeline_instance
    if _generation_pipeline_instance is not None:
        await _generation_pipeline_instance.aclose()
        _generation_pipeline_instance = None

async def aclose_retrieval_pipeline() -> None:
    global _retrieval_pipeline_instance
    if _retrieval_pipeline_instance is not None:
        await _retrieval_pipeline_instance.aclose()
        _retrieval_pipeline_instance = None