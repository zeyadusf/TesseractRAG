"""
main.py
-------
FastAPI application entry point for TesseractRAG.

CORS is configured to allow X-Owner-ID as a custom header — this is required
for browser clients to send the header cross-origin. Without explicitly listing
it in allow_headers, the browser will block the preflight OPTIONS request.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.api.v1 import sessions, documents, chat, evaluation
from app.utils.logger import get_logger

from app.core.ingestion.embedder import get_embedding_model
from app.core.retrieval.reranker import get_reranker
from app.core.session_manager import SessionManager
from app.dependencies import get_session_manager

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ──────────────────────────────────────
    logger.info("TesseractRAG starting...")

    # Load embedding model — needed for every request
    logger.info("Loading embedding model...")
    get_embedding_model()
    logger.info("Embedding model ready ✓")

    # Skip reranker preload — load lazily on first request
    # to stay within Railway free tier 512MB RAM limit
    logger.info("Reranker will load on first request (memory optimization)")

    # Reload persisted sessions
    logger.info("Loading persisted sessions...")
    manager = get_session_manager()
    logger.info(f"Sessions loaded: {len(manager._sessions)} ✓")

    logger.info("TesseractRAG ready ✓")
    yield

    # ── SHUTDOWN ─────────────────────────────────────
    logger.info("TesseractRAG shutting down...")


app = FastAPI(
    lifespan=lifespan,
    title="TesseractRAG",
    description="Adaptive Multi-Session RAG System",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "https://tesseractrag.ziayd-usf.workers.dev","*"
    ],
    allow_methods=["*"],
    allow_headers=[
        "*",
        "X-Owner-ID",
    ],
    # Expose X-Owner-ID in responses so the frontend can read it if needed
    expose_headers=["X-Owner-ID"],
)

app.include_router(
    sessions.router,
    prefix="/api/v1/sessions",
    tags=["Sessions"],
)

app.include_router(
    chat.router,
    prefix="/api/v1/sessions",
    tags=["Chat"],
)

app.include_router(evaluation.router,
        prefix="/api/v1/sessions",
        tags=["Evaluation"],)

app.include_router(
    documents.router,
    prefix="/api/v1/sessions",
    tags=["Documents"],
)


@app.get("/health", tags=["Health"])
async def health():
    """
    Health check endpoint for deployment monitoring.

    Returns model readiness state. The reranker is lazily loaded so
    it will always report 'lazy_load' on this endpoint.
    """
    try:
        embedder_ready = get_embedding_model() is not None
    except Exception:
        embedder_ready = False

    return {
        "status":       "healthy" if embedder_ready else "loading",
        "version":      "1.2.0",
        "models_ready": embedder_ready,
        "embedder":     embedder_ready,
        "reranker":     "lazy_load",
        "owner_id_auth": "header",      # documents the auth mechanism in use
    }
