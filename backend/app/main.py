from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.api.v1 import sessions, documents, chat
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

    # Load embedding model into memory NOW — blocks until ready
    logger.info("Loading embedding model...")
    get_embedding_model()
    logger.info("Embedding model ready ✓")

    # Load reranker model into memory NOW
    logger.info("Loading reranker model...")
    get_reranker()
    logger.info("Reranker model ready ✓")

    # Reload persisted sessions from disk
    logger.info("Loading persisted sessions...")
    manager = get_session_manager()
    logger.info(f"Sessions loaded: {len(manager._sessions)} ✓")

    logger.info("TesseractRAG ready — all models loaded")
    yield

    # ── SHUTDOWN ─────────────────────────────────────
    logger.info("TesseractRAG shutting down...")

app = FastAPI(
  lifespan = lifespan,
	title="TesseractRAG",
	description="Adaptive Multi-Session RAG System",
	version="1.0.0",
	docs_url="/docs",
	redoc_url="/redoc"
)

app.add_middleware(
	CORSMiddleware,
	allow_credentials=True,
	allow_origins=["*"],
	allow_methods=["*"],
	allow_headers=["*"],
)

app.include_router(
	sessions.router,
	prefix = "/api/v1/sessions",
	tags = ["Sessions"]
)

app.include_router(
	chat.router,
	prefix = "/api/v1/sessions",
	tags = ["Chat"]
)

app.include_router(
	documents.router,
	prefix = "/api/v1/sessions",
	tags = ["Documents"]
)


@app.get("/health",tags=["Health"])
async def health():
    try:
        embedder_ready = get_embedding_model() is not None
        reranker_ready = get_reranker() is not None
    except Exception:
        embedder_ready = False
        reranker_ready = False

    all_ready = embedder_ready and reranker_ready
    return {
        "status": "healthy" if all_ready else "loading",
        "models_ready": all_ready,
        "embedder": embedder_ready,
        "reranker": reranker_ready,
    }