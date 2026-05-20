from fastapi import FastAPI,HTTPException
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware


from fastapi.responses import JSONResponse

from backend.core.limiter import limiter
from backend.core import get_logger, get_config
from backend.core.dependencies import (aclose_evaluator,aclose_generation_pipeline,aclose_retrieval_pipeline,aclose_embedder,get_async_session)
from backend.core.security.seed_superuser import seed_superuser
from backend.rag.components.reranker.bge_hf import get_reranker
from backend.models.healthStatus import HealthStatus
from backend.api.v1 import auth, chat, documents, evaluation, sessions, admin
from backend.services.exceptions import *
from backend.storage.db.connections.connection import engine,SessionLocal
logger = get_logger(__name__)
config = get_config()


# 
@asynccontextmanager
async def lifespan(app:FastAPI):

    # ── STARTUP ────────────────────────────────────── 
    logger.info("TesseractRAG starting...")
    # app.state.db_engine = engine
    async for session in get_async_session():
        await seed_superuser(session)
    await get_reranker().warmup()
    yield
    # ── SHUTDOWN ─────────────────────────────────────
    logger.info("TesseractRAG shutting down...")
    # temp
    # await app.state.db_engine.dispose()
    # logger.info("TesseractRAG [DataBase engine] shutting down...")
    await aclose_embedder()
    logger.info("TesseractRAG [Embedder Cilent] Close...")
    await aclose_evaluator()
    logger.info("TesseractRAG [Evaluatot Cilent] Close...")
    await aclose_generation_pipeline()
    logger.info("TesseractRAG [Query Rewriter Cilent] Close...")
    logger.info("TesseractRAG [Generator Cilent] Close...")
    await aclose_retrieval_pipeline()
    logger.info("TesseractRAG [Reranker Cilent] Close...")
    logger.info("TesseractRAG shutdown complete ✅✅✅")

app = FastAPI(
    lifespan=lifespan,
    title=config.APP_NAME,
    version=config.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    )

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)  

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500",
        "http://127.0.0.1:5500",
        "https://tesseractrag.ziayd-usf.workers.dev",],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ــــــــــــــــــــــــــــ

#  Attach error handler so 429 returns a clean JSON response

@app.exception_handler(AuthenticationError)
async def auth_exception_handler(request, exc):
    return JSONResponse(
        status_code=401,
        content={"detail": str(exc)},
    )

@app.exception_handler(ConflictError)
async def conflict_handler(request, exc):
    return JSONResponse(
        status_code=409,
        content={"detail": str(exc)},
    )

# ــــــــــــــــــــــــــــ

app.include_router(
    sessions.router,
    prefix="/api/v1/sessions",
    tags=["Sessions"],
)

app.include_router(
    documents.router,
    prefix="/api/v1/sessions",
    tags=["Documents"],
)

app.include_router(
    chat.router,
    prefix="/api/v1/sessions",
    tags=["Chats"],
)

app.include_router(
    evaluation.router,
    prefix="/api/v1/sessions",
    tags=["Evaluation"],
)

app.include_router(
    auth.router,
    prefix="/api/v1/auth",
    tags=["Auth"],
)
app.include_router(
    admin.router,
    prefix="/api/v1/admin",
    tags=["Admin"],
)

# ــــــــــــــــــــــــــــ

@app.get('/health',tags=['Health'])
async def health():

    return HealthStatus(
        status= "healthy",
        version=config.APP_VERSION
    )