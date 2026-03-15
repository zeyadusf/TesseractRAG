from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.api.v1 import sessions, documents, chat
from app.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
	# === STARTUP CODE HERE ===
	# runs once when server starts
	logger.info("Tesseract starting...")
	yield  # server is now running and handling requests"
	# === SHUTDOWN CODE HERE ===
	# runs once when server stops
	logger.info("Tesseract shutting Down...")

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

@app.get("/health", tags=["Health"])
async def health():
	return {
			"status": "healthy",
			"version": "1.0.0"
	}