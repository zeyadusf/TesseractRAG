"""
config.py
---------
Application settings loaded from environment variables via Pydantic Settings.

Priority order (highest to lowest):
    1. System environment variables
    2. Values in .env file
    3. Default values defined here
"""

from functools import lru_cache
from typing import Optional, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    # ── App ────────────────────────────────────────────────────────
    APP_NAME: str = "TesseractRAG"
    DEBUG: bool = False

    # ── HuggingFace ────────────────────────────────────────────────
    HF_API_TOKEN: Optional[str] = None
    COHERE_API_KEY: Optional[str] = None

    # ── Models ─────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    # RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    LLM_MODEL_1: str = "meta-llama/Llama-3.1-8B-Instruct"

    # ── Chunking ───────────────────────────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    FINAL_TOP_K: int = 3
    DIM_EMBEDDING: int = 384
    MAX_CONTEXT_CHARS: int = 3000

    # ── Cloudflare R2 / S3-Compatible Storage ──────────────────────
    # Get these from: Cloudflare Dashboard → R2 → Manage R2 API Tokens
    # Endpoint format: https://{account_id}.r2.cloudflarestorage.com
    R2_ENDPOINT_URL: str = ""
    R2_ACCESS_KEY_ID: str = ""
    R2_SECRET_ACCESS_KEY: str = ""
    R2_BUCKET_NAME: str = "tesseractrag"

    # ═══════════════════════════════════════════════════════════════
    # 🔌 NEW: Pluggable Backend Selection (S3/FAISS ↔ PostgreSQL/pgvector)
    # ═══════════════════════════════════════════════════════════════

    # Backend for metadata/chunks storage: "s3" | "postgresql"
    METADATA_BACKEND: Literal["s3", "postgresql"] = "s3"

    # Backend for vector search: "faiss" | "pgvector"
    VECTOR_STORE_BACKEND: Literal["faiss", "pgvector"] = "faiss"

    # PostgreSQL connection string (used when METADATA_BACKEND or VECTOR_STORE_BACKEND = postgresql)
    # Format: postgresql+asyncpg://user:pass@host:port/dbname
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/tessrag"

    # Local directory for FAISS indexes (used when VECTOR_STORE_BACKEND = faiss)
    FAISS_INDEX_DIR: str = "./data/sessions"

    # ── Paths ──────────────────────────────────────────────────────
    DATA_DIR: str = "./data"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached singleton Settings instance.
    Reads .env once on first call, returns the same object every time after.
    """
    return Settings()