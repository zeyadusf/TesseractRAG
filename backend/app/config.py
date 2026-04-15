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
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    # ── App ────────────────────────────────────────────────────────
    APP_NAME: str = "TesseractRAG"
    DEBUG: bool = False

    # ── HuggingFace ────────────────────────────────────────────────
    HF_API_TOKEN: Optional[str] = None
    COHERE_API_KEY : Optional[str] = None

    # ── Models ─────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    # RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    LLM_MODEL_1: str = "meta-llama/Llama-3.1-8B-Instruct"

    # ── Chunking ───────────────────────────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    FINAL_TOP_K: int = 3
    DIM_FAISS: int = 384
    MAX_CONTEXT_CHARS: int = 3000

    # ── Cloudflare R2 ──────────────────────────────────────────────
    # Get these from: Cloudflare Dashboard → R2 → Manage R2 API Tokens
    # Endpoint format: https://{account_id}.r2.cloudflarestorage.com
    R2_ENDPOINT_URL: str = ""
    R2_ACCESS_KEY_ID: str = ""
    R2_SECRET_ACCESS_KEY: str = ""
    R2_BUCKET_NAME: str = "tesseractrag"

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