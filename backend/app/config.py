from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Priority order (highest to lowest):
        1. System environment variables (setx / os.environ)
        2. Values in .env file
        3. Default values defined here
    """

    # ── App ────────────────────────────────────────────
    APP_NAME: str = "TesseractRAG"
    DEBUG: bool = False

    # ── HuggingFace ────────────────────────────────────
    # Optional so the app starts even without a token.
    # Token is only required when making LLM API calls.
    HF_API_TOKEN: Optional[str] = None

    # ── Models ─────────────────────────────────────────
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    # LLM_MODEL: str ="mistralai/Mistral-7B-Instruct-v0.3:hf-inference"
    LLM_MODEL_1 : str = "meta-llama/Llama-3.1-8B-Instruct"
    # ── Storage ────────────────────────────────────────
    DATA_DIR: str = "./data"

    # ── Chunking ───────────────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    FINAL_TOP_K: int = 3
    DIM_FAISS : int = 384
    # ----
    MAX_CONTEXT_CHARS : int = 3000 # context length 

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

    lru_cache ensures Settings() is only instantiated once
    for the entire application lifetime.

    Usage anywhere in the codebase:
        from app.config import get_settings
        settings = get_settings()
        print(settings.APP_NAME)
    """
    return Settings()