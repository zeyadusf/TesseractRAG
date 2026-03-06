import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Priority order (highest to lowest):
        1. System environment variables (setx / os.environ)
        2. Values in .env file
        3. Default values defined here

    This means HF_API_TOKEN set via setx will automatically
    take priority over anything in the .env file.
    """

    # ── App ────────────────────────────────────────────────────────
    APP_NAME: str = "TesseractRAG"
    DEBUG: bool = False

    # ── HuggingFace ────────────────────────────────────────────────
    HF_API_TOKEN: str = os.getenv('HF_API_TOKEN') 

    # ── Models ─────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    LLM_MODEL: str = "mistralai/Mistral-7B-Instruct-v0.3"

    # ── Storage ────────────────────────────────────────────────────
    DATA_DIR: str = "./data"

    # ── Chunking ───────────────────────────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    FINAL_TOP_K: int = 3

    model_config = SettingsConfigDict(
        env_file="backend/.env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached singleton Settings instance.

    lru_cache ensures Settings() is only instantiated once
    for the entire application lifetime — reads .env once,
    then returns the same object on every subsequent call.

    Usage anywhere in the codebase:
        from app.config import get_settings
        settings = get_settings()
        print(settings.APP_NAME)
    """
    return Settings()