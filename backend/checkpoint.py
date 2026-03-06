from app.config import get_settings
from app.utils.logger import get_logger


if __name__ == "__main__":
    settings = get_settings()
    logger = get_logger("checkpoint")

    logger.info("─────────────────────────────────────")
    logger.info("TesseractRAG — Phase 0 Checkpoint")
    logger.info("─────────────────────────────────────")
    logger.info("App name:        %s", settings.APP_NAME)
    logger.info("Embedding model: %s", settings.EMBEDDING_MODEL)
    logger.info("LLM model:       %s", settings.LLM_MODEL)
    logger.info("Data directory:  %s", settings.DATA_DIR)
    logger.info("Chunk size:      %s", settings.CHUNK_SIZE)
    logger.info("HF token set:    %s", bool(settings.HF_API_TOKEN))
    logger.info("─────────────────────────────────────")

    assert settings.APP_NAME == "TesseractRAG", "APP_NAME missing"
    assert settings.CHUNK_SIZE == 512, "CHUNK_SIZE wrong type"
    assert isinstance(settings.DEBUG, bool), "DEBUG should be bool"

    print()
    print("✅ Phase 0 complete — all systems ready")
