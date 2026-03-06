import logging
import sys
from functools import lru_cache


@lru_cache()
def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger for the given module name.

    Usage in any module:
        from app.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Session created: %s", session_id)

    Passing __name__ gives you logger names like:
        app.core.session_manager
        app.core.ingestion.parser
    This makes it easy to trace exactly which file
    produced each log line.

    Args:
        name: Module name, typically pass __name__

    Returns:
        Configured Logger instance
    """
    from app.config import get_settings
    settings = get_settings()

    logger = logging.getLogger(name)

    # Guard: don't add handlers if already configured
    # Without this check you get duplicate log lines
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent log messages bubbling up to root logger
    logger.propagate = False

    return logger