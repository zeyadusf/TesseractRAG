from langdetect import detect, LangDetectException
from backend.core import get_config, get_logger

config = get_config()
logger = get_logger(__name__)

SUPPORTED_LANGUAGES = config.SUPPORTED_LANGUAGES
DEFAULT_LANGUAGE = config.DEFAULT_LANGUAGE

def detect_language(text: str) -> str:
    if not text or not text.strip():
        return "unknown"
    
    try:
        sample = text[:500] if len(text) > 500 else text
        return detect(sample)
    except LangDetectException:
        logger.warning("Language detection failed for text snippet: %s", text[:50])
        return "unknown"


def is_supported_language(lang_code: str) -> bool:

    if not lang_code or lang_code == "unknown":
        return False
    return lang_code.lower() in SUPPORTED_LANGUAGES


def get_language_or_fallback(text: str) -> str:
    detected = detect_language(text)
    if is_supported_language(detected):
        return detected
    logger.info(f"Language '{detected}' not supported, falling back to '{DEFAULT_LANGUAGE}'")
    return DEFAULT_LANGUAGE