from typing import List, Dict
from collections import Counter
from backend.rag.components.lang_detector import detect_language, is_supported_language
from backend.core import get_logger, get_config

logger = get_logger(__name__)
config = get_config()


class CrossLanguageStrategySelector:
    """
    Determines if a query should use 'hybrid' strategy due to language mismatch.
    
    Rationale: When query language ≠ document language, pure lexical (BM25)
    often fails due to lack of keyword overlap. Hybrid (BM25 + vector) provides
    the best cross-lingual retrieval performance.
    """
    
    def __init__(
        self,
        *,
        min_chunks_to_analyze: int = 5,
        language_confidence_threshold: float = 0.7,
        enabled: bool = True,
    ):
        self.min_chunks = min_chunks_to_analyze
        self.confidence_threshold = language_confidence_threshold
        self.enabled = enabled
    
    def should_force_semantic(self, query: str, chunks: List[Dict]) -> bool:
        """
        Return True if query language differs from majority language of chunks.
        
        Args:
            query: User's question
            chunks: List of chunk dicts with 'content' key
            
        Returns:
            bool: True if hybrid strategy should be forced
        """
        if not self.enabled or not query.strip() or not chunks:
            return False
        
        # Detect query language
        query_lang = detect_language(query)
        if not is_supported_language(query_lang):
            logger.debug(f"[CROSS-LANG] Query language '{query_lang}' not supported, skipping override")
            return False
        
        # Analyze chunk languages (sample first N chunks for performance)
        chunk_langs = []
        for chunk in chunks[:self.min_chunks]:
            content = chunk.get("content", "")
            if len(content) < 50:  # Skip very short chunks
                continue
            lang = detect_language(content[:500])  # First 500 chars enough
            if is_supported_language(lang):
                chunk_langs.append(lang)
        
        if not chunk_langs:
            logger.debug("[CROSS-LANG] No valid chunk languages detected, skipping override")
            return False
        
        # Find majority language in chunks
        lang_counts = Counter(chunk_langs)
        majority_lang, majority_count = lang_counts.most_common(1)[0]
        majority_ratio = majority_count / len(chunk_langs)
        
        # Decide: force hybrid if majority is clear AND languages differ
        if majority_ratio >= self.confidence_threshold and query_lang != majority_lang:
            logger.info(
                f"[CROSS-LANG] Language mismatch detected: "
                f"query='{query_lang}' vs chunks='{majority_lang}' (confidence={majority_ratio:.2f}). "
                f"Forcing 'semantic' strategy."
            )
            return True
        
        logger.debug(
            f"[CROSS-LANG] No mismatch: query='{query_lang}', "
            f"chunks_majority='{majority_lang}' (ratio={majority_ratio:.2f})"
        )
        return False


# Singleton factory for easy injection
_cross_lang_selector_instance: CrossLanguageStrategySelector | None = None


def _config_to_dict(config_obj) -> dict:
    """
    Safely convert Pydantic model (v1/v2) or other config objects to dict.
    
    Handles:
    - Pydantic v2: .model_dump()
    - Pydantic v1: .dict()
    - Plain objects: vars()/__dict__
    - None or dict: pass-through
    """
    if config_obj is None:
        return {}
    if isinstance(config_obj, dict):
        return config_obj
    # Pydantic v2
    if hasattr(config_obj, "model_dump") and callable(getattr(config_obj, "model_dump")):
        return config_obj.model_dump()
    # Pydantic v1
    if hasattr(config_obj, "dict") and callable(getattr(config_obj, "dict")):
        return config_obj.dict()
    # Plain object with __dict__
    if hasattr(config_obj, "__dict__"):
        return {
            k: v for k, v in vars(config_obj).items() 
            if not k.startswith("_") and not callable(v)
        }
    # Fallback
    logger.warning(f"[CONFIG] Unable to convert config object to dict: {type(config_obj)}")
    return {}


def get_cross_language_selector(
    *,
    reset: bool = False,
    **kwargs,
) -> CrossLanguageStrategySelector:
    """
    Get or create the shared CrossLanguageStrategySelector instance.
    
    Args:
        reset: Force recreation of the singleton instance
        **kwargs: Override config values at runtime
        
    Returns:
        CrossLanguageStrategySelector: Configured selector instance
    """
    global _cross_lang_selector_instance
    
    if reset or _cross_lang_selector_instance is None:
        # Load config from app config (Pydantic model expected)
        selector_config = config.CROSS_LANGUAGE_STRATEGY
        
        # Safely convert Pydantic model to dict for unpacking
        config_dict = _config_to_dict(selector_config)
        
        # Filter out None values to avoid overriding defaults unintentionally
        config_dict = {k: v for k, v in config_dict.items() if v is not None}
        
        # Merge runtime kwargs (higher priority) with config values
        init_params = {**config_dict, **kwargs}
        
        _cross_lang_selector_instance = CrossLanguageStrategySelector(**init_params)
        logger.debug(f"[CROSS-LANG] Selector initialized with: {init_params}")
    
    return _cross_lang_selector_instance