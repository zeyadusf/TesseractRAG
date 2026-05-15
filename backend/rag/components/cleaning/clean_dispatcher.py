from .ar_clean import ArabicPostProcessor
from .en_clean import EnglishPostProcessor
from typing import Dict, Type, Any
from backend.core import get_config
from backend.models.metadata import Metadata


class CleanerDispatcher:
    def __init__(self):
        self._processors: Dict[str, Type] = {
            "ar": ArabicPostProcessor,
            "en": EnglishPostProcessor,
        }

    def clean(self, text: str, metadata: Metadata) -> Dict[str, Any]:
        config = get_config()

        if not text:
            return {"text": "", "metadata": metadata.model_dump()}

        language = metadata.language or config.DEFAULT_LANGUAGE

        processor_cls = self._processors.get(language)

        if not processor_cls:
            processor_cls = self._processors[config.DEFAULT_LANGUAGE]

        processor = processor_cls()

        cleaned_text = processor.process(text)

        return {
            "text": cleaned_text,
            "metadata": metadata
        }