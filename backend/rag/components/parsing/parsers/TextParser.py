from backend.rag.components.parsing.base_parser import BaseParser
from ...lang_detector import get_language_or_fallback
from pathlib import Path
from backend.models.metadata import Metadata

class TextParser(BaseParser):
    """Paresr txt md extension  ~ _ -"""
    def parse(self, file_bytes, filename):
        text = file_bytes.decode("utf-8", errors="ignore")
        
        language = get_language_or_fallback(text)
        ext = Path(filename).suffix.lower()
        count_chars = len(text)

        yield {
            "text" :text.strip(),
            "metadata":Metadata(
                source=filename,
                ext=ext,
                language=language,
                chars=count_chars)
        }