from backend.rag.components.parsing.base_parser import BaseParser
from pathlib import Path
from typing import Generator, Dict, Any
from docx import Document
from backend.models.metadata import Metadata
import io

from ...lang_detector import get_language_or_fallback


class DocxParser(BaseParser):
    """Parser for DOCX files"""

    def parse(self, file_bytes: bytes, filename: str) -> Generator[Dict[str, Any], None, None]:

        doc = Document(io.BytesIO(file_bytes))

        paragraphs = []

        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                paragraphs.append(text)

        raw_text = "\n\n".join(paragraphs)

        language = get_language_or_fallback(raw_text)
        count_chars = len(raw_text)
        ext= Path(filename).suffix.lower()
        
        yield {
            "text": raw_text,
            "metadata":Metadata(
                source=filename,
                ext=ext,
                language=language,
                chars=count_chars)
            }