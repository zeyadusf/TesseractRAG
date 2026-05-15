import fitz  
from pathlib import Path
from typing import Generator, Dict, Any
from backend.models.metadata import Metadata

from ...lang_detector import get_language_or_fallback


class PdfParser:

    def parse(self, file_bytes: bytes, filename: str) -> Generator[Dict[str, Any], None, None]:

        doc = fitz.open(stream=file_bytes, filetype="pdf")

        texts = []
        total_pages = doc.page_count

        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            text = page.get_text("text") or ""

            if text.strip():
                texts.append(f"[Page {page_num + 1}]\n{text}")

        raw_text = "\n\n".join(texts)

        doc.close()

        language = get_language_or_fallback(raw_text[:500])
        count_chars = len(raw_text)
        ext =  Path(filename).suffix.lower()

        yield {
            "text": raw_text,
            "metadata":Metadata(
                source=filename,
                ext=ext,
                language=language,
                chars=count_chars,
                pages=total_pages)
        }