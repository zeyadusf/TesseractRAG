from backend.rag.components.parsing.base_parser import BaseParser
from .parsers.DocxParser import DocxParser
from .parsers.PdfParser import PdfParser
from .parsers.TextParser import TextParser
from pathlib import Path
from backend.core import get_config, get_logger
from typing import Generator, Dict, Any, Type


class ParserDispatcher(BaseParser):

    def __init__(self):
        self.config = get_config()
        self.logger = get_logger(__name__)

        self.parsers: Dict[str, Type[BaseParser]] = {
            ".pdf": PdfParser,
            ".txt": TextParser,
            ".md": TextParser,
            ".docx": DocxParser,
        }

    def is_allowed_ext(self, filename: str) -> bool:
        return Path(filename).suffix.lower() in self.config.ALLOWED_EXTENSIONS

    def parse(
        self,
        file_bytes: bytes,
        filename: str
    ) -> Generator[Dict[str, Any], None, None]:

        ext = Path(filename).suffix.lower()

        if not self.is_allowed_ext(filename):
            self.logger.warning("Unsupported extension: %s", ext)
            raise ValueError(f"Extension '{ext}' not supported")

        parser_cls = self.parsers.get(ext)

        if not parser_cls:
            self.logger.critical("Parser missing for extension: %s", ext)
            raise ValueError(f"No parser registered for extension '{ext}'")

        parser_instance = parser_cls()

        yield from parser_instance.parse(file_bytes, filename)