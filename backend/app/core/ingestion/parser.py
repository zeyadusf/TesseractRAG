import io
from PyPDF2 import PdfReader
from pathlib import Path

class DocumentParser:
    def parse(self, file_bytes: bytes, filename: str) -> str:
        # Detect file type from filename extension
        ext = Path(filename).suffix.lower()
        allowed = [".pdf", ".md", ".txt"]
        if ext not in allowed:
            raise ValueError("Unsupported file type, Only[pdf, md, txt]")
        
        # .txt or .md  → decode bytes as UTF-8 text
        elif ext in [".md", ".txt"]:
            return self._parse_text(file_bytes)
        # .pdf         → extract text using PyPDF2
        elif ext == '.pdf':
            return self._parse_pdf(file_bytes)


    def _parse_text(self, file_bytes: bytes) -> str:
        text = file_bytes.decode("utf-8", errors="ignore")
        return " ".join(text.split())

    def _parse_pdf(self, file_bytes: bytes) -> str:
        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return " ".join(text.split())


if __name__ == '__main__':
    parser = DocumentParser()
    
    # Test TXT
    txt_bytes = b"Hello world.  This is a   test document.\n\nWith multiple lines."
    result = parser.parse(txt_bytes, "test.txt")
    print(f"TXT: {result}")
    
    # Test unsupported type
    try:
        parser.parse(b"data", "file.docx")
    except ValueError as e:
        print(f"Expected error: {e}")