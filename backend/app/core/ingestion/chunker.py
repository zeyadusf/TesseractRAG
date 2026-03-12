from app.utils.logger import get_logger

logger = get_logger(__name__)

class RecursiveChunker:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_len: int = 50
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_len = min_chunk_len
        self.separators = ["\n\n", "\n", ". ", " "]

    def _recursive_split(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size : return [text]
        pieces = []
        
        # Try each separator in order
        for separator in self.separators:
            if separator in text:
                parts = text.split(separator)
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue
                    if len(part) <= self.chunk_size:
                        pieces.append(part)
                    else:
                        # Still too long — recurse with remaining separators
                        pieces.extend(self._recursive_split(part))
                return pieces
        
        # No separator worked — just return the text as-is
        pieces.append(text)
        return pieces

    def _merge_with_overlap(self, pieces: list[str]) -> list[str]:
        chunks = []
        current = ""
        
        for piece in pieces:
            # If adding this piece exceeds chunk_size — save current and start new
            if len(current) + len(piece) > self.chunk_size and current:
                chunks.append(current.strip())
                # Take last chunk_overlap characters as the start of next chunk
                current = current[-self.chunk_overlap:] + " " + piece
            else:
                current = current + " " + piece if current else piece
        
        # Don't forget the last chunk
        if current.strip():
            chunks.append(current.strip())
        
        return chunks

    def split(self, text: str, document_name: str) -> list[dict]:
        """
        Full pipeline: split text recursively, merge with overlap, and return metadata.
        """
        pieces = self._recursive_split(text)
        chunks = self._merge_with_overlap(pieces)

        result = []
        for idx, chunk in enumerate(chunks):
            word_count = len(chunk.split())
            if len(chunk) < self.min_chunk_len:
                continue
            result.append({
                "content": chunk,
                "document_name": document_name,
                "chunk_index": idx,
                "word_count": word_count
            })
        logger.info(f"Split '{document_name}' into {len(result)} chunks")
        return result


if __name__ == '__main__':
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20, min_chunk_len=10)
    
    text = """Retrieval Augmented Generation is a technique that grounds LLM responses in retrieved documents.

It combines a retrieval system with a language model. The retrieval system finds relevant passages.
The language model generates an answer based on those passages.

This reduces hallucination significantly. The model can only use what was retrieved."""

    chunks = chunker.split(text, "test_document.txt")
    print(f"Total chunks: {len(chunks)}")
    for c in chunks:
        print(f"  [{c['chunk_index']}] words={c['word_count']} | {c['content'][:60]}...")