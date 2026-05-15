from backend.core import get_logger
from backend.models.metadata import Metadata, ChunkMetadata
from ..base_chunker import BaseChunker
from typing import Generator,Dict

logger = get_logger(__name__)


class RecursiveChunker(BaseChunker):

    def __init__(self):
        super().__init__()

        # Ordered from coarsest to finest — the splitter tries each in order.
        # Semantic boundaries first → better context preservation for RAG.
        self.separators = [
            "\n\n",   # paragraph break  — strongest semantic boundary
            "\n",     # line break
            ". ",     # sentence end
            "! ",     # exclamation sentence
            "? ",     # question sentence
            "; ",     # clause boundary
            ", ",     # sub-clause boundary
            " ",      # word boundary — last resort
        ]

    def _recursive_split(self, text: str, sep_index: int = 0) -> list[str]:
        """
        Split *text* using separators[sep_index].
        If a part is still too long, recurse with the *next* separator
        instead of restarting from index 0 — this is the key correctness fix.
        """
        if len(text) <= self.chunk_size or sep_index >= len(self.separators):
            # Base case: fits in one chunk, or no more separators to try.
            return [text]

        separator = self.separators[sep_index]

        if separator not in text:
            # This separator doesn't appear — move to the next one directly.
            return self._recursive_split(text, sep_index + 1)

        pieces: list[str] = []
        for part in text.split(separator):
            part = part.strip()
            if not part:
                continue
            if len(part) <= self.chunk_size:
                pieces.append(part)
            else:
                # Still too long → try the *next* separator (not from 0).
                pieces.extend(self._recursive_split(part, sep_index + 1))

        return pieces

    def _clean_overlap(self, text: str) -> str:
        """
        Return the last *chunk_overlap* characters of *text*,
        trimmed to start at a word boundary so we never cut mid-word.
        """
        if self.chunk_overlap <= 0 or len(text) <= self.chunk_overlap:
            return text

        overlap = text[-self.chunk_overlap:]
        space_idx = overlap.find(" ")
        # If there's a space inside the overlap window, start after it.
        if space_idx != -1:
            return overlap[space_idx + 1:]
        return overlap  # no space found — return as-is (very rare)

    def _merge_with_overlap(self, pieces: list[str]) -> list[str]:
        """
        Greedily merge pieces into chunks ≤ chunk_size chars.
        When a chunk is full, carry the tail (clean overlap) into the next one
        so retrieval context is never lost at chunk boundaries.
        """
        chunks: list[str] = []
        current = ""

        for piece in pieces:
            would_be = len(current) + 1 + len(piece)  # +1 for the space
            if would_be > self.chunk_size and current:
                chunks.append(current.strip())
                # Start the next chunk with an overlapping tail.
                current = self._clean_overlap(current) + " " + piece
            else:
                current = current + " " + piece if current else piece

        if current.strip():
            chunks.append(current.strip())

        return chunks

    def chunk(
        self,
        text: str,
        metadata: Metadata,
    ) -> Generator[Dict, None, None]:
        
        pieces = self._recursive_split(text)
        raw_chunks = self._merge_with_overlap(pieces)

        idx = 0
        for raw_chunk in raw_chunks:
            raw_chunk = raw_chunk.strip()

            # Drop chunks that are too short to be useful for retrieval.
            if len(raw_chunk) < self.min_chunk_len:
                continue

            yield {
                "text" :raw_chunk ,
                "metadata":ChunkMetadata(
                # ── inherited from document metadata ──────────────────
                source=metadata.source,
                ext=metadata.ext,
                language=metadata.language,
                chars=metadata.chars,
                pages=metadata.pages,
                document_id=metadata.document_id,
                # ── chunk-specific ────────────────────────────────────
                chunk_index=idx,
                chunk_size=len(raw_chunk),
                word_count=len(raw_chunk.split()),
                chunker = 'recursive'
            )}

            idx += 1

        logger.info(
            "Chunked '%s' → %d chunks (size=%d, overlap=%d)",
            metadata.source,
            idx,
            self.chunk_size,
            self.chunk_overlap,
        )