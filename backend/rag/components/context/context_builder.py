from __future__ import annotations

import hashlib
from typing import Optional

from backend.core import get_logger, get_config

logger = get_logger(__name__)


class ContextBuilder:
    """
    Builds a formatted context string from retrieved chunks for RAG generation.

    Features:
    - Deduplication via fast ID-based hashing (fallback to MD5)
    - Citation-ready formatting: [Source N]
    - Smart truncation per chunk to respect MAX_CONTEXT_CHARS
    - Metadata enrichment (relevance score, page number if available)
    """

    def __init__(self, max_chars: int | None = None):
        config = get_config()
        self.max_chars = max_chars or config.MAX_CONTEXT_CHARS
        self.max_chunk_chars = getattr(config, 'MAX_CHUNK_CHARS', 1000)

    def _get_chunk_id(self, chunk: dict) -> str:
        """
        Generate a fast, unique ID for deduplication.
        Fallback to MD5 only if metadata is missing.
        """
        doc_name = chunk.get('document_name', '')
        chunk_idx = chunk.get('chunk_index', 0)
        if doc_name and chunk_idx is not None:
            return f"{doc_name}:{chunk_idx}"

        content = chunk.get('content', '')
        return hashlib.md5(content.encode()).hexdigest()

    def _truncate_chunk(self, text: str, max_len: int) -> tuple[str, bool]:
        """
        Truncate text to max_len chars with ellipsis.
        Returns: (truncated_text, was_truncated)
        """
        if len(text) <= max_len:
            return text, False
        keep_start = max_len * 2 // 3
        keep_end = max_len - keep_start - 3
        return f"{text[:keep_start]}...{text[-keep_end:]}", True

    def _format_chunk(self, chunk: dict, source_num: int) -> str:
        """Format a single chunk with citation-ready header."""
        content = chunk.get('content', '').strip()
        doc_name = chunk.get('document_name', 'Unknown')
        chunk_idx = chunk.get('chunk_index', 0)

        metadata_parts = []
        if 'page' in chunk:
            metadata_parts.append(f"Page {chunk['page']}")
        if 'section' in chunk:
            metadata_parts.append(f"Section: {chunk['section']}")
        if 'score' in chunk:
            metadata_parts.append(f"Score: {chunk['score']:.2f}")

        meta_str = f" ({', '.join(metadata_parts)})" if metadata_parts else ""
        header = f"[Source {source_num}] {doc_name}, Chunk {chunk_idx}{meta_str}:"
        return f"{header}\n{content}"

    def build(self, chunks: list[dict]) -> tuple[str, dict]:
        """
        Build formatted context string from retrieved chunks.

        Args:
            chunks: List of dicts with keys: content, document_name, chunk_index,
                   [optional: score, page, section]

        Returns:
            tuple: (formatted_context_string, metadata_dict)
        """
        seen_ids = set()
        context_parts = []
        total_chars = 0
        stats = {
            'chunks_used': 0,
            'chunks_deduped': 0,
            'chunks_truncated': 0,
            'sources': []
        }

        for chunk in chunks:
            content = chunk.get('content', '')
            if not content or not content.strip():
                continue

            # Deduplication
            chunk_id = self._get_chunk_id(chunk)
            if chunk_id in seen_ids:
                stats['chunks_deduped'] += 1
                continue
            seen_ids.add(chunk_id)

            # Truncation per chunk
            truncated, was_truncated = self._truncate_chunk(content, self.max_chunk_chars)
            if was_truncated:
                stats['chunks_truncated'] += 1

            # Check global limit
            if total_chars + len(truncated) > self.max_chars:
                logger.debug(f"Context limit reached ({total_chars}/{self.max_chars} chars). Stopping.")
                break

            # Format with citation-ready header
            source_num = stats['chunks_used'] + 1
            formatted = self._format_chunk(chunk, source_num)

            context_parts.append(formatted)
            total_chars += len(formatted) + 2  # +2 for "\n\n" separator

            # Track source metadata for citation mapping
            stats['sources'].append({
                'id': source_num,
                'document_name': chunk.get('document_name'),
                'chunk_index': chunk.get('chunk_index'),
                'page': chunk.get('page'),
                'score': chunk.get('score')
            })
            stats['chunks_used'] += 1

        context_str = "\n\n".join(context_parts)
        stats['total_chars'] = total_chars

        logger.info(
            f"Context built: {stats['chunks_used']} chunks, {total_chars} chars, "
            f"{stats['chunks_deduped']} deduped, {stats['chunks_truncated']} truncated"
        )

        return context_str, stats