from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.core import get_config, get_logger
from backend.core.dependencies import get_embedder
from backend.models.embedding import EmbeddingChunk, EmbeddingMeta
from backend.rag.components.chunking.chunk_dispatcher import ChunkerDispatcher
from backend.rag.components.cleaning.clean_dispatcher import CleanerDispatcher
from backend.rag.components.parsing.parse_dispatcher import ParserDispatcher

logger = get_logger(__name__)


class IngestionPipeline:
    def __init__(self):
        self.config = get_config()
        self._parse = ParserDispatcher()
        self._clean = CleanerDispatcher()
        self._chunk = ChunkerDispatcher(chunker_technique=self.config.DEFAULT_CHUNK)
        self._embed = get_embedder()

    async def run(self, *, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        logger.info("[IngestionPipeline] Start processing '%s'...", filename)

        chunk_records: List[Dict[str, Any]] = []
        doc_metadata = None

        # [1] Parse: extract text and document metadata
        for parsed in self._parse.parse(file_bytes=file_bytes, filename=filename):
            raw_text: str = parsed["text"]
            doc_metadata = parsed["metadata"]

            # [2] Clean: normalize and sanitize text
            cleaned = self._clean.clean(text=raw_text, metadata=doc_metadata)
            clean_text: str = cleaned["text"]
            clean_metadata = cleaned.get("metadata", doc_metadata)

            # [3] Chunk: split text into manageable pieces
            for chunked in self._chunk.chunk(text=clean_text, metadata=clean_metadata):
                chunk_text: str = chunked["text"]
                chunk_meta = chunked.get("metadata", {})

                chunk_records.append(
                    {
                        "text": chunk_text,
                        "chunk_size": len(chunk_text),
                        "word_count": len(chunk_text.split()),
                        "chunk_metadata": chunk_meta,
                    }
                )

        if not chunk_records:
            logger.warning("[IngestionPipeline] No chunks produced for '%s'", filename)
            return {"chunks": [], "embeddings": [], "metadata": None}

        # [4] Embed: generate vector embeddings for each chunk
        texts: List[str] = [c["text"] for c in chunk_records]
        embeddings: List[Dict[str, Any]] = []
        embed_meta: Optional[EmbeddingMeta] = None

        async for item in self._embed.embed_documents(texts):
            if isinstance(item, EmbeddingChunk):
                embeddings.append(
                    {
                        "vector": item.embedding,  # Returns List[float] directly
                        "index": item.index,
                        "text": item.text,
                        "tokens": item.tokens,
                        "is_estimate": item.is_estimate,
                    }
                )
            elif isinstance(item, EmbeddingMeta):
                embed_meta = item

        total_tokens = getattr(embed_meta, "total_tokens", None)
        logger.info(
            "[IngestionPipeline] Done. chunks=%d tokens=%s filename='%s'",
            len(chunk_records),
            total_tokens,
            filename,
        )

        return {
            "chunks": chunk_records,
            "embeddings": embeddings,
            "metadata": {
                # Embedding-level info
                "model": getattr(embed_meta, "model", None),
                "total_tokens": total_tokens,
                "total_chunks": getattr(embed_meta, "total_chunks", None),
                "dimensions": getattr(embed_meta, "dimensions", None),
                # Document-level info (from parser metadata)
                "filename": getattr(doc_metadata, "source", None) if doc_metadata else None,
                "ext": getattr(doc_metadata, "ext", None) if doc_metadata else None,
                "language": getattr(doc_metadata, "language", None) if doc_metadata else None,
                "pages": getattr(doc_metadata, "pages", None) if doc_metadata else None,
                "chars": getattr(doc_metadata, "chars", None) if doc_metadata else None,
            }
            if embed_meta or doc_metadata
            else None,
        }