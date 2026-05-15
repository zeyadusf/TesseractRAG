from __future__  import annotations
from .base_repo import BaseRepository
from backend.storage.db.postgres.schemas import Embedding
from typing import List
from sqlalchemy import select,delete,text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession


class EmbeddingRepository(BaseRepository[Embedding]):
    def __init__(self,sessionConn:AsyncSession):
        self._s = sessionConn

    # ── CRUD ───────────────────────────────────────────────────────────────

    async def create(
        self,
        *,
        chunk_id: UUID,
        session_id: UUID,
        model_name: str,
        dimensions: int,
        vector: List[float],
        token_usage: int | None = None,
    ) -> Embedding:
        obj = Embedding(
            chunk_id=chunk_id,
            session_id=session_id,
            model_name=model_name,
            dimensions=dimensions,
            vector=vector,
            token_usage=token_usage,
        )
        self._s.add(obj)
        await self._s.flush()
        await self._s.refresh(obj)
        return obj

    async def bulk_create(self, embeddings: List[dict]) -> List[Embedding]:
        """
        Insert multiple embeddings in a single flush.
        Always use this during ingestion — never loop over create().

        Example embedding dict:
        {
            "chunk_id":    UUID(...),
            "session_id":  UUID(...),
            "model_name":  "jina-embeddings-v3",
            "dimensions":  1024,
            "vector":      [0.12, -0.34, ...],   # list of 512 floats
            "token_usage": 91,
        }
        """
        if not embeddings:
            return []
        objs = [Embedding(**e) for e in embeddings]
        self._s.add_all(objs)
        await self._s.flush()
        return objs

    async def get_by_id(self, record_id: UUID) -> Embedding | None:
        result = await self._s.execute(
            select(Embedding).where(Embedding.id == record_id)
        )
        return result.scalar_one_or_none()

    async def get_by_chunk_and_model(
        self, chunk_id: UUID, model_name: str
    ) -> Embedding | None:
        """Check if an embedding already exists for this chunk + model combination."""
        result = await self._s.execute(
            select(Embedding).where(
                Embedding.chunk_id == chunk_id,
                Embedding.model_name == model_name,
            )
        )
        return result.scalar_one_or_none()

    async def list_by_session(
        self, session_id: UUID, model_name: str | None = None
    ) -> List[Embedding]:
        stmt = select(Embedding).where(Embedding.session_id == session_id)
        if model_name:
            stmt = stmt.where(Embedding.model_name == model_name)
        result = await self._s.execute(stmt)
        return list(result.scalars().all())

    async def upsert_by_chunk(
        self,
        *,
        chunk_id: UUID,
        session_id: UUID,
        model_name: str,
        dimensions: int,
        vector: List[float],
        token_usage: int | None = None,
    ) -> Embedding:
        """
        Insert or update the embedding for a chunk+model pair.
        Used when re-indexing an already-embedded document.
        """
        existing = await self.get_by_chunk_and_model(chunk_id, model_name)
        if existing:
            existing.vector = vector
            existing.token_usage = token_usage
            await self._s.flush()
            await self._s.refresh(existing)
            return existing
        
        return await self.create(
            chunk_id=chunk_id,
            session_id=session_id,
            model_name=model_name,
            dimensions=dimensions,
            vector=vector,
            token_usage=token_usage,
        )

    async def update(self, record_id: UUID, **kwargs) -> Embedding | None:
        obj = await self.get_by_id(record_id)
        if obj is None:
            return None
        for field, value in kwargs.items():
            setattr(obj, field, value)
        await self._s.flush()
        await self._s.refresh(obj)
        return obj

    async def delete(self, record_id: UUID) -> bool:
        obj = await self.get_by_id(record_id)
        if obj is None:
            return False
        await self._s.delete(obj)
        await self._s.flush()
        return True

    async def delete_by_chunk(self, chunk_id: UUID) -> int:
        result = await self._s.execute(
            delete(Embedding)
            .where(Embedding.chunk_id == chunk_id)
            .returning(Embedding.id)
        )
        await self._s.flush()
        return len(result.fetchall())

    async def delete_by_session(self, session_id: UUID) -> int:
        result = await self._s.execute(
            delete(Embedding)
            .where(Embedding.session_id == session_id)
            .returning(Embedding.id)
        )
        await self._s.flush()
        return len(result.fetchall())


    # ── Vector Search ──────────────────────────────────────────────────────

    async def similarity_search(
        self,
        *,
        query_vector: List[float],
        session_id: UUID,
        model_name: str,
        top_k: int = 10,
    ) -> List[dict]:
        """
        Cosine similarity ANN search scoped to a single session.

        HOW IT WORKS:
        1. PostgreSQL uses the B-tree index on (session_id, model_name)
            to filter the candidate rows before touching the HNSW index.
        2. pgvector's HNSW index then runs approximate nearest-neighbour
            search on that filtered subset using the <=> (cosine distance) operator.
        3. We convert distance → similarity score: score = 1 - distance.
            Score range: 0.0 (unrelated) → 1.0 (identical).

        Returns:
            List of dicts ordered by relevance descending:
            [{"chunk_id": UUID, "score": float}, ...]
        """
        stmt = text(
            """
            SELECT
                e.chunk_id,
                1 - (e.vector <=> CAST(:query AS vector)) AS score
            FROM embeddings e
            WHERE
                e.session_id = :session_id
                AND e.model_name = :model_name
            ORDER BY e.vector <=> CAST(:query AS vector)
            LIMIT :top_k
            """
        )
        result = await self._s.execute(
            stmt,
            {
                "query":      str(query_vector),
                "session_id": str(session_id),
                "model_name": model_name,
                "top_k":      top_k,
            },
        )
        rows = result.fetchall()

        return [
            {"chunk_id": str(row.chunk_id), "score": float(row.score)}
            for row in rows
        ]

    async def similarity_search_with_filter(
        self,
        *,
        query_vector: list[float],
        session_id: UUID,
        model_name: str,
        top_k: int = 10,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        """
        Cosine similarity search with optional JSONB metadata pre-filter.

        metadata_filter examples:
            {"language": "ar"}          → Arabic chunks only
            {"file_extension": "pdf"}   → PDF chunks only

        The metadata filter narrows the candidate set BEFORE the ANN scan,
        which improves both accuracy and performance for filtered queries.

        Returns same shape as similarity_search().
        """
        # Build the JSONB filter clause dynamically
        filter_clause = ""
        params: dict = {
            "query":      str(query_vector),
            "session_id": str(session_id),
            "model_name": model_name,
            "top_k":      top_k,
        }

        if metadata_filter:
            conditions = []
            for i, (key, val) in enumerate(metadata_filter.items()):
                param_key = f"meta_val_{i}"
                conditions.append(
                    f"c.metadata @> jsonb_build_object('{key}', :{param_key})"
                )
                params[param_key] = val
            filter_clause = "AND " + " AND ".join(conditions)

        stmt = text(
            f"""
            SELECT
                e.chunk_id,
                1 - (e.vector <=> CAST(:query AS vector)) AS score
            FROM embeddings e
            JOIN chunks c ON c.id = e.chunk_id
            WHERE
                e.session_id = :session_id
                AND e.model_name = :model_name
                {filter_clause}
            ORDER BY e.vector <=> CAST(:query AS vector)
            LIMIT :top_k
            """
        )
        result = await self._s.execute(stmt, params)
        rows = result.fetchall()
        return [
            {"chunk_id": str(row.chunk_id), "score": float(row.score)}
            for row in rows
        ]

