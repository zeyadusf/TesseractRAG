
from __future__ import annotations
from sqlalchemy.ext.asyncio import AsyncSession

def init_postgres_repos(session: AsyncSession):
    from backend.storage.db.postgres.repositories.user_repo import UserRepository
    from backend.storage.db.postgres.repositories.session_repo import SessionRepository
    from backend.storage.db.postgres.repositories.document_repo import DocumentRepository
    from backend.storage.db.postgres.repositories.chunk_repo import ChunkRepository
    from backend.storage.db.postgres.repositories.embedding_repo import EmbeddingRepository
    from backend.storage.db.postgres.repositories.message_repo import MessageRepository
    from backend.storage.db.postgres.repositories.evaluation_repo import EvaluationRepository

    return {
        "users": UserRepository(session),
        "sessions": SessionRepository(session),
        "documents": DocumentRepository(session),
        "chunks": ChunkRepository(session),
        "embeddings": EmbeddingRepository(session),
        "messages": MessageRepository(session),
        "evaluations": EvaluationRepository(session),
    }