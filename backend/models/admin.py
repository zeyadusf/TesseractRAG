from __future__ import annotations

from uuid import UUID
from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from backend.models.enums.doc_status import DocumentStatus 
from backend.models.auth import UserOut
from backend.models.enums.message_role import MessageRole


# ── Evaluation scores (reused in multiple responses) ──────────────────────────

class AvgScores(BaseModel):
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None


# ── Dashboard ─────────────────────────────────────────────────────────────────

class AdminDashboardOut(BaseModel):
    total_users: int
    total_superusers: int          # ← added
    total_sessions: int
    total_messages: int
    total_documents: int
    global_avg_scores: AvgScores


# ── User detail ───────────────────────────────────────────────────────────────

class UserAdminOut(UserOut):
    session_count: int = 0
    message_count: int = 0
    document_count: int = 0
    avg_scores: AvgScores = AvgScores()

    model_config = {"from_attributes": True}

    @classmethod
    def from_stats_dict(cls, data: dict) -> "UserAdminOut":
        user_orm = data["user"]
        return cls(
            id=user_orm.id,
            email=user_orm.email,
            username=user_orm.username,
            is_active=user_orm.is_active,
            is_superuser=user_orm.is_superuser,
            session_count=data.get("session_count", 0),
            message_count=data.get("message_count", 0),
            document_count=data.get("document_count", 0),
            avg_scores=AvgScores(
                faithfulness=data.get("avg_faithfulness"),
                answer_relevancy=data.get("avg_answer_relevancy"),
                context_precision=data.get("avg_context_precision"),
                context_recall=data.get("avg_context_recall"),
            ),
        )


# ── Session list ──────────────────────────────────────────────────────────────





class DocumentBriefOut(BaseModel):
    """Lightweight document summary embedded inside SessionAdminOut."""
    id: UUID
    filename: str
    status: DocumentStatus

    model_config = {"from_attributes": True}


class SessionAdminOut(BaseModel):
    id: UUID
    user_id: UUID
    name: str
    description: Optional[str]
    is_active: bool
    created_at: datetime
    document_count: int = 0
    message_count: int = 0
    documents: list[DocumentBriefOut] = []

    model_config = {"from_attributes": True}

    @classmethod
    def from_row(cls, data: dict) -> "SessionAdminOut":
        """
        data keys:
            session        → Session ORM object
            document_count → int
            message_count  → int
            documents      → list[dict] with id, filename, status
        """
        session = data["session"]
        return cls(
            id=session.id,
            user_id=session.user_id,
            name=session.name,
            description=session.description,
            is_active=session.is_active,
            created_at=session.created_at,
            document_count=data["document_count"],
            message_count=data["message_count"],
            documents=[
                DocumentBriefOut(
                    id=doc["id"],
                    filename=doc["filename"],
                    status=doc["status"],
                )
                for doc in data.get("documents", [])
            ],
        )


# ── Message list ──────────────────────────────────────────────────────────────

class MessageEvalOut(BaseModel):
    """Evaluation scores attached to a single message. All optional."""
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None


class MessageAdminOut(BaseModel):
    id: UUID
    session_id: UUID
    user_id: UUID
    role: MessageRole
    content: str
    created_at: datetime
    evaluation: MessageEvalOut = MessageEvalOut()

    model_config = {"from_attributes": True}

    @classmethod
    def from_row(cls, row) -> "MessageAdminOut":
        msg = row[0]
        return cls(
            id=msg.id,
            session_id=msg.session_id,
            user_id=row.user_id,
            role=msg.role,
            content=msg.content,
            created_at=msg.created_at,
            evaluation=MessageEvalOut(
                faithfulness=_f(row.faithfulness),
                answer_relevancy=_f(row.answer_relevancy),
                context_precision=_f(row.context_precision),
                context_recall=_f(row.context_recall),
            ),
        )


# ── Control responses ─────────────────────────────────────────────────────────

class ToggleStatusOut(BaseModel):
    id: UUID
    is_active: bool
    detail: str


# ── helpers ───────────────────────────────────────────────────────────────────

def _f(val) -> Optional[float]:
    return float(val) if val is not None else None