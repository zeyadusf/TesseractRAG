"""
chat.py
-------
Chat / question-answering endpoint for TesseractRAG API v1.

Requires X-Owner-ID header on every request. The owner_id is checked
before any retrieval or generation work begins — an unauthorized request
is rejected immediately at the session lookup stage.
"""

from fastapi import APIRouter, Depends, HTTPException
from app.models.chat import ChatRequest, ChatResponse
from app.core.session_manager import SessionManager
from app.core.retrieval.router import RetrievalRouter
from app.core.retrieval.hybrid_retriever import HybridRetriever
from app.core.generation.context_builder import ContextBuilder
from app.core.generation.prompt_builder import PromptBuilder
from app.dependencies import (
    get_session_manager,
    get_embedder,
    get_reranker,
    get_llm_client,
    get_owner_id,                                       # ← NEW
)
import time

router = APIRouter()


@router.post("/{session_id}/chat", response_model=ChatResponse)
async def chat(
    session_id: str,
    request: ChatRequest,
    manager: SessionManager = Depends(get_session_manager),
    owner_id: str = Depends(get_owner_id),              # ← NEW
):
    """
    Submit a question to a session's knowledge base and receive a grounded answer.

    Ownership is checked at Step 1 — if the requesting browser does not own
    this session, the request is rejected with 403 before any ML compute runs.

    Pipeline:
        1. Ownership check + session load
        2. Query routing → optimal retrieval strategy
        3. Guard — no documents uploaded yet
        4. Hybrid retrieval (BM25 + FAISS)
        5. Cross-encoder reranking
        6. Context building (dedup + format)
        7. Prompt construction (system + history + context + question)
        8. LLM generation (Mistral-7B via HF API)
        9. Persist both turns to R2
        10. Return answer + sources + timing metrics
    """
    # Step 1: load session — get_session with owner_id enforces ownership
    session = manager.get_session(session_id, owner_id=owner_id)   # ← NEW

    # Step 2: route the query to optimal retrieval strategy
    strategy = RetrievalRouter().route(request.question, request.strategy)

    # Step 3: guard — no documents uploaded yet
    if not session.chunks:
        return {
            "answer":        (
                "No documents have been uploaded to this session yet. "
                "Please upload a document before asking questions."
            ),
            "sources":       [],
            "strategy_used": strategy,
            "retrieval_ms":  0,
            "generate_ms":   0,
        }

    # Step 4: retrieve chunks
    start = time.time()
    embedder = get_embedder()
    chunks = HybridRetriever(
        session.bm25_retriever,
        session.faiss_index,
        embedder,
        session.chunks,
    ).retrieve(request.question, strategy)

    # Step 5: rerank chunks
    reranked = get_reranker().rerank(request.question, chunks)
    retrieval_ms = int((time.time() - start) * 1000)

    # Step 6: build context string
    context = ContextBuilder().build(chunks=reranked)

    # Step 7: build prompt — last 3 exchanges only to stay within context window
    history = session.messages[-6:]
    messages = PromptBuilder().build(
        request.question,
        context=context,
        history=history,
    )

    # Step 8: generate answer
    gen_start = time.time()
    answer = await get_llm_client().generate(messages)
    generate_ms = int((time.time() - gen_start) * 1000)

    # Step 9: persist both turns to R2 — ownership already verified above
    manager.add_message(session_id, role="user",      content=request.question, owner_id=owner_id)  # ← NEW
    manager.add_message(session_id, role="assistant", content=answer,            owner_id=owner_id)  # ← NEW

    # Step 10: return response
    return {
        "answer":        answer,
        "sources":       reranked,
        "strategy_used": strategy,
        "retrieval_ms":  retrieval_ms,
        "generate_ms":   generate_ms,
    }
