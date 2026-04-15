"""
chat.py
-------
Chat endpoint with async RAG evaluation (RAGAS).
"""

from fastapi import APIRouter, Depends, BackgroundTasks
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
    get_owner_id,
)
from app.services.ragas_service import run_ragas_eval
import time

router = APIRouter()

@router.post("/{session_id}/chat")
async def chat(
    session_id: str,
    request:ChatRequest,
    background_tasks: BackgroundTasks,
    manager: SessionManager = Depends(get_session_manager),
    owner_id: str = Depends(get_owner_id),
):
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

    # RAGAS sample
    ragas_sample = {
        # "session_id":session_id ,
        # "owner_id":owner_id ,
        "query": request.question,
        "response_text": answer,
        "contexts": [c["content"] for c in reranked],
    }

    # background evaluation
    background_tasks.add_task(run_ragas_eval, ragas_sample,session_id,owner_id)

    # Step 10: return response
    return {
        "answer":        answer,
        "sources":       reranked,
        "strategy_used": strategy,
        "retrieval_ms":  retrieval_ms,
        "generate_ms":   generate_ms,
    }