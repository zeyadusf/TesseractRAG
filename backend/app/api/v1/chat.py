from fastapi import APIRouter, Depends
from app.models.chat import ChatRequest,ChatResponse
from app.core.session_manager import SessionManager
# 
from app.core.retrieval.router import RetrievalRouter
from app.core.retrieval.hybrid_retriever import HybridRetriever
# 
from app.core.generation.context_builder import ContextBuilder
from app.core.generation.prompt_builder import  PromptBuilder
# 
from app.dependencies import get_session_manager, get_embedder, get_reranker, get_llm_client
from fastapi import HTTPException
import time

router = APIRouter()

@router.post("/sessions/{session_id}/chat", response_model=ChatResponse)
async def chat(session_id: str, request: ChatRequest, manager: SessionManager = Depends(get_session_manager)):

    # Step 1: get session, raise 404 if not found
    session = manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Step 2: route the query
    strategy = RetrievalRouter().route(request.question,request.strategy)
    start = time.time()
    # Step 3: retrieve chunks using HybridRetriever
    embedder = get_embedder()
    chunks = HybridRetriever(session.bm25_retriever,session.faiss_index,embedder,
                            session.chunks).retrieve(request.question,strategy)

    # Step 4: rerank chunks
    reranked = get_reranker().rerank(request.question,chunks)
    retrieval_ms = int((time.time() - start) * 1000)

    # Step 5: build context string
    context = ContextBuilder().build(chunks=reranked)

    # Step 6: build messages list
    messages = PromptBuilder().build(request.question,context=context,history=session.messages)

    # Step 7: generate answer
    gen_start = time.time()
    answer = await get_llm_client().generate(messages)
    generate_ms = int((time.time() - gen_start) * 1000)
    # Step 8: save question and answer to session.messages
    session.messages.append({"role":'user',"content" : request.question})
    session.messages.append({"role":'assistant',"content" : answer})

    # Step 9: return ChatResponse
    return {
        "answer" : answer,
        "sources" : reranked,
        "strategy_used" : strategy,
        "retrieval_ms" : retrieval_ms,
        "generate_ms" : generate_ms
    }
