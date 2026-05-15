"""
Test script for GenerationPipeline
Run: python -m backend.testing.test_generation_pipeline
"""

from __future__ import annotations

import asyncio
import time

from backend.rag.pipelines.generation_pipeline import GenerationPipeline, GenerationInput,reset_generation_pipeline_cache

# ── Mock data for testing ─────────────────────────────────────────────────────
MOCK_CHUNKS = [
    {
        "chunk_id": "doc1_chunk1",
        "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
        "document_name": "AI_Guide.pdf",
        "chunk_index": 1,
        "score": 0.95,
        "page": 12,
    },
    {
        "chunk_id": "doc1_chunk2",
        "content": "Deep learning is a type of machine learning that uses neural networks with many layers (deep neural networks) to analyze various factors of data.",
        "document_name": "AI_Guide.pdf",
        "chunk_index": 2,
        "score": 0.88,
        "page": 15,
    },
    {
        "chunk_id": "doc2_chunk1",
        "content": "Natural language processing (NLP) is a branch of AI that helps computers understand, interpret and manipulate human language.",
        "document_name": "NLP_Basics.md",
        "chunk_index": 1,
        "score": 0.82,
    },
]


async def test_pipeline_basic():
    """Test basic pipeline flow with mock data."""
    print("\n🚀 Testing GenerationPipeline (Basic)...\n")
    
    pipeline = GenerationPipeline(enable_rewrite=True)
    
    input_data: GenerationInput = {
        "original_query": "What is machine learning?",
        "query_vector": None,
        "chunks": MOCK_CHUNKS,
        "vector_results": None,
        "strategy": "auto",
    }
    
    start = time.perf_counter()
    result = await pipeline.run(input_data)
    elapsed = time.perf_counter() - start
    
    # ── Assertions ───────────────────────────────────────────────────────────
    assert "answer" in result, "Missing 'answer' in output"
    assert isinstance(result["answer"], str), "Answer should be string"
    assert len(result["answer"].strip()) > 0, "Answer should not be empty"
    assert "sources" in result, "Missing 'sources' in output"
    assert "timing" in result, "Missing 'timing' in output"
    
    # ── Output ───────────────────────────────────────────────────────────────
    print(f"✅ Answer ({len(result['answer'])} chars):")
    print(f"   {result['answer'][:200]}{'...' if len(result['answer']) > 200 else ''}")
    print(f"\n📊 Sources: {len(result['sources'])} chunks used")
    for src in result['sources'][:3]:  # Show first 3
        print(f"   • [Source {src['id']}] {src['document_name']} (score: {src.get('score', 'N/A')})")
    print(f"\n⏱️  Timing: {result['timing']}")
    print(f"\n✨ Total time: {elapsed*1000:.0f}ms")
    
    # await pipeline.close()
    print("\n✅ Basic test passed!\n")


async def test_pipeline_simple():
    """Test the convenience run_simple() method."""
    print("🚀 Testing GenerationPipeline.run_simple()...\n")
    reset_generation_pipeline_cache()
    pipeline = GenerationPipeline(enable_rewrite=False)  # Disable rewrite for speed
    
    answer = await pipeline.run_simple(
        query="Explain deep learning",
        chunks=MOCK_CHUNKS
    )
    
    assert isinstance(answer, str), "run_simple should return string"
    assert len(answer.strip()) > 0, "Answer should not be empty"
    
    print(f"✅ Answer: {answer[:150]}{'...' if len(answer) > 150 else ''}\n")
    
    await pipeline.close()
    print("✅ Simple test passed!\n")


async def test_pipeline_with_fallback():
    """Test pipeline behavior when components fail (mocked)."""
    print("🚀 Testing GenerationPipeline fallback behavior...\n")
    
    # Test with empty chunks (should trigger fallback message)
    pipeline = GenerationPipeline(enable_rewrite=False)
    
    result = await pipeline.run_simple(
        query="What is AI?",
        chunks=[]  # Empty → should return fallback message
    )
    
    # Should return graceful fallback, not crash
    assert "error" in result.lower() or "sorry" in result.lower() or "information" in result.lower(), \
        f"Expected fallback message, got: {result}"
    
    print(f"✅ Fallback message: {result}\n")
    
    await pipeline.close()
    print("✅ Fallback test passed!\n")

# backend/testing/test_identity_prompt.py
import asyncio

# ── Imports for reset functions ──────────────────────────────────────────────
from backend.rag.components.generator.groq_generator import (
    get_groq_generator,
    reset_groq_generator_cache,  # ✅ دي المهمة
)
from backend.rag.components.query_rewrite.groq_query_rewriter import reset_groq_rewriter_cache

async def test_identity_response():
    # ✅ ريست الكاش قبل ما تبدأ عشان تاخد instance جديد بـ client مفتوح
    reset_groq_generator_cache()
    reset_groq_rewriter_cache()
    
    generator = get_groq_generator()
    
    # Test 1: Identity question
    answer1 = await generator.generate(
        question="Who are you?",
        context="Some dummy context here."
    )
    print(f"🤖 Q: Who are you?\n   A: {answer1}\n")
    
    # Test 2: Content question
    answer2 = await generator.generate(
        question="What is machine learning?",
        context="Machine learning is a subset of AI that enables systems to learn from data."
    )
    print(f"🤖 Q: What is machine learning?\n   A: {answer2}\n")
    
    # Test 3: Arabic identity question
    answer3 = await generator.generate(
        question="إنت إيه؟ وإيه المشروع ده؟",
        context="محتوى تجريبي"
    )
    print(f"🤖 س: إنت إيه؟\n   ج: {answer3}\n")
    
    # ⚠️ في التستات: ماتقفلش الـ generator عشان الـ singleton يفضل شغال
    # await generator.close()
    
    print("✅ Identity test completed!")


async def main():
    """Run all tests."""
    print(f"\n{'='*60}")
    print("  GenerationPipeline — Test Suite")
    print(f"{'='*60}\n")
    
    try:
        await test_pipeline_basic()
        await test_pipeline_simple()
        await test_pipeline_with_fallback()
        await test_identity_response()
        
        print(f"{'='*60}")
        print("  ✅ All tests passed!")
        print(f"{'='*60}\n")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}\n")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}\n")
        raise


if __name__ == "__main__":
    asyncio.run(main())

