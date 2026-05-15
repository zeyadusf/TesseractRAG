"""
Embedding layer smoke tests.

Run:
    python -m scripts.test_embedder
or:
    python scripts/test_embedder.py
"""

import asyncio
import sys
from backend.models import EmbeddingChunk, EmbeddingMeta
from backend.core.dependencies import get_embedder, close_embedder


# ── helpers ────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    width = 60
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def _ok(msg: str) -> None:
    print(f"  ✓  {msg}")


def _fail(msg: str) -> None:
    print(f"  ✗  {msg}", file=sys.stderr)


# ── individual tests ────────────────────────────────────────────────────────

async def test_embed_query(embedder) -> bool:
    _section("TEST: embed_query")
    try:
        vec = await embedder.embed_query("What is machine learning?")

        assert isinstance(vec, list), "Result should be a list"
        assert len(vec) > 0,         "Vector should not be empty"
        assert all(isinstance(v, float) for v in vec), "All values should be floats"

        _ok(f"Vector length : {len(vec)}")
        _ok(f"First 5 values: {[round(v, 6) for v in vec[:5]]}")
        return True

    except Exception as exc:
        _fail(f"embed_query failed: {exc}")
        return False


async def test_embed_documents(embedder) -> bool:
    _section("TEST: embed_documents (late_chunking=False)")

    texts = [
        "Machine learning is a subset of AI.",
        "Deep learning uses neural networks.",
        "Transformers changed NLP forever.",
    ]

    chunks_seen = 0
    meta_seen = 0

    try:
        async for item in embedder.embed_documents(texts, late_chunking=False):

            if isinstance(item, EmbeddingChunk):
                chunks_seen += 1
                assert isinstance(item.embedding, list) and len(item.embedding) > 0, \
                    f"Chunk {item.index} has empty embedding"
                print(
                    f"  [Chunk {item.index}] "
                    f"tokens={item.tokens} "
                    f"estimated={item.is_estimate} "
                    f"text='{item.text[:40]}...'"
                )

            elif isinstance(item, EmbeddingMeta):
                meta_seen += 1
                _ok(f"Model        : {item.model}")
                _ok(f"Total tokens : {item.total_tokens}")
                _ok(f"Total chunks : {item.total_chunks}")
                assert item.total_chunks == len(texts), \
                    f"Expected {len(texts)} chunks, got {item.total_chunks}"

        assert chunks_seen == len(texts), \
            f"Expected {len(texts)} EmbeddingChunk items, got {chunks_seen}"
        assert meta_seen == 1, \
            f"Expected exactly 1 EmbeddingMeta, got {meta_seen}"

        _ok("All assertions passed")
        return True

    except Exception as exc:
        _fail(f"embed_documents failed: {exc}")
        return False


async def test_embed_documents_late_chunking(embedder) -> bool:
    _section("TEST: embed_documents (late_chunking=True)")

    texts = [
        "The Eiffel Tower is located in Paris.",
        "It was built in 1889.",
        "It stands 330 metres tall.",
    ]

    try:
        chunks = []
        async for item in embedder.embed_documents(texts, late_chunking=True):
            if isinstance(item, EmbeddingChunk):
                chunks.append(item)
            elif isinstance(item, EmbeddingMeta):
                _ok(f"Tokens (late_chunking): {item.total_tokens}")

        assert len(chunks) == len(texts), \
            f"Expected {len(texts)} chunks, got {len(chunks)}"
        _ok(f"Received {len(chunks)} chunks with late_chunking=True")
        return True

    except Exception as exc:
        _fail(f"late_chunking test failed: {exc}")
        return False


async def test_empty_documents(embedder) -> bool:
    _section("TEST: embed_documents (empty list)")
    try:
        items = []
        async for item in embedder.embed_documents([]):
            items.append(item)

        assert len(items) == 1 and isinstance(items[0], EmbeddingMeta), \
            "Empty input should yield exactly one EmbeddingMeta"
        assert items[0].total_chunks == 0
        _ok("Empty input handled gracefully")
        return True

    except Exception as exc:
        _fail(f"empty-list test failed: {exc}")
        return False


async def test_empty_query(embedder) -> bool:
    _section("TEST: embed_query (empty string)")
    try:
        await embedder.embed_query("   ")
        _fail("Should have raised ValueError for empty query")
        return False
    except ValueError:
        _ok("ValueError raised correctly for empty query")
        return True
    except Exception as exc:
        _fail(f"Unexpected exception: {exc}")
        return False


# ── main ────────────────────────────────────────────────────────────────────

async def main() -> None:
    embedder = get_embedder("jina")

    tests = [
        test_embed_query,
        test_embed_documents,
        test_embed_documents_late_chunking,
        test_empty_documents,
        test_empty_query,
    ]

    results = []
    try:
        for test_fn in tests:
            passed = await test_fn(embedder)
            results.append((test_fn.__name__, passed))
    finally:
        await close_embedder()  # always clean up

    # ── summary ─────────────────────────────────────────────────────────
    _section("SUMMARY")
    passed_count = sum(1 for _, ok in results if ok)
    for name, ok in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}  {name}")

    print(f"\n  {passed_count}/{len(results)} tests passed")

    if passed_count < len(results):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())