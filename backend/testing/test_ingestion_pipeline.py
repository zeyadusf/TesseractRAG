"""
IngestionPipeline smoke test.

Run:
    python -m python -m backend.testig.test_ingestion_pipeline

"""

import asyncio
import sys
from backend.rag.pipelines import IngestionPipeline


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


# ── test cases ──────────────────────────────────────────────────────────────

async def test_pipeline_arabic(pipeline: IngestionPipeline) -> bool:
    _section("TEST: IngestionPipeline — Arabic (.txt)")

    doc_ar = """
    هذا نص تجريبي لاختبار معالجة الملفات في نظام RAG.
    يحتوي هذا الملف على عدة فقرات مختلفة للتأكد من أن النظام يستطيع التعامل مع النصوص الطويلة بشكل صحيح دون فقدان أي معلومات.
    الفصل الأول: البيانات النصية
    يتم استخدام هذا القسم لاختبار قدرة الـ parser على استخراج النصوص من الملفات بشكل دقيق.
    الفصل الثاني: الأرقام والرموز
    يحتوي النص على أرقام مثل 12345 و 987654321، بالإضافة إلى رموز خاصة مثل !؟@#%&*().
    الفصل الثالث: اللغة الطبيعية
    هذا الجزء يحتوي على جمل متداخلة ومعقدة قليلاً لاختبار قدرة الـ chunker على الحفاظ على السياق.
    الهدف النهائي هو التأكد من أن النظام يعمل بكفاءة من مرحلة القراءة إلى مرحلة التقسيم والتجهيز للـ embeddings.
    """

    try:
        result = await pipeline.run(
            file_bytes=doc_ar.encode("utf-8"),
            filename="test_arabic.txt",
        )

        _assert_result_shape(result)

        _ok(f"Chunks produced  : {len(result['chunks'])}")
        _ok(f"Embeddings       : {len(result['embeddings'])}")
        _ok(f"Model            : {result['metadata']['model']}")
        _ok(f"Total tokens     : {result['metadata']['total_tokens']}")
        _ok(f"Dimensions       : {result['metadata']['dimensions']}")

        _print_chunks(result["chunks"], result["embeddings"], limit=3)
        return True

    except Exception as exc:
        _fail(f"Arabic pipeline test failed: {exc}")
        return False


async def test_pipeline_english(pipeline: IngestionPipeline) -> bool:
    _section("TEST: IngestionPipeline — English (.md)")

    doc_en = """
This is a comprehensive sample document used for testing a full RAG pipeline including parsing, cleaning, and chunking.
Section 1: Text Processing
This section ensures that the parser correctly reads multi-paragraph structured text without losing formatting or meaning.
Section 2: Numbers and Special Characters
The document includes numeric values such as 12345, 987654321, and floating numbers like 3.14159, along with symbols like !?@#%&*().
Section 3: Natural Language Complexity
This part contains longer and slightly complex sentences to evaluate how well the chunker preserves semantic meaning across splits.
Final Goal:
To validate that the system correctly handles encoding, decoding, cleaning noise, and producing high-quality chunks suitable for embeddings.
"""

    try:
        result = await pipeline.run(
            file_bytes=doc_en.encode("utf-8"),
            filename="test_english.md",
        )

        _assert_result_shape(result)

        _ok(f"Chunks produced  : {len(result['chunks'])}")
        _ok(f"Embeddings       : {len(result['embeddings'])}")
        _ok(f"Model            : {result['metadata']['model']}")
        _ok(f"Total tokens     : {result['metadata']['total_tokens']}")
        _ok(f"Dimensions       : {result['metadata']['dimensions']}")

        _print_chunks(result["chunks"], result["embeddings"], limit=3)
        return True

    except Exception as exc:
        _fail(f"English pipeline test failed: {exc}")
        return False


async def test_pipeline_empty(pipeline: IngestionPipeline) -> bool:
    _section("TEST: IngestionPipeline — empty content")

    try:
        result = await pipeline.run(
            file_bytes=b"   ",
            filename="empty.txt",
        )

        assert result["chunks"] == [],     "Expected empty chunks list"
        assert result["embeddings"] == [], "Expected empty embeddings list"
        assert result["metadata"] is None, "Expected None metadata for empty input"

        _ok("Empty input handled gracefully")
        return True

    except Exception as exc:
        _fail(f"Empty-content test failed: {exc}")
        return False


async def test_chunk_embedding_alignment(pipeline: IngestionPipeline) -> bool:
    _section("TEST: chunk ↔ embedding index alignment")

    doc = "Alpha. Beta. Gamma. Delta. Epsilon. Zeta. Eta. Theta."

    try:
        result = await pipeline.run(
            file_bytes=doc.encode("utf-8"),
            filename="alignment_test.txt",
        )

        chunks     = result["chunks"]
        embeddings = result["embeddings"]

        assert len(chunks) == len(embeddings), (
            f"Chunk count ({len(chunks)}) != embedding count ({len(embeddings)})"
        )

        for emb in embeddings:
            idx = emb["index"]
            assert emb["text"] == chunks[idx]["text"], (
                f"Text mismatch at index {idx}"
            )
            assert isinstance(emb["vector"], list) and len(emb["vector"]) > 0, (
                f"Empty vector at index {idx}"
            )

        _ok(f"All {len(chunks)} chunk↔embedding pairs aligned correctly")
        return True

    except Exception as exc:
        _fail(f"Alignment test failed: {exc}")
        return False


# ── shared assertions & display ─────────────────────────────────────────────

def _assert_result_shape(result: dict) -> None:
    assert "chunks"     in result, "Missing 'chunks' key"
    assert "embeddings" in result, "Missing 'embeddings' key"
    assert "metadata"   in result, "Missing 'metadata' key"

    assert len(result["chunks"]) > 0,     "No chunks produced"
    assert len(result["embeddings"]) > 0, "No embeddings produced"
    assert result["metadata"] is not None, "metadata is None despite valid input"

    assert len(result["chunks"]) == len(result["embeddings"]), (
        f"chunks={len(result['chunks'])} != embeddings={len(result['embeddings'])}"
    )


def _print_chunks(chunks: list, embeddings: list, *, limit: int = 3) -> None:
    print()
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        if i >= limit:
            remaining = len(chunks) - limit
            print(f"  ... and {remaining} more chunk(s)")
            break

        meta = chunk.get("chunk_metadata", {})
        lang     = getattr(meta, "language", "?")
        chunker  = getattr(meta, "chunker",  "?")
        ch_index = getattr(meta, "chunk_index", i)

        print(
            f"  [{ch_index}] "
            f"words={chunk['word_count']:>4}  "
            f"tokens={emb['tokens']:>5}  "
            f"estimated={emb['is_estimate']}  "
            f"lang={lang}  chunker={chunker}"
        )
        print(f"       text='{chunk['text'][:70]}...'")
        print(f"       vec[:4]={[round(v, 6) for v in emb['vector'][:4]]}")


# ── main ────────────────────────────────────────────────────────────────────

async def main() -> None:
    pipeline = IngestionPipeline()

    tests = [
        test_pipeline_arabic,
        test_pipeline_english,
        test_pipeline_empty,
        test_chunk_embedding_alignment,
    ]

    results = []
    for test_fn in tests:
        passed = await test_fn(pipeline)
        results.append((test_fn.__name__, passed))

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