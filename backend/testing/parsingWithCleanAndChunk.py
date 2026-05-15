from backend.rag.components import ParserDispatcher, CleanerDispatcher, ChunkerDispatcher
from backend.models.metadata import Metadata

chunker = ChunkerDispatcher()
cleaner = CleanerDispatcher()
parser  = ParserDispatcher()

def test(doc: bytes, docname: str):
    for chunk in parser.parse(doc, docname):

        raw_text = chunk["text"]
        raw_meta = chunk["metadata"]  # Pydantic object

        # 1) clean
        cleaned = cleaner.clean(
            raw_text,
            raw_meta
        )

        cleaned_text = cleaned["text"]

        # 2) build metadata (UPDATED after cleaning)
        doc_metadata = Metadata(
            source=raw_meta.source,
            ext=raw_meta.ext,
            language=raw_meta.language,
            chars=len(cleaned_text),
            pages=raw_meta.pages,
            document_id=raw_meta.document_id,
        )

        # 3) chunking
        yield from chunker.chunk(cleaned_text, doc_metadata)

if __name__ == "__main__":

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

    print("=== Arabic ===")
    for chunk_meta in test(doc=doc_ar.encode("utf-8"), docname="test.txt"):
        print(f"[{chunk_meta['metadata'].chunk_index}] words={chunk_meta['metadata'].word_count} | {chunk_meta['text'][:60]}...")
        print(f"[{chunk_meta['metadata'].chunk_index}] lang={chunk_meta['metadata'].language} | ext={chunk_meta['metadata'].ext} | chunker={chunk_meta['metadata'].chunker} ")

    print("-=*=-" * 20)

    print("=== English ===")
    for chunk_meta in test(doc_en.encode("utf-8"), docname="test.md"):
        print(f"[{chunk_meta['metadata'].chunk_index}] words={chunk_meta['metadata'].word_count} | {chunk_meta['text'][:60]}...")
        print(f"[{chunk_meta['metadata'].chunk_index}] lang={chunk_meta['metadata'].language} |  chunker={chunk_meta['metadata'].chunker}")