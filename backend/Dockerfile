# ── Stage: base ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_VERBOSITY=error

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Install dependencies ───────────────────────────────────────────────────────
# COPY requirements first — so code changes don't invalidate this layer
COPY requirements.txt .

# torch CPU-only (~700MB vs ~2.5GB GPU version)
RUN pip install --no-cache-dir torch==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-download models ────────────────────────────────────────────────────────
# Baked into image so container starts in seconds with no runtime downloads.
# Using lighter reranker (90MB) instead of bge-reranker-base (278MB)
# to stay within Railway free tier 512MB RAM limit.

RUN python -c "\
from sentence_transformers import SentenceTransformer; \
print('Downloading embedding model...'); \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
print('Embedding model cached.')"

RUN python -c "\
from sentence_transformers import CrossEncoder; \
print('Downloading reranker model...'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('Reranker model cached.')"

# ── Copy application code ──────────────────────────────────────────────────────
# Last layer — code changes only rebuild from here, not pip or model layers
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]