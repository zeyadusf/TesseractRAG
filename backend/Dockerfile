# ── Stage: base ────────────────────────────────────────────────────────────────
# python:3.11-slim is ~50MB vs python:3.11 at ~900MB.
# "slim" has no build tools by default — we add only what we need.
FROM python:3.11-slim

# Set working directory inside the container.
# All subsequent COPY and RUN commands resolve relative to this.
WORKDIR /app

# Tell Python not to write .pyc files (saves a little disk space)
# and to run in unbuffered mode (logs appear immediately in Render's log stream).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # HuggingFace model cache — baked into the image during build.
    # At runtime, models load from here instantly (no download).
    HF_HOME=/app/.cache/huggingface \
    # Suppress the HuggingFace "token not set" warning during model download.
    # We only need a token for the Inference API, not for downloading public models.
    TRANSFORMERS_VERBOSITY=error

# ── Install system dependencies ────────────────────────────────────────────────
# gcc and g++ are required to compile some Python packages (e.g. faiss-cpu).
# We install them, use them, then remove them to keep the image lean.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies ────────────────────────────────────────────────
# COPY requirements.txt BEFORE copying the rest of the code.
# This is the most important Dockerfile optimization:
#   - If requirements.txt hasn't changed, Docker uses the cached pip layer.
#   - Only if requirements.txt changes does Docker re-run pip install.
#   - A typo fix in chat.py does NOT trigger a 4-minute pip install.
COPY requirements.txt .

# Install torch CPU-only FIRST — it's the largest package and has its own index.
# The --index-url pulls the CPU-only build (~700MB vs ~2.5GB for the GPU version).
# Render free tier has no GPU, so the GPU build is wasted space and may OOM.
RUN pip install --no-cache-dir torch==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Install all other dependencies.
# torch is already installed above so pip won't reinstall it.
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-download HuggingFace models ───────────────────────────────────────────
# This is the key optimization for free-tier cold starts.
# We download both models NOW, during docker build, so they are baked into
# the image layer. At runtime, SentenceTransformer() and CrossEncoder()
# load from the local cache in milliseconds instead of downloading from HF.
#
# Why a separate RUN for each model?
#   Docker caches each RUN layer independently. If the embedding model
#   download succeeds but the reranker fails, a rebuild only re-downloads
#   the reranker — not both. Saves time during development.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
print('Downloading embedding model...'); \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
print('Embedding model cached.')"

RUN python -c "\
from sentence_transformers import CrossEncoder; \
print('Downloading reranker model...'); \
CrossEncoder('BAAI/bge-reranker-base'); \
print('Reranker model cached.')"

# ── Copy application code ──────────────────────────────────────────────────────
# This layer changes every time you edit code.
# It comes LAST so that code changes don't invalidate the pip or model layers.
COPY . .

# ── Expose and run ─────────────────────────────────────────────────────────────
# Render maps its own port to whatever your container EXPOSEs.
# We use 8000 to match local development.
EXPOSE 8000

# --host 0.0.0.0 is required in Docker — without it, uvicorn binds to
# localhost inside the container, which is unreachable from outside.
# --workers 1 because Render free tier has limited CPU and our models
# are already loaded as singletons — multiple workers would duplicate them.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]