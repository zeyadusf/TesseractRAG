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

# Path is relative to repo root — works when Dockerfile is at repo root
COPY backend/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir torch==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

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

# Copy entire backend folder as app code
COPY backend/ .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]