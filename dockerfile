FROM python:3.11-slim
WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependencies first
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "bilingual_legal_rag.app.main:app", "--host", "0.0.0.0", "--port", "8000"]