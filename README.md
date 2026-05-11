# Bilingual Legal RAG — Technical Report

> A bilingual (English/Arabic) Retrieval-Augmented Generation system for querying legal documents, built with FastAPI, LanceDB, MongoDB, and Ollama.

---

## Executive Summary

Bilingual Legal RAG is a locally-deployed, privacy-preserving question-answering system that allows users to query a curated corpus of English and Arabic legal texts in natural language. The system combines a **query transformation** step (rewriting user queries into semantically richer search strings), **dense vector retrieval** over a LanceDB vector store, and **Ollama-served LLM inference** to produce grounded, context-faithful answers — all without sending any data to external APIs.

The entire stack is containerised and launched with a single `docker compose up -d` command. Key design goals were:

- **Bilingual parity** — identical pipeline for English and Arabic queries
- **Groundedness** — the model is strictly instructed to answer only from retrieved context
- **Low resource footprint** — runs on consumer hardware (CPU or modest GPU)
- **Reproducibility** — deterministic seeding of MongoDB and LanceDB from static JSON files

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Docker Compose Network                       │
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐               │
│  │  Client  │───▶│   FastAPI    │───▶│   MongoDB   │               │
│  │  (HTTP)  │    │  :8000       │    │  :27017     │               │
│  └──────────┘    │              │    │  (law store)│               │
│                  │  /api/v1/rag │    └─────────────┘               │
│                  │  /chat       │                                    │
│                  │  /query      │    ┌─────────────┐               │
│                  │              │───▶│   LanceDB   │               │
│                  │              │    │  (on disk)  │               │
│                  │              │    │  vector idx │               │
│                  └──────┬───────┘    └─────────────┘               │
│                         │                                           │
│                         ▼                                           │
│                  ┌──────────────┐                                   │
│                  │    Ollama    │                                   │
│                  │  :11434      │                                   │
│                  │  qwen3:0.8b  │                                   │
│                  └──────────────┘                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Request / Response Flow — `/api/v1/rag/chat`

```
User Query  ──▶  [1] Query Transformation (Ollama)
                        │
                        ▼
                 Optimised Search Query
                        │
                        ▼
                 [2] Dense Retrieval (LanceDB)
                        │  top-8 chunks (cosine similarity)
                        ▼
                 Retrieved Context Chunks
                        │
                        ▼
                 [3] Prompt Construction
                      (context + original query)
                        │
                        ▼
                 [4] Answer Generation (Ollama)
                        │
                        ▼
                 JSON Response
                 {original_query, optimised_query,
                  answer, retrieved_context}
```

---

## API Documentation

### Base URL

```
http://localhost:8000/api/v1/rag
```

All endpoints accept and return `application/json`.

---

### `POST /chat`

Natural-language legal Q&A with query transformation and RAG. **Primary endpoint.**

#### Request Body

```json
{
  "query": "What are the conditions for terminating an employment contract?",
  "lang": "en"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | `string` | ✅ | The user's legal question |
| `lang` | `string` | ✅ | Language code — `"en"` or `"ar"` |

#### Response Body

```json
{
  "original_query": "What are the conditions for terminating an employment contract?",
  "optimized_search_query": "Conditions and legal grounds for employment contract termination under labour law",
  "answer": "According to Article 41 of the Labour Law, an employer may terminate...",
  "retrieved_context": "Article 41: The employer may terminate the contract if..."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `original_query` | `string` | The query exactly as submitted |
| `optimized_search_query` | `string` | The LLM-rewritten vector search query |
| `answer` | `string` | The model's grounded answer |
| `retrieved_context` | `string` | The raw chunks passed as context |

#### Error Responses

| Status | Condition |
|--------|-----------|
| `400 Bad Request` | `lang` is not `"en"` or `"ar"` |
| `500 Internal Server Error` | LLM or DB failure |

---

### `POST /query`

Basic RAG query without query transformation. Returns structured chunks and source metadata.

#### Request Body

```json
{
  "query": "penalty for breach of contract",
  "top_k": 5
}
```

| Field | Type | Required | Default | Constraints |
|-------|------|----------|---------|-------------|
| `query` | `string` | ✅ | — | min length 3 |
| `top_k` | `integer` | ❌ | `5` | 1 – 10 |

#### Response Body

```json
{
  "answer": "...",
  "sources": [
    {
      "title": "Contract Law Act 1990",
      "category": "civil",
      "act_id": "CLA-1990",
      "url_source": "https://...",
      "chunk_index": "12"
    }
  ],
  "retrieved_chunks": [
    {
      "text": "Section 17: Any party in breach of...",
      "score": 0.87,
      "metaData": { ... }
    }
  ]
}
```

---

## Embedding Model & Chunking Strategy

### Embedding Model

**Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

| Property | Value |
|----------|-------|
| Dimensions | 384 |
| Languages supported | 50+ (including Arabic) |
| Model size | ~118 MB |
| Inference | CPU-friendly |

**Justification:** Legal texts in two scripts (Latin and Arabic) require a single embedding space where semantically equivalent concepts — regardless of language — map to nearby vectors. `paraphrase-multilingual-MiniLM-L12-v2` was specifically trained for cross-lingual semantic similarity and consistently outperforms monolingual models on multilingual retrieval benchmarks. Its small size keeps Docker image build times short and allows the model to be bundled into the image at build time (see Dockerfile `RUN uv run python -c "from sentence_transformers..."`) so cold-start latency is eliminated.

### Chunking Strategy

Laws are stored as structured documents in MongoDB (one document per act/statute). The `embedding.py` script reads each document and applies **paragraph-level chunking** with the following properties:

- **Chunk unit:** Natural paragraph / article boundary (legal texts are already article-segmented)
- **Chunk size:** ~300–500 tokens per chunk, respecting article delimiters
- **Overlap:** None — article boundaries are semantically complete units; overlap would duplicate article headers and confuse retrieval
- **Metadata stored per chunk:** `title`, `category`, `act_id`, `url_source`, `chunk_index`, `lang`

**Why article-level chunking?** Legal reasoning is article-centric. A question about "termination conditions" should retrieve Article 41 as a whole, not a mid-sentence fragment. Splitting mid-article would break the logical unit the model needs to reason over.

---

## Edge Cases & Known Limitations

### 1. Out-of-Scope Queries (No Relevant Chunks Found)

**Scenario:** The user asks a question whose answer does not exist in the seeded legal corpus, or submits a malformed/incomplete question (e.g. `"what about section..."`).

**Observed behaviour:** LanceDB still returns the top-8 closest vectors by cosine similarity, even when similarity scores are low. The model receives weakly-relevant chunks and, without a relevance threshold gate, may hallucinate an answer derived from superficially related articles.

**Mitigation applied:** The system prompt instructs the model to respond with the exact phrase `"I cannot answer this based on the provided legal text."` when context is insufficient. However, this relies on model instruction-following, which is imperfect at 0.8B scale.

**Recommended fix:** Introduce a minimum cosine similarity threshold (e.g. `score < 0.45` → return fallback response without calling the LLM). This would be implemented in the `get_top_chunks` method of the DB manager.

---

### 2. Language Mismatch Between Query and `lang` Parameter

**Scenario:** User submits an English question but sets `lang: "ar"`, or vice versa.

**Observed behaviour:** The query transformation step rewrites the query in the wrong language, causing LanceDB to search the wrong language partition of the vector index. Retrieved chunks are in the wrong language. The answer generation prompt is assembled with mismatched-language context, leading to broken or hallucinated output.

**Root cause:** Language routing is based solely on the `lang` parameter, with no automatic detection of the actual query language.

**Recommended fix:** Add lightweight language detection (e.g. `langdetect` or a character-script heuristic — Arabic script vs Latin script) before the query transformation step. If a mismatch is detected, either override `lang` or return a clear `400` error prompting the user to correct the parameter.

---

### 3. Complex Cross-Reference Dependencies in Retrieved Chunks

**Scenario:** A retrieved article contains phrases like *"subject to the provisions of Section 7(b) and Article 23 of the Civil Code"*, where the full legal meaning depends on articles that were not retrieved.

**Observed behaviour:** The model attempts to interpret the reference in isolation, frequently misattributing the scope or conditions of the referenced sections, producing legally incorrect (hallucinated) answers.

**Root cause:** Dense retrieval is query-driven — it retrieves articles most similar to the query, not the dependency graph of the top-ranked article. Cross-referenced articles are invisible to the model unless they independently ranked in the top-8.

**Recommended fix (two options):**
- **Post-retrieval expansion:** Parse chunk text for article/section reference patterns (regex on `Article \d+`, `Section \d+`) and fetch those articles from MongoDB to append to the context window.
- **Prompt-level hedging:** Add an instruction to the system prompt: *"If the retrieved text references other articles not provided, explicitly state that the answer may be incomplete due to unresolved legal references."*

---

## Installation & Running

### Prerequisites

| Requirement | Notes |
|-------------|-------|
| Docker ≥ 24 | Required |
| Docker Compose ≥ 2.20 | Required |
| `embedding.py` run once | After first `docker compose up` |
| ~4 GB disk space | For model weights + data |

> **No GPU required.** Ollama will run `qwen3:0.8b` on CPU. Inference will be slower (~5–15 s/query) but fully functional.

---

### Step 1 — Clone & Configure

```bash
git clone <repository-url>
cd bilingual-legal-rag
cp .env.example .env   # Edit if needed; defaults work out of the box
```

### Step 2 — Build & Start All Services

```bash
docker compose up -d --build
```

This will:
1. Pull and start **MongoDB** on port `27017`
2. Pull and start **Ollama** on port `11434`
3. Run `ollama pull qwen3:0.8b` via the `ollama-init` container
4. Build the **FastAPI** image (installs uv deps + downloads embedding model into image)
5. Start the FastAPI server on port `8000`
6. On first startup, FastAPI checks if MongoDB is empty and **auto-seeds** the database from `./seed_data/`

> The `ollama-init` container will exit once the model is downloaded — this is expected.

### Step 3 — Generate Embeddings (One-Time Setup)

After all services are healthy, run the embedding script **once** to chunk the seeded laws and populate LanceDB:

```bash
# From your host machine (uv must be installed locally), OR:
uv run python embedding.py

# Alternatively, exec into the running API container:
docker exec -it fastapi_rag_app uv run python embedding.py
```

This reads from MongoDB, chunks each law by article, embeds with `paraphrase-multilingual-MiniLM-L12-v2`, and writes vectors to `./lancedb_data/` (mounted into the container).

### Step 4 — Query the API

```bash
# English query
curl -X POST http://localhost:8000/api/v1/rag/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the penalties for contract breach?", "lang": "en"}'

# Arabic query
curl -X POST http://localhost:8000/api/v1/rag/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "ما هي شروط إنهاء عقد العمل؟", "lang": "ar"}'
```

### Stopping the Stack

```bash
docker compose down          # Stop containers, keep volumes
docker compose down -v       # Stop containers AND delete all data volumes
```

---

## Tech Stack Summary

| Component | Technology | Reason |
|-----------|------------|--------|
| API framework | FastAPI | Async support, automatic OpenAPI docs |
| LLM server | Ollama (`qwen3:0.8b`) | Local inference, no API keys |
| Vector DB | LanceDB | Embedded, zero-server, low memory |
| Document DB | MongoDB 4.4 | Flexible schema for heterogeneous legal docs |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | Bilingual semantic space |
| Package manager | uv | Fast, reproducible Python deps |
| Containerisation | Docker Compose | Single-command full-stack launch |

---

*Report generated for NLP Course — Phase 5 Technical Documentation.*
