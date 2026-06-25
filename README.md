# SecRAG

**SecRAG** (Secure Retrieval-Augmented Generation) is a production-grade private RAG system for querying PDF documents. Built over four months, it evolved from a basic vector search demo into a fully-featured pipeline with hybrid retrieval, a cross-encoder reranker, citation verification, and an automated evaluation framework.

---

## What It Does

- Upload PDF documents and query them in natural language
- Retrieve context using semantic search, BM25 keyword search, or hybrid fusion
- Generate grounded answers with inline chunk citations
- Verify every citation using an LLM-as-judge pass
- Run fully locally or inside Docker with no data leaving your machine

---

## Evaluation Results

Evaluated on a 48-question golden Q&A dataset against the BERT paper (Devlin et al., 2019):

| Metric | Score |
|---|---|
| Average answer quality | 4.15 / 5.0 |
| Citation accuracy | 99.6% |
| Questions scored 4 or 5 | 83% |
| Total questions | 48 |

---

## How It Was Built

### Month 1 — Core RAG Foundation
Started with the fundamentals: PDF text extraction, sentence-aware chunking, MiniLM embeddings, and a basic cosine similarity retriever. The goal was to get a working end-to-end pipeline before adding any complexity.

- PDF ingestion with `pypdf`
- Sentence-based chunking with character span tracking
- `all-MiniLM-L6-v2` embeddings via `sentence-transformers`
- FastAPI backend with API key auth and request logging
- React + Vite frontend with dark mode and retrieval mode selector

### Month 2 — Hybrid Retrieval
Added BM25 sparse retrieval alongside dense vector search, then combined them using a weighted fusion layer. This significantly improved performance on technical documents where exact keyword matching matters.

- `rank-bm25` integration for sparse lexical scoring
- Weighted hybrid fusion (configurable alpha)
- Three retrieval modes: Semantic, BM25, Hybrid
- Toggleable from the frontend with no backend restart needed

### Month 3 — Production Hardening
Migrated from flat `.npy` file storage to ChromaDB for persistent, scalable vector indexing. Added near-duplicate detection at ingestion time, three chunking strategies, and Reciprocal Rank Fusion to replace the weighted linear combination.

- **ChromaDB** persistent vector store replacing `.npy` files
- **Near-duplicate deduplication** at ingestion (cosine threshold 0.95)
- **Three chunking strategies**: fixed-size, sentence-aware, semantic (topic-boundary detection)
- **Reciprocal Rank Fusion (RRF)** replacing weighted hybrid combination
- **Cross-encoder reranker** (`ms-marco-MiniLM-L-6-v2`) as a second-pass filter: top-20 candidates → top-5

### Month 4 — Quality Layer and Eval Framework
Added citation verification using GPT-4o-mini as judge, an automated evaluation framework with a golden Q&A dataset, and chunking strategy comparison tooling. Wired everything into the API response so every answer now returns a `citation_accuracy` score.

- **Citation verification**: parses `[Chunk N]` markers, checks each claim against its source chunk
- **LLM-as-judge** scoring with per-citation support/unsupported verdicts
- **Golden Q&A eval framework**: `GoldenDataset` class, `run_eval()`, per-difficulty and per-category breakdowns
- **Chunking strategy comparison**: run all three strategies against the same eval suite
- **Structured answer format**: `verified_answer`, `citation_accuracy`, `citation_details` in every `/answer` response

---

## Architecture

### Retrieval Pipeline

```
Query
  → Embed (MiniLM)
  → Dense retrieval from ChromaDB (top-20)
  → BM25 sparse retrieval (top-20)
  → RRF fusion
  → Cross-encoder reranker (top-20 → top-5)
  → LLM answer generation with [Chunk N] citations
  → Citation verification (LLM-as-judge)
  → Response
```

### Backend Structure

```
backend/
├── app.py
├── utils/
│   ├── vector_store.py        # ChromaDB layer + deduplication
│   ├── chunking_strategies.py # fixed / sentence / semantic
│   ├── retriever.py           # RRF fusion + reranker
│   ├── reranker.py            # cross-encoder reranker
│   ├── citation_verifier.py   # LLM-as-judge citation check
│   ├── eval_framework.py      # golden Q&A eval suite
│   ├── embeddings.py          # MiniLM embedding layer
│   ├── bm25.py                # BM25 sparse retrieval
│   ├── uploader.py            # PDF ingestion pipeline
│   └── llm.py                 # OpenAI answer generation
├── data/
│   └── chroma/                # ChromaDB persistent storage
├── requirements.txt
└── Dockerfile
```

### Frontend Structure

```
frontend/
├── src/
│   └── App.jsx                # React single-file app
├── Dockerfile
└── vite.config.js
```

---

## API Response Format

Every `/answer` call returns:

```json
{
  "filename": "document.pdf",
  "query": "What is X?",
  "answer": "X is ... [Chunk 3]. It was introduced in ... [Chunk 7].",
  "verified_answer": "X is ... [Chunk 3]. It was introduced in ... [Chunk 7].",
  "citation_accuracy": 0.92,
  "citation_details": [
    {
      "chunk_id": "3",
      "claim": "X is ...",
      "supported": true,
      "confidence": 0.95,
      "reason": "Source text directly states this."
    }
  ],
  "citations": [
    { "chunk_id": "3", "score": 0.033, "char_range": [412, 595] }
  ]
}
```

---

## Setup

### Requirements

```
Python 3.10+
Node.js 18+
OpenAI API key
```

### Environment

Backend `.env`:
```
OPENAI_API_KEY=your_key_here
SECRAG_API_KEY=                  # optional, leave blank for local use
ALLOWED_ORIGINS=http://localhost:5173
MAX_UPLOAD_MB=25
```

Frontend `.env`:
```
VITE_API_BASE=http://localhost:8000
VITE_API_KEY=
```

### Run Locally

```bash
# Backend
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
uvicorn app:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`

### Docker

```bash
docker compose up --build
```

Access:
- Backend: `http://localhost:8000`
- Frontend: `http://localhost:5173`

---

## Key Dependencies

| Package | Purpose |
|---|---|
| `fastapi` | API framework |
| `sentence-transformers` | MiniLM embeddings + cross-encoder reranker |
| `chromadb` | Persistent vector store |
| `rank-bm25` | Sparse keyword retrieval |
| `pypdf` | PDF text extraction |
| `openai` | LLM answer generation + citation verification |
| `numpy` | Vector operations |

---

## Design Decisions

**Why ChromaDB over FAISS?** ChromaDB persists to disk automatically, supports metadata filtering, and requires zero infrastructure. FAISS is faster at scale but needs manual index management.

**Why RRF over weighted fusion?** Weighted fusion requires tuning alpha per use case. RRF is rank-based so it's robust to score scale differences between dense and sparse retrievers, and it consistently outperforms weighted fusion without any tuning.

**Why a cross-encoder reranker?** Bi-encoders (like MiniLM) encode query and document independently, which is fast but loses interaction signals. A cross-encoder sees both together, giving much higher relevance precision on the final top-5 results at acceptable latency.

**Why citation verification?** RAG systems can retrieve the right chunks but still hallucinate claims not present in those chunks. Verifying each `[Chunk N]` citation independently catches this class of error that retrieval metrics alone cannot detect.

---

## What SecRAG Demonstrates

- End-to-end RAG pipeline from ingestion to verified answer
- Hybrid retrieval with RRF fusion
- Two-stage retrieval: fast recall (top-20) then precise reranking (top-5)
- Production observability: structured logging, request IDs, health endpoint
- Quality measurement: citation accuracy as a deployable metric
- Modular architecture: each component is independently replaceable

---

## Name

SecRAG = **Sec**ure **R**etrieval-**A**ugmented **G**eneration.

Built for private document environments where data does not leave the deployment boundary.