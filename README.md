# SecRAG Pro

A production-grade RAG system with intelligent LLM cost routing; combining hybrid retrieval, citation verification, and automated model selection to deliver high-quality answers at a fraction of the cost.

**62.3% cost reduction on answer generation | 99.6% citation accuracy | 4.15/5.0 answer quality**

---

## What it is

SecRAG Pro integrates two production systems:

- **SecRAG**: A hybrid RAG pipeline with BM25 + semantic retrieval, cross-encoder reranking, and LLM-as-judge citation verification
- **LLM Cost Autopilot**: An intelligent routing layer that classifies each query's complexity and routes it to the cheapest capable model

The result: every question asked of a document gets answered by the right model at the right cost. A simple factual lookup routes to Llama 3.2 for free. A complex synthesis question routes to GPT-4o or Claude Sonnet. The system decides automatically; no configuration needed per query.

---

## Architecture

```
User Question
      |
      v
SecRAG Pro — Hybrid Retrieval (BM25 + Semantic)
      |
      v
Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)
      |
      v
Top-K Chunks selected
      |
      v
LLM Cost Autopilot /v1/completions
      |
      +-- Tier 1 (simple)   --> Llama 3.2 (free, local)
      +-- Tier 2 (moderate) --> GPT-4o Mini or Claude Haiku
      +-- Tier 3 (complex)  --> Claude Sonnet or GPT-4o
      |
      v
Answer with inline citations
      |
      v
Citation Verifier (LLM-as-judge, GPT-4o Mini)
      |
      v
Verified answer + citation accuracy score returned to caller
```

---

## Results

| Metric | Value |
|--------|-------|
| Citation accuracy | 99.6% |
| Average answer quality | 4.15 / 5.0 |
| Cost reduction vs all-GPT-4o | 62.3% |
| Total requests in load test | 513 |
| Failed requests | 0 |
| Classifier accuracy | 97%+ |

### Model distribution on load test
| Model | Traffic share |
|-------|--------------|
| GPT-4o Mini | 41.3% |
| Llama 3.2 (free, local) | 28.3% |
| GPT-4o | 19.3% |
| Claude Sonnet | 11.1% |

---

## Stack

- **Backend**: FastAPI, Python 3.11
- **Retrieval**: BM25 (rank-bm25) + Semantic (sentence-transformers/all-MiniLM-L6-v2)
- **Reranker**: cross-encoder ms-marco-MiniLM-L-6-v2
- **Vector store**: ChromaDB
- **LLM routing**: LLM Cost Autopilot (scikit-learn RandomForest classifier, 221 training prompts)
- **LLM providers**: OpenAI (GPT-4o, GPT-4o Mini), Anthropic (Claude Sonnet, Claude Haiku), Ollama (Llama 3.2)
- **Citation verification**: LLM-as-judge (GPT-4o Mini)
- **Frontend**: React + Tailwind
- **Containerization**: Docker + docker-compose

---

## How it works

**1. Upload a document**
PDF is chunked using sentence-aware splitting, embedded with all-MiniLM-L6-v2, and stored in ChromaDB.

**2. Ask a question**
The query runs through hybrid retrieval; BM25 for keyword matching, semantic search for meaning, combined with a configurable alpha weight. A cross-encoder reranker scores the top candidates.

**3. Intelligent routing**
The top chunks and query are sent to the LLM Cost Autopilot. A RandomForest classifier (trained on 221 labeled prompts, 97%+ accuracy) classifies the query complexity into Tier 1, 2, or 3 and routes to the cheapest model that can handle it.

**4. Answer generation**
The routed model generates an answer using only the retrieved chunks, with mandatory inline citations ([Chunk N]) after every claim.

**5. Citation verification**
GPT-4o Mini verifies each citation against the source chunk. Unsupported citations are flagged in the response.

**6. Response**
The API returns the answer, verified answer, citation accuracy score, cost, model used, and routing metadata.

---

## Setup

**Prerequisites**
- LLM Cost Autopilot running on port 8000 (see [llm-cost-autopilot](https://github.com/vaibhav-badoliasoft/llm-cost-autopilot))
- Ollama running locally with Llama 3.2 pulled

**1. Clone and create virtual environment**
```bash
git clone https://github.com/vsaini06/SecRAG-Pro.git
cd SecRAG-Pro
python -m venv venv
venv\Scripts\activate
```

**2. Install dependencies**
```bash
cd backend
pip install -r requirements.txt
```

**3. Configure environment**
```bash
# backend/.env
OPENAI_API_KEY=sk-...
AUTOPILOT_URL=http://localhost:8000/v1/completions
USE_AUTOPILOT=true
AUTOPILOT_TIMEOUT=120
```

**4. Start LLM Cost Autopilot first**
```bash
cd path/to/llm-cost-autopilot/backend
python app.py
```

**5. Start SecRAG Pro**
```bash
cd SecRAG-Pro/backend
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

**6. Open API docs**
```
http://localhost:8001/docs
```

---

## API

| Endpoint | Method | Purpose |
|----------|--------|---------|
| /upload | POST | Upload and process a PDF |
| /answer | POST | Ask a question, get routed answer with citations |
| /summarize | POST | Summarize a document with routing metadata |
| /retrieve | POST | Raw hybrid retrieval without generation |
| /list_docs | GET | List uploaded documents |
| /health | GET | Health check |

---

## Related projects

- [LLM Cost Autopilot](https://github.com/vsaini06/llm-cost-autopilot): the routing layer powering SecRAG Pro's model selection
- [SecRAG](https://github.com/vsaini06/secrag): the original RAG system this is built on

---

## Author

Vaibhav Saini: [GitHub](https://github.com/vsaini06) · [Portfolio](https://vaibhavsaini-portfolio.vercel.app)
