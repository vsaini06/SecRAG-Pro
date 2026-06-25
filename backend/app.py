import os
import time
import uuid
from datetime import datetime
from pathlib import Path
import json
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from utils.uploader import process_pdf_upload
from utils.retriever import retrieve_top_k, load_chunks
from utils.naming import get_artifact_paths, safe_pdf_name, get_all_related_paths
from utils.llm import generate_answer
from utils.summarizer import summarize_from_chunks
from utils.citation_verifier import verify_citations

load_dotenv()

logger = logging.getLogger("secrag")
logging.basicConfig(level=logging.INFO, format="%(message)s")

SECRAG_API_KEY = os.getenv("SECRAG_API_KEY", "").strip()

app = FastAPI()

def _parse_origins(val: str | None):
    if not val:
        return ["http://localhost:5173"]
    return [o.strip().rstrip("/") for o in val.split(",") if o.strip()]

ALLOWED_ORIGINS = _parse_origins(os.getenv("ALLOWED_ORIGINS"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR / ".." / "data").resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

@app.middleware("http")
async def log_and_auth(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start = time.time()

    path = request.url.path

    if SECRAG_API_KEY:
        allowlist = {"/health", "/docs", "/openapi.json"}
        if path not in allowlist:
            incoming = request.headers.get("X-API-KEY", "")
            if incoming != SECRAG_API_KEY:
                elapsed = time.time() - start
                logger.info(json.dumps({
                    "request_id": request_id,
                    "method": request.method,
                    "path": path,
                    "status": 401,
                    "ms": int(elapsed * 1000),
                    "msg": "unauthorized",
                }))
                raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        response = await call_next(request)
        status = response.status_code
    except Exception as e:
        elapsed = time.time() - start
        logger.info(json.dumps({
            "request_id": request_id,
            "method": request.method,
            "path": path,
            "status": 500,
            "ms": int(elapsed * 1000),
            "error": str(e),
        }))
        raise

    elapsed = time.time() - start
    logger.info(json.dumps({
        "request_id": request_id,
        "method": request.method,
        "path": path,
        "status": status,
        "ms": int(elapsed * 1000),
    }))

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{elapsed:.4f}"
    return response

def normalize_pdf_filename(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="filename cannot be empty")
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name


def safe_file_write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def artifact_stats_for_pdf(pdf_name: str):
    chunk_path, emb_path = get_artifact_paths(pdf_name, DATA_DIR)

    if not chunk_path.exists() or not emb_path.exists():
        raise HTTPException(status_code=404, detail="Artifacts not found. Upload PDF first.")

    chunks = load_chunks(chunk_path)
    emb_size = emb_path.stat().st_size
    chunk_size = chunk_path.stat().st_size

    last_ingested_ts = max(chunk_path.stat().st_mtime, emb_path.stat().st_mtime)
    last_ingested = datetime.fromtimestamp(last_ingested_ts).isoformat()

    import numpy as np
    emb = np.load(emb_path)
    dim = int(emb.shape[1]) if emb.ndim == 2 else 0

    return {
        "filename": pdf_name,
        "total_chunks": len(chunks),
        "embedding_dim": dim,
        "artifacts": {
            "chunks_path": str(chunk_path),
            "chunks_bytes": int(chunk_size),
            "embedding_path": str(emb_path),
            "embedding_bytes": int(emb_size),
        },
        "last_ingested": last_ingested,
    }


@app.get("/health")
def health_check():
    return {"status": "SecRAG backend is running", "allowed_origins": ALLOWED_ORIGINS}


@app.get("/list_docs")
def list_docs():
    pdfs = [f.name for f in DATA_DIR.glob("*.pdf")]
    return {"documents": pdfs}


@app.get("/stats")
def stats(filename: str):
    pdf_name = normalize_pdf_filename(filename)
    return artifact_stats_for_pdf(pdf_name)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), chunk_strategy: str = "sentence"):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = await file.read()

    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max allowed is {MAX_UPLOAD_MB} MB."
        )

    pdf_path = DATA_DIR / file.filename

    try:
        safe_file_write(pdf_path, contents)
        return process_pdf_upload(str(pdf_path), str(DATA_DIR), chunk_strategy=chunk_strategy)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {e}")


class RetrieveRequest(BaseModel):
    filename: str
    query: str
    top_k: int = 5
    min_score: float | None = None
    mode: str = "hybrid"
    alpha: float = 0.7


@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    pdf_name = normalize_pdf_filename(req.filename)

    chunk_path, emb_path = get_artifact_paths(pdf_name, DATA_DIR)

    if not chunk_path.exists():
        raise HTTPException(status_code=404, detail="Chunks file not found. Upload PDF first.")
    if not emb_path.exists():
        raise HTTPException(status_code=404, detail="Embedding file not found. Upload PDF first.")

    try:
        results = retrieve_top_k(
            chunks_path=chunk_path,
            embeddings_path=emb_path,
            query=req.query,
            top_k=req.top_k,
            min_score=req.min_score,
            mode=req.mode,
            alpha=req.alpha
        )
        return {"filename": pdf_name, "query": req.query, "top_k": req.top_k, "mode": req.mode, "results": results}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")


class AnswerRequest(BaseModel):
    filename: str
    query: str
    top_k: int = 5
    min_score: float | None = None
    mode: str = "hybrid"
    alpha: float = 0.7


@app.post("/answer")
def answer(req: AnswerRequest):
    pdf_name = normalize_pdf_filename(req.filename)

    chunk_path, emb_path = get_artifact_paths(pdf_name, DATA_DIR)

    if not chunk_path.exists() or not emb_path.exists():
        raise HTTPException(status_code=404, detail="Artifacts not found. Upload PDF first.")

    try:
        retrieved = retrieve_top_k(
            query=req.query,
            pdf_name=pdf_name,
            top_k=req.top_k,
            min_score=req.min_score,
            mode=req.mode,
            use_reranker=True,
        )

        if not retrieved:
            return {"filename": pdf_name, "query": req.query, "answer": "No relevant context found.", "citations": []}

        answer_text = generate_answer(req.query, retrieved)

        from utils.citation_verifier import verify_citations
        verification = verify_citations(answer_text, retrieved)

        return {
            "filename": pdf_name,
            "query": req.query,
            "top_k": req.top_k,
            "mode": req.mode,
            "answer": answer_text,
            "verified_answer": verification["verified_answer"],
            "citation_accuracy": verification["citation_accuracy"],
            "citation_details": verification["citations"],
            "citations": [
                {
                    "chunk_id": c["chunk_id"],
                    "score": c["score"],
                    "char_range": [c["metadata"]["char_start"], c["metadata"]["char_end"]],
                }
                for c in retrieved
            ],
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SummarizeRequest(BaseModel):
    filename: str
    intro_chunks: int = 3
    top_k: int = 5
    max_output_tokens: int = 350
    min_score: float | None = None
    mode: str = "hybrid"
    alpha: float = 0.7


@app.post("/summarize")
def summarize(req: SummarizeRequest):
    pdf_name = normalize_pdf_filename(req.filename)

    if req.intro_chunks <= 0 or req.intro_chunks > 10:
        raise HTTPException(status_code=400, detail="intro_chunks must be between 1 and 10")
    if req.top_k <= 0 or req.top_k > 20:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 20")

    chunk_path, emb_path = get_artifact_paths(pdf_name, DATA_DIR)

    if not chunk_path.exists() or not emb_path.exists():
        raise HTTPException(status_code=404, detail="Artifacts not found. Upload PDF first.")

    try:
        all_chunks = load_chunks(chunk_path)
        intro = all_chunks[: req.intro_chunks]

        retrieved = retrieve_top_k(
            chunks_path=chunk_path,
            embeddings_path=emb_path,
            query="Summarize this document.",
            top_k=req.top_k,
            min_score=req.min_score,
            mode=req.mode,
            alpha=req.alpha
        )

        merged = {}

        for c in intro:
            cid = c.get("chunk_id")
            if cid is None:
                continue
            merged[cid] = {
                "chunk_id": cid,
                "content": c.get("content", ""),
                "score": 0.0,
                "metadata": {"char_start": c.get("char_start"), "char_end": c.get("char_end")},
                "source": "intro"
            }

        for r in retrieved:
            cid = r.get("chunk_id")
            if cid is None:
                continue
            score = float(r.get("score", 0.0))
            if cid in merged:
                if score > float(merged[cid].get("score", 0.0)):
                    merged[cid]["score"] = score
                merged[cid]["source"] = "hybrid"
            else:
                merged[cid] = {
                    "chunk_id": cid,
                    "content": r.get("content", ""),
                    "score": score,
                    "metadata": {
                        "char_start": r.get("metadata", {}).get("char_start"),
                        "char_end": r.get("metadata", {}).get("char_end"),
                    },
                    "source": "retrieved"
                }

        intro_ids = [c.get("chunk_id") for c in intro if c.get("chunk_id") is not None]
        retrieved_ids_sorted = sorted(
            [cid for cid in merged.keys() if cid not in intro_ids],
            key=lambda cid: float(merged[cid].get("score", 0.0)),
            reverse=True
        )

        final_ids = intro_ids + retrieved_ids_sorted
        final_chunks = [merged[cid] for cid in final_ids if cid in merged]

        if not final_chunks:
            return {"filename": pdf_name, "summary": "I do not know.", "citations": []}

        summary_text = summarize_from_chunks(
            filename=pdf_name,
            chunks=final_chunks,
            max_output_tokens=req.max_output_tokens
        )

        citations = [
            {
                "chunk_id": c["chunk_id"],
                "score": c.get("score", 0.0),
                "source": c.get("source", ""),
                "char_range": [c.get("metadata", {}).get("char_start"), c.get("metadata", {}).get("char_end")],
            }
            for c in final_chunks
        ]

        return {
            "filename": pdf_name,
            "intro_chunks": req.intro_chunks,
            "top_k": req.top_k,
            "mode": req.mode,
            "summary": summary_text,
            "citations": citations
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarize failed: {e}")

@app.delete("/documents/{filename}")
def delete_document(filename: str):
    pdf_name = safe_pdf_name(filename)

    paths = get_all_related_paths(pdf_name, DATA_DIR)

    deleted = []
    missing = []

    for p in paths:
        try:
            if p.exists() and p.is_file():
                p.unlink()
                deleted.append(p.name)
            else:
                missing.append(p.name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete {p.name}: {e}")

    return {
        "filename": pdf_name,
        "deleted": deleted,
        "missing": missing,
    }

