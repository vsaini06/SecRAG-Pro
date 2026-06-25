from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from utils.embeddings import embed_query
from utils.bm25 import build_bm25, bm25_scores
from utils.vector_store import query_collection

logger = logging.getLogger("secrag.retriever")



def load_chunks(chunks_path: Path) -> list[dict]:
    with open(chunks_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Chunks JSON must be a list")
    return data


def load_embeddings(embeddings_path: Path) -> np.ndarray:
    emb = np.load(embeddings_path).astype(np.float32)
    if emb.ndim != 2:
        raise ValueError("Embeddings must be 2D (num_chunks, dim)")
    return emb


def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx - mn < 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


def _build_results(chunks: list, indices: np.ndarray, scores: np.ndarray, min_score=None) -> list[dict]:
    results = []
    for idx in indices:
        sc = float(scores[int(idx)])
        if min_score is not None and sc < float(min_score):
            continue
        ch = chunks[int(idx)]
        results.append({
            "score": sc,
            "chunk_id": ch.get("chunk_id"),
            "filename": ch.get("filename"),
            "content": ch.get("content"),
            "metadata": {
                "source_path": ch.get("source_path"),
                "created_at": ch.get("created_at"),
                "char_start": ch.get("char_start"),
                "char_end": ch.get("char_end"),
            },
        })
    return results



def _rrf_fuse(dense_ranked: list[dict], sparse_ranked: list[dict], k: int = 60) -> list[dict]:

    scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for rank, item in enumerate(dense_ranked, start=1):
        cid = str(item["chunk_id"])
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        chunk_map[cid] = item

    for rank, item in enumerate(sparse_ranked, start=1):
        cid = str(item["chunk_id"])
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        if cid not in chunk_map:
            chunk_map[cid] = item

    fused = []
    for cid, rrf_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        item = dict(chunk_map[cid])
        item["score"] = round(rrf_score, 6)
        item["fusion_method"] = "rrf"
        fused.append(item)

    return fused


def retrieve_top_k(
    query: str,
    pdf_name: str | None = None,
    top_k: int = 5,
    min_score: float | None = None,
    mode: str = "hybrid",
    alpha: float = 0.7,
    use_reranker: bool = True,
    chroma_dir: str = "./data/chroma",
    chunks_path: Path | None = None,
    embeddings_path: Path | None = None,
    candidate_mult: int = 4,
) -> list[dict]:

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    mode = (mode or "hybrid").lower().strip()
    if mode not in {"hybrid", "semantic", "bm25"}:
        raise ValueError("mode must be one of: hybrid, semantic, bm25")

    if pdf_name is not None:
        return _retrieve_chromadb(
            query=query,
            pdf_name=pdf_name,
            top_k=top_k,
            min_score=min_score,
            mode=mode,
            use_reranker=use_reranker,
            candidate_mult=candidate_mult,
            chroma_dir=chroma_dir,
        )

    if chunks_path is None or embeddings_path is None:
        raise ValueError("Either pdf_name or both chunks_path and embeddings_path must be provided")
    return _retrieve_legacy(
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        query=query,
        top_k=top_k,
        min_score=min_score,
        mode=mode,
        alpha=alpha,
        candidate_mult=candidate_mult,
    )


def _retrieve_chromadb(
    query: str,
    pdf_name: str,
    top_k: int,
    min_score: float | None,
    mode: str,
    use_reranker: bool,
    candidate_mult: int,
    chroma_dir: str = "./data/chroma",
) -> list[dict]:
    query_vec = embed_query(query)
    candidate_k = top_k * candidate_mult

    dense_results: list[dict] = []
    sparse_results: list[dict] = []

    if mode in {"semantic", "hybrid"}:
        dense_results = query_collection(pdf_name, query_vec, top_k=candidate_k, persist_dir=chroma_dir)

    if mode in {"bm25", "hybrid"}:
        if dense_results:
            bm25_corpus = dense_results
        else:
            bm25_corpus = query_collection(pdf_name, query_vec, top_k=min(200, candidate_k * 5), persist_dir=chroma_dir)

        if bm25_corpus:
            bm25 = build_bm25(bm25_corpus)
            bm25_raw = np.array(bm25_scores(bm25, query), dtype=np.float32)
            bm25_norm = _minmax_norm(bm25_raw)
            sparse_results = [
                {**bm25_corpus[i], "score": float(bm25_norm[i])}
                for i in np.argsort(-bm25_norm)[:candidate_k]
            ]

 
    if mode == "semantic":
        candidates = dense_results[:candidate_k]
    elif mode == "bm25":
        candidates = sparse_results[:candidate_k]
    else:
        candidates = _rrf_fuse(dense_results, sparse_results)[:candidate_k]

    if min_score is not None:
        candidates = [c for c in candidates if c["score"] >= min_score]

    if use_reranker and candidates:
        try:
            from utils.reranker import rerank
            candidates = rerank(query, candidates, top_k=top_k)
        except Exception as e:
            logger.warning(f"Reranker failed ({e}), using fusion order")
            candidates = candidates[:top_k]
    else:
        candidates = candidates[:top_k]

    return candidates


def _retrieve_legacy(
    chunks_path: Path,
    embeddings_path: Path,
    query: str,
    top_k: int,
    min_score,
    mode: str,
    alpha: float,
    candidate_mult: int,
) -> list[dict]:
    chunks = load_chunks(chunks_path)
    n = len(chunks)
    if n == 0:
        return []

    bm25_norm = None
    if mode in {"bm25", "hybrid"}:
        bm25 = build_bm25(chunks)
        bm25_raw = np.array(bm25_scores(bm25, query), dtype=np.float32)
        bm25_norm = _minmax_norm(bm25_raw)

    emb_norm = None
    if mode in {"semantic", "hybrid"}:
        embeddings = load_embeddings(embeddings_path)
        if n != embeddings.shape[0]:
            raise ValueError("Mismatch: chunks count != embeddings rows")
        q = embed_query(query)
        emb_scores = (embeddings @ q).astype(np.float32)
        emb_norm = ((emb_scores + 1.0) / 2.0).clip(0.0, 1.0).astype(np.float32)

    if mode == "semantic":
        k = min(top_k, n)
        idx = np.argpartition(-emb_norm, k - 1)[:k]
        idx = idx[np.argsort(-emb_norm[idx])]
        return _build_results(chunks, idx, emb_norm, min_score=min_score)

    if mode == "bm25":
        k = min(top_k, n)
        idx = np.argpartition(-bm25_norm, k - 1)[:k]
        idx = idx[np.argsort(-bm25_norm[idx])]
        return _build_results(chunks, idx, bm25_norm, min_score=min_score)

    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be between 0 and 1")
    k_candidates = min(n, top_k * candidate_mult)
    idx_emb = np.argpartition(-emb_norm, k_candidates - 1)[:k_candidates]
    idx_bm = np.argpartition(-bm25_norm, k_candidates - 1)[:k_candidates]
    candidate_set = set(map(int, idx_emb)) | set(map(int, idx_bm))
    candidate_idx = np.array(list(candidate_set), dtype=np.int32)
    final_scores = alpha * emb_norm[candidate_idx] + (1.0 - alpha) * bm25_norm[candidate_idx]
    order = np.argsort(-final_scores)
    chosen = candidate_idx[order][: min(top_k, len(order))]
    final_full = np.zeros((n,), dtype=np.float32)
    final_full[candidate_idx] = final_scores.astype(np.float32)
    return _build_results(chunks, chosen, final_full, min_score=min_score)
