import hashlib
from pathlib import Path

import chromadb
import numpy as np

_client: chromadb.ClientAPI | None = None


def _get_client(persist_dir: str = "./data/chroma") -> chromadb.Client:
    global _client
    if _client is None:
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=persist_dir)
    return _client


def _collection_name(pdf_name: str) -> str:
    stem = Path(pdf_name).stem.lower()
    safe = "".join(c if c.isalnum() else "_" for c in stem)[:60]
    return f"secrag_{safe}"


def collection_exists(pdf_name: str, persist_dir: str = "./data/chroma") -> bool:
    client = _get_client(persist_dir)
    try:
        client.get_collection(_collection_name(pdf_name))
        return True
    except Exception:
        return False


def get_collection(pdf_name: str, persist_dir: str = "./data/chroma"):
    client = _get_client(persist_dir)
    return client.get_or_create_collection(
        name=_collection_name(pdf_name),
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(
    pdf_name: str,
    chunk_data: list[dict],
    vectors: np.ndarray,
    persist_dir: str = "./data/chroma",
):

    collection = get_collection(pdf_name, persist_dir)

    ids, embeddings, documents, metadatas = [], [], [], []

    for idx, (chunk, vec) in enumerate(zip(chunk_data, vectors)):
        if near_duplicate_exists_vec(collection, vec, threshold=0.95):
            continue

        chunk_id = str(chunk.get("chunk_id", idx))
        ids.append(chunk_id)
        embeddings.append(vec.tolist())
        documents.append(chunk.get("content", ""))
        metadatas.append({
            "filename": chunk.get("filename", pdf_name),
            "source_path": chunk.get("source_path", ""),
            "created_at": chunk.get("created_at", ""),
            "char_start": chunk.get("char_start", 0),
            "char_end": chunk.get("char_end", 0),
            "chunk_strategy": chunk.get("chunk_strategy", "sentence"),
        })

    if ids:
        collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    return len(ids)


def query_collection(
    pdf_name: str,
    query_vec: np.ndarray,
    top_k: int = 20,
    persist_dir: str = "./data/chroma",
) -> list[dict]:

    collection = get_collection(pdf_name, persist_dir)

    count = collection.count()
    if count == 0:
        return []

    k = min(top_k, count)
    result = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    out = []
    for i, (doc, meta, dist) in enumerate(zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    )):
        score = float(1.0 - dist / 2.0)
        out.append({
            "score": score,
            "chunk_id": result["ids"][0][i],
            "filename": meta.get("filename", pdf_name),
            "content": doc,
            "metadata": {
                "source_path": meta.get("source_path", ""),
                "created_at": meta.get("created_at", ""),
                "char_start": meta.get("char_start", 0),
                "char_end": meta.get("char_end", 0),
                "chunk_strategy": meta.get("chunk_strategy", "sentence"),
            },
        })
    return out


def near_duplicate_exists(
    pdf_name: str,
    vector: np.ndarray,
    threshold: float = 0.95,
    persist_dir: str = "./data/chroma",
) -> bool:
    collection = get_collection(pdf_name, persist_dir)
    return near_duplicate_exists_vec(collection, vector, threshold)


def near_duplicate_exists_vec(collection, vector: np.ndarray, threshold: float = 0.95) -> bool:
    if collection.count() == 0:
        return False
    result = collection.query(
        query_embeddings=[vector.tolist()],
        n_results=1,
        include=["distances"],
    )
    if not result["distances"] or not result["distances"][0]:
        return False
    dist = result["distances"][0][0]
    similarity = 1.0 - dist / 2.0
    return similarity >= threshold


def delete_collection(pdf_name: str, persist_dir: str = "./data/chroma"):
    client = _get_client(persist_dir)
    try:
        client.delete_collection(_collection_name(pdf_name))
    except Exception:
        pass
