from __future__ import annotations

import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path

from pypdf import PdfReader

from utils.chunking_strategies import chunk_text
from utils.embeddings import embed_texts
from utils.vector_store import upsert_chunks, collection_exists


def process_pdf_upload(
    file_path: str,
    data_dir: str,
    chunk_strategy: str = "sentence",
    chunk_size: int = 500,
    chroma_dir: str | None = None,
):

    file_path = Path(file_path)
    data_dir = Path(data_dir)
    chroma_dir = chroma_dir or str(data_dir / "chroma")

    reader = PdfReader(str(file_path))
    extracted_text = ""
    for page in reader.pages:
        extracted_text += page.extract_text() or ""

    text_path = data_dir / (file_path.stem + ".txt")
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(extracted_text, encoding="utf-8")

    raw_chunks = chunk_text(extracted_text, strategy=chunk_strategy, chunk_size=chunk_size)
    created_at = datetime.utcnow().isoformat()
    pdf_name = file_path.name

    chunk_data = []
    for index, (char_start, char_end, chunk_content, strategy_name) in enumerate(raw_chunks):
        chunk_data.append({
            "chunk_id": index,
            "filename": pdf_name,
            "source_path": str(file_path),
            "created_at": created_at,
            "char_start": char_start,
            "char_end": char_end,
            "content": chunk_content,
            "chunk_strategy": strategy_name,
        })


    texts = [c["content"] for c in chunk_data]
    vectors = embed_texts(texts)

    inserted_count = upsert_chunks(
        pdf_name=pdf_name,
        chunk_data=chunk_data,
        vectors=vectors,
        persist_dir=chroma_dir,
    )
    dedup_skipped = len(chunk_data) - inserted_count

    chunk_filename = file_path.stem + "_chunks.json"
    chunk_path = data_dir / chunk_filename
    with open(chunk_path, "w", encoding="utf-8") as f:
        json.dump(chunk_data, f, indent=2)

    emb_filename = file_path.stem + "_embedding.npy"
    emb_path = data_dir / emb_filename
    np.save(str(emb_path), vectors)

    return {
        "filename": pdf_name,
        "total_characters": len(extracted_text),
        "total_chunks_raw": len(chunk_data),
        "chunks_inserted_to_chroma": inserted_count,
        "chunks_dedup_skipped": dedup_skipped,
        "chunk_strategy": chunk_strategy,
        "embedding_dim": int(vectors.shape[1]) if vectors.ndim == 2 else 0,
        "first_chunk_preview": chunk_data[0]["content"][:200] if chunk_data else "",
        "chroma_dir": chroma_dir,
    }
