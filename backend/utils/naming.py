from pathlib import Path

def safe_pdf_name(filename: str) -> str:
    name = Path(filename).name
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name

def get_artifact_paths(filename: str, data_dir: Path):
    stem = Path(filename).stem
    chunk_path = data_dir / f"{stem}_chunks.json"
    embedding_path = data_dir / f"{stem}_embedding.npy"
    return chunk_path, embedding_path

def get_all_related_paths(filename: str, data_dir: Path) -> list[Path]:

    pdf_name = safe_pdf_name(filename)
    stem = Path(pdf_name).stem

    candidates = [
        data_dir / pdf_name,
        data_dir / f"{stem}.txt",
        data_dir / f"{stem}_chunks.json",
        data_dir / f"{stem}_embedding.npy",

        data_dir / f"{stem}_meta.json",
        data_dir / f"{stem}_stats.json",
    ]
    return candidates
