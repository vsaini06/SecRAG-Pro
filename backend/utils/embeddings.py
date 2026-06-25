from sentence_transformers import SentenceTransformer
import numpy as np

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_texts(texts):
    model = get_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return vectors.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    model = get_model()
    vec = model.encode([query], normalize_embeddings=True)
    return vec[0].astype(np.float32)
