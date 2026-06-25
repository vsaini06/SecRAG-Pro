from rank_bm25 import BM25Okapi


def tokenize(text: str):
    return [t for t in text.lower().split() if t.strip()]


def build_bm25(chunks: list):
    tokenized_corpus = [tokenize(c.get("content", "")) for c in chunks]
    return BM25Okapi(tokenized_corpus)


def bm25_scores(bm25: BM25Okapi, query: str):
    return bm25.get_scores(tokenize(query))