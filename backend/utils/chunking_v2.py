import re
from typing import List, Tuple


_SENTENCE_RE = re.compile(r".+?(?:[.!?]+(?=\s|$)|$)", re.DOTALL)


def split_sentences_with_spans(text: str) -> List[Tuple[int, int, str]]:
    spans = []
    for m in _SENTENCE_RE.finditer(text):
        s = m.group(0)
        if not s or not s.strip():
            continue
        start, end = m.start(), m.end()
        spans.append((start, end, s.strip()))
    return spans


def chunk_text_sentence_based(
    text: str,
    chunk_size: int = 500,
    overlap_sentences: int = 1
) -> List[Tuple[int, int, str]]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences must be >= 0")

    sents = split_sentences_with_spans(text)
    if not sents:
        return []

    chunks = []
    i = 0
    n = len(sents)

    while i < n:
        start = sents[i][0]
        current_text_parts = []
        j = i

        while j < n:
            candidate = (" ".join(current_text_parts + [sents[j][2]])).strip()
            if len(candidate) <= chunk_size or not current_text_parts:
                current_text_parts.append(sents[j][2])
                j += 1
            else:
                break

        end = sents[j - 1][1]
        chunk_text = " ".join(current_text_parts).strip()
        chunks.append((start, end, chunk_text))

        if j >= n:
            break

        i = max(j - overlap_sentences, i + 1)

    return chunks
