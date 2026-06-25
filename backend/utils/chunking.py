def chunk_text(text, chunk_size=500, overlap=100):
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append((start, end, chunk))
        start += chunk_size - overlap

    return chunks
