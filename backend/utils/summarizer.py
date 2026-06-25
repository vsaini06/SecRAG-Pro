import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def summarize_from_chunks(filename: str, chunks: list, max_output_tokens: int = 350) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set.")

    client = OpenAI(api_key=api_key)

    context = "\n\n".join(
        f"[Chunk {c.get('chunk_id')} | Score {round(float(c.get('score', 0.0)), 3)}]\n{c.get('content','')}"
        for c in chunks
    )

    prompt = f"""
You are a helpful assistant.
Summarize the document using ONLY the provided context.
If the context is not enough, say "I do not know."

Document: {filename}

Context:
{context}

Output format:
- 6–10 bullet points of key ideas
- Then 1 short paragraph overview
"""

    resp = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        max_output_tokens=max_output_tokens,
    )

    return resp.output_text.strip()