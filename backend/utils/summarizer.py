import os
from dotenv import load_dotenv

load_dotenv()


def summarize_from_chunks(filename: str, chunks: list, max_output_tokens: int = 350) -> dict:
    from utils.autopilot import route

    context = "\n\n".join(
        f"[Chunk {c.get('chunk_id')} | Score {round(float(c.get('score', 0.0)), 3)}]\n{c.get('content','')}"
        for c in chunks
    )

    prompt = f"""You are a helpful assistant.
Summarize the document using ONLY the provided context.
If the context is not enough, say "I do not know."

Document: {filename}

Context:
{context}

Output format:
- 6-10 bullet points of key ideas
- Then 1 short paragraph overview
"""

    result = route(prompt, max_tokens=max_output_tokens)

    return {
        "summary": result.get("answer", ""),
        "routing": {
            "model_used": result.get("model_used"),
            "complexity_tier": result.get("complexity_tier"),
            "classifier_confidence": result.get("classifier_confidence"),
            "cost": result.get("cost"),
            "cost_if_gpt4o": result.get("cost_if_gpt4o"),
            "cost_saved": result.get("cost_saved"),
            "latency_ms": result.get("latency_ms"),
            "escalated": result.get("escalated"),
        },
    }