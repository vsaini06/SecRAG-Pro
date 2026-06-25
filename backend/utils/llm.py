import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set.")
    return OpenAI(api_key=api_key)


def generate_answer(query: str, retrieved_chunks: list):
    client = _get_client()

    context = "\n\n".join(
        f"[Chunk {c['chunk_id']}]\n{c['content']}"
        for c in retrieved_chunks
    )

    system_prompt = """You are a precise document assistant.
Answer using ONLY the provided context chunks.
You MUST include inline citations like [Chunk 3] after EVERY factual claim.
Every sentence MUST end with [Chunk N] before the period.

Example of correct format:
"Vaibhav graduated in 2023 [Chunk 4]. His GPA was 3.96 [Chunk 2]."

Never write a sentence without a citation at the end.
Never answer without at least one citation.
If the answer is not in the context, say exactly: "I don't know based on the provided document." """

    user_prompt = f"""Context:
{context}

Question: {query}

Answer with inline [Chunk N] citations after every claim:"""

    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_output_tokens=500,
    )

    return response.output_text.strip()


def generate_sample_questions(filename: str, context: str, max_output_tokens: int = 220) -> list[str]:

    client = _get_client()

    prompt = f"""
You are helping build a RAG demo UI.

Generate 6 short, helpful sample questions a user can ask about this document.
Questions must be specific to the content and NOT generic.
Do not mention "chunks" or "context".
Do not include numbering like "1.".

Document: {filename}

Context:
{context}

Return ONLY a JSON array of strings.
Example:
["Question 1?", "Question 2?", "Question 3?"]
"""

    resp = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        max_output_tokens=max_output_tokens,
    )

    text = (resp.output_text or "").strip()

    try:
        arr = json.loads(text)
        if isinstance(arr, list):
            cleaned = []
            for x in arr:
                s = str(x).strip()
                if s:
                    cleaned.append(s)
            return cleaned[:6]
    except Exception:
        pass

    lines = []
    for line in text.splitlines():
        s = line.strip().lstrip("-•").strip()
        if not s:
            continue
        if len(s) > 2 and (s[0].isdigit() and s[1] in [".", ")", "-"]):
            s = s[2:].strip()
        if s:
            lines.append(s)

    return lines[:6]
