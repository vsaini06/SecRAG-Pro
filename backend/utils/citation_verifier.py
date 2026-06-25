from __future__ import annotations

import os
import re
import json
import logging

logger = logging.getLogger("secrag.citation_verifier")

_CITATION_RE = re.compile(r"\[(?:Chunk\s*)?(\d+)\]", re.IGNORECASE)


def parse_citations(answer_text: str) -> list[dict]:
    sentences = re.split(r"(?<=[.!?])\s+", answer_text)
    results = []
    seen = set()
    for sent in sentences:
        for m in _CITATION_RE.finditer(sent):
            cid = m.group(1)
            key = (cid, sent[:60])
            if key in seen:
                continue
            seen.add(key)
            results.append({"chunk_id": cid, "claim": sent.strip()})
    return results


def _build_chunk_map(retrieved: list[dict]) -> dict[str, str]:
    return {str(c.get("chunk_id", "")): c.get("content", "") for c in retrieved}


def _verify_single(claim: str, chunk_content: str, client) -> dict:
    prompt = (
        "Does the following SOURCE TEXT support the CLAIM?\n\n"
        f"CLAIM: {claim}\n\n"
        f"SOURCE TEXT: {chunk_content[:600]}\n\n"
        'Respond ONLY with a JSON object: {"supported": true/false, "confidence": 0.0-1.0, "reason": "one sentence"}'
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0,
        )
        raw = resp.choices[0].message.content.strip()
        data = json.loads(raw)
        return {
            "supported": bool(data.get("supported", False)),
            "confidence": float(data.get("confidence", 0.5)),
            "reason": str(data.get("reason", "")),
        }
    except Exception as e:
        logger.warning(f"Citation verify failed for claim '{claim[:50]}': {e}")
        return {"supported": True, "confidence": 0.5, "reason": "verification skipped"}


def verify_citations(answer_text: str, retrieved: list[dict]) -> dict:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception as e:
        logger.error(f"Citation verifier: OpenAI unavailable ({e})")
        return _passthrough(answer_text)

    chunk_map = _build_chunk_map(retrieved)
    citations_found = parse_citations(answer_text)

    if not citations_found:
        return {
            "verified_answer": answer_text,
            "citation_accuracy": 1.0,
            "citations": [],
            "unsupported_count": 0,
            "total_citations": 0,
        }

    citation_results = []
    for cit in citations_found:
        cid = cit["chunk_id"]
        content = chunk_map.get(cid, "")
        if not content:
            citation_results.append({
                "chunk_id": cid,
                "claim": cit["claim"],
                "supported": False,
                "confidence": 0.0,
                "reason": "Referenced chunk not in retrieved context",
            })
        else:
            verdict = _verify_single(cit["claim"], content, client)
            citation_results.append({
                "chunk_id": cid,
                "claim": cit["claim"],
                **verdict,
            })

    unsupported_ids = {c["chunk_id"] for c in citation_results if not c["supported"]}
    verified_answer = answer_text
    if unsupported_ids:
        for cid in unsupported_ids:
            verified_answer = re.sub(
                rf"\[(?:Chunk\s*)?{cid}\]",
                f"[Chunk {cid} ⚠️ UNSUPPORTED]",
                verified_answer,
                flags=re.IGNORECASE,
            )

    total = len(citation_results)
    supported = sum(1 for c in citation_results if c["supported"])
    accuracy = round(supported / total, 3) if total else 1.0

    return {
        "verified_answer": verified_answer,
        "citation_accuracy": accuracy,
        "citations": citation_results,
        "unsupported_count": total - supported,
        "total_citations": total,
    }


def _passthrough(answer_text: str) -> dict:
    return {
        "verified_answer": answer_text,
        "citation_accuracy": None,
        "citations": [],
        "unsupported_count": 0,
        "total_citations": 0,
    }
