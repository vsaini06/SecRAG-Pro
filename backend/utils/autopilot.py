"""
LLM Cost Autopilot client.
Replaces direct OpenAI calls with routed requests through the Cost Autopilot API.
Falls back to direct OpenAI if the Autopilot is unreachable.
"""

import os
import httpx
from openai import OpenAI

AUTOPILOT_URL = os.getenv("AUTOPILOT_URL", "http://localhost:8000/v1/completions")
AUTOPILOT_TIMEOUT = float(os.getenv("AUTOPILOT_TIMEOUT", "120"))
USE_AUTOPILOT = os.getenv("USE_AUTOPILOT", "true").lower() == "true"


async def _route_async(prompt: str, max_tokens: int = 1024) -> dict:
    async with httpx.AsyncClient(timeout=AUTOPILOT_TIMEOUT) as client:
        response = await client.post(
            AUTOPILOT_URL,
            json={"prompt": prompt, "max_tokens": max_tokens},
        )
        response.raise_for_status()
        return response.json()


def route(prompt: str, max_tokens: int = 1024) -> dict:
    """
    Send a prompt to the Cost Autopilot router.
    Returns the full response dict including answer, model_used, cost, cost_saved etc.
    Falls back to direct GPT-4o-mini if Autopilot is unreachable.
    """
    if not USE_AUTOPILOT:
        return _fallback_direct(prompt, max_tokens)

    try:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError
            result = loop.run_until_complete(_route_async(prompt, max_tokens))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(_route_async(prompt, max_tokens))
        return result
    except Exception as e:
        print(f"[Autopilot] Unreachable ({e}), falling back to direct OpenAI")
        return _fallback_direct(prompt, max_tokens)


def _fallback_direct(prompt: str, max_tokens: int) -> dict:
    """Direct OpenAI call used when Autopilot is down."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return {
        "answer": response.choices[0].message.content.strip(),
        "model_used": "gpt-4o-mini (fallback)",
        "complexity_tier": None,
        "cost": 0.0,
        "cost_if_gpt4o": 0.0,
        "cost_saved": 0.0,
        "latency_ms": 0,
        "escalated": False,
    }