import os
import asyncio
import aiohttp
from typing import Optional
from .types import SimilarityMode

TOGETHER_API_BASE = "https://api.together.xyz/v1"

GENERATOR_MODEL  = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
EMBEDDING_MODEL  = "togethercomputer/m2-bert-80M-8k-retrieval"
JUDGE_MODEL      = "Qwen/Qwen2.5-72B-Instruct-Turbo"


def get_api_key() -> str:
    key = os.environ.get("TOGETHER_API_KEY")
    if not key:
        raise EnvironmentError(
            "TOGETHER_API_KEY environment variable not set.\n"
            "Get a key at https://api.together.xyz and run:\n"
            "  export TOGETHER_API_KEY=your_key_here"
        )
    return key


async def generate(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    session: Optional[aiohttp.ClientSession] = None,
) -> str:
    """Single LLM generation call. Returns response text."""
    api_key = get_api_key()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": GENERATOR_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    close_session = session is None
    if session is None:
        session = aiohttp.ClientSession()

    try:
        async with session.post(
            f"{TOGETHER_API_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["choices"][0]["message"]["content"].strip()
    finally:
        if close_session:
            await session.close()


async def get_embedding(text: str, session: aiohttp.ClientSession) -> list[float]:
    """Get text embedding from Together AI. Returns float vector."""
    api_key = get_api_key()
    async with session.post(
        f"{TOGETHER_API_BASE}/embeddings",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": EMBEDDING_MODEL, "input": text},
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()
        return data["data"][0]["embedding"]


async def judge_divergence(
    output_a: str,
    output_b: str,
    session: aiohttp.ClientSession,
) -> int:
    """
    Ask Qwen to rate semantic divergence between two outputs.
    Returns integer 0-10. 0 = identical meaning, 10 = completely different.
    Uses different model family from generator (Qwen vs Llama) for independence.
    """
    api_key = get_api_key()
    prompt = f"""You are a semantic equivalence evaluator.

Rate how semantically different these two responses are — not surface differences, but differences in meaning, recommendations, facts, or intent.

Response A:
{output_a}

Response B:
{output_b}

Return a single integer from 0 to 10. 0 = identical meaning. 10 = completely different. Return only the number."""

    async with session.post(
        f"{TOGETHER_API_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": JUDGE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 5,
            "temperature": 0,
        },
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()
        raw = data["choices"][0]["message"]["content"].strip()
        try:
            score = int(raw)
            return max(0, min(10, score))
        except ValueError:
            return 5  # neutral fallback
