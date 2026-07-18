import os
import asyncio
import aiohttp
from typing import Optional
from .types import SimilarityMode
from .provider import (
    GenerationProvider,
    EmbeddingProvider,
    JudgeProvider,
    TogetherProvider,
    configure_provider,
    get_configured_provider,
    get_configured_embedding_provider,
    get_configured_judge_provider,
)

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
    provider = get_configured_provider()
    return await provider.generate(prompt, system_prompt, temperature, max_tokens, session)


async def get_embedding(text: str, session: aiohttp.ClientSession) -> list[float]:
    """Get text embedding. Returns float vector."""
    provider = get_configured_embedding_provider()
    return await provider.get_embedding(text, session)


async def judge_divergence(
    output_a: str,
    output_b: str,
    session: aiohttp.ClientSession,
) -> int:
    """
    Ask a judge model to rate semantic divergence between two outputs.
    Returns integer 0-10. 0 = identical meaning, 10 = completely different.
    """
    provider = get_configured_judge_provider()
    return await provider.judge_divergence(output_a, output_b, session)
