"""
LLM provider abstraction with capability protocols.

Providers:
- TogetherProvider: full capability (generation + embedding + judge)
- OpenAIProvider: generation + embedding + judge (via generation)
- AnthropicProvider: generation only
"""

import asyncio
import os
import json
from typing import Protocol, Optional, runtime_checkable

import aiohttp

from .resilience import with_retry, RetryConfig, DEFAULT_CONFIG


# Default models (same as original generator.py constants)
GENERATOR_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
EMBEDDING_MODEL = "togethercomputer/m2-bert-80M-8k-retrieval"
JUDGE_MODEL = "Qwen/Qwen2.5-72B-Instruct-Turbo"

JUDGE_PROMPT_TEMPLATE = """You are a semantic equivalence evaluator.

Rate how semantically different these two responses are — not surface differences, but differences in meaning, recommendations, facts, or intent.

Response A:
{output_a}

Response B:
{output_b}

Return a single integer from 0 to 10. 0 = identical meaning. 10 = completely different. Return only the number."""


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class GenerationProvider(Protocol):
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> str: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    async def get_embedding(
        self, text: str, session: aiohttp.ClientSession
    ) -> list[float]: ...


@runtime_checkable
class JudgeProvider(Protocol):
    async def judge_divergence(
        self, output_a: str, output_b: str, session: aiohttp.ClientSession
    ) -> int: ...


class LLMProvider(GenerationProvider, EmbeddingProvider, JudgeProvider, Protocol):
    """Full-capability provider (generation + embedding + judge)."""

    pass


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


class TogetherProvider:
    """Full-capability provider using Together AI APIs."""

    def __init__(
        self,
        api_key: str | None = None,
        generator_model: str = GENERATOR_MODEL,
        embedding_model: str = EMBEDDING_MODEL,
        judge_model: str = JUDGE_MODEL,
        retry_config: RetryConfig = DEFAULT_CONFIG,
    ):
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY", "")
        self.generator_model = generator_model
        self.embedding_model = embedding_model
        self.judge_model = judge_model
        self.base_url = "https://api.together.xyz/v1"
        self._retry_config = retry_config

    def _get_api_key(self) -> str:
        if not self.api_key:
            raise EnvironmentError(
                "TOGETHER_API_KEY environment variable not set.\n"
                "Get a key at https://api.together.xyz and run:\n"
                "  export TOGETHER_API_KEY=your_key_here"
            )
        return self.api_key

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> str:
        api_key = self._get_api_key()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.generator_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        close_session = session is None
        if session is None:
            session = aiohttp.ClientSession()

        try:
            async def _do_request():
                async with asyncio.timeout(self._retry_config.timeout):
                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    ) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        return data["choices"][0]["message"]["content"].strip()

            return await with_retry(_do_request, config=self._retry_config, operation_name="TogetherProvider.generate")
        finally:
            if close_session:
                await session.close()

    async def get_embedding(
        self, text: str, session: aiohttp.ClientSession
    ) -> list[float]:
        api_key = self._get_api_key()

        async def _do_request():
            async with asyncio.timeout(self._retry_config.timeout):
                async with session.post(
                    f"{self.base_url}/embeddings",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"model": self.embedding_model, "input": text},
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    return data["data"][0]["embedding"]

        return await with_retry(_do_request, config=self._retry_config, operation_name="TogetherProvider.get_embedding")

    async def judge_divergence(
        self, output_a: str, output_b: str, session: aiohttp.ClientSession
    ) -> int:
        api_key = self._get_api_key()
        prompt = JUDGE_PROMPT_TEMPLATE.format(output_a=output_a, output_b=output_b)

        async def _do_request():
            async with asyncio.timeout(self._retry_config.timeout):
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.judge_model,
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

        return await with_retry(_do_request, config=self._retry_config, operation_name="TogetherProvider.judge_divergence")


class OpenAIProvider:
    """Provider using OpenAI APIs. Supports generation, embedding, and judge (via generation)."""

    def __init__(
        self,
        api_key: str | None = None,
        generator_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        retry_config: RetryConfig = DEFAULT_CONFIG,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.generator_model = generator_model
        self.embedding_model = embedding_model
        self.base_url = "https://api.openai.com/v1"
        self._retry_config = retry_config

    def _get_api_key(self) -> str:
        if not self.api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable not set.\n"
                "Get a key at https://platform.openai.com and run:\n"
                "  export OPENAI_API_KEY=your_key_here"
            )
        return self.api_key

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> str:
        api_key = self._get_api_key()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.generator_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        close_session = session is None
        if session is None:
            session = aiohttp.ClientSession()

        try:
            async def _do_request():
                async with asyncio.timeout(self._retry_config.timeout):
                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    ) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        return data["choices"][0]["message"]["content"].strip()

            return await with_retry(_do_request, config=self._retry_config, operation_name="OpenAIProvider.generate")
        finally:
            if close_session:
                await session.close()

    async def get_embedding(
        self, text: str, session: aiohttp.ClientSession
    ) -> list[float]:
        api_key = self._get_api_key()

        async def _do_request():
            async with asyncio.timeout(self._retry_config.timeout):
                async with session.post(
                    f"{self.base_url}/embeddings",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"model": self.embedding_model, "input": text},
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    return data["data"][0]["embedding"]

        return await with_retry(_do_request, config=self._retry_config, operation_name="OpenAIProvider.get_embedding")

    async def judge_divergence(
        self, output_a: str, output_b: str, session: aiohttp.ClientSession
    ) -> int:
        api_key = self._get_api_key()
        prompt = JUDGE_PROMPT_TEMPLATE.format(output_a=output_a, output_b=output_b)

        async def _do_request():
            async with asyncio.timeout(self._retry_config.timeout):
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.generator_model,
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

        return await with_retry(_do_request, config=self._retry_config, operation_name="OpenAIProvider.judge_divergence")


class AnthropicProvider:
    """Provider using Anthropic Messages API. Supports generation only."""

    def __init__(
        self,
        api_key: str | None = None,
        generator_model: str = "claude-sonnet-4-20250514",
        retry_config: RetryConfig = DEFAULT_CONFIG,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.generator_model = generator_model
        self.base_url = "https://api.anthropic.com/v1"
        self._retry_config = retry_config

    def _get_api_key(self) -> str:
        if not self.api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable not set.\n"
                "Get a key at https://console.anthropic.com and run:\n"
                "  export ANTHROPIC_API_KEY=your_key_here"
            )
        return self.api_key

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> str:
        api_key = self._get_api_key()
        messages = [{"role": "user", "content": prompt}]

        payload: dict = {
            "model": self.generator_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            payload["system"] = system_prompt

        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        close_session = session is None
        if session is None:
            session = aiohttp.ClientSession()

        try:
            async def _do_request():
                async with asyncio.timeout(self._retry_config.timeout):
                    async with session.post(
                        f"{self.base_url}/messages",
                        headers=headers,
                        json=payload,
                    ) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        return data["content"][0]["text"].strip()

            return await with_retry(_do_request, config=self._retry_config, operation_name="AnthropicProvider.generate")
        finally:
            if close_session:
                await session.close()

    async def get_embedding(
        self, text: str, session: aiohttp.ClientSession
    ) -> list[float]:
        raise NotImplementedError(
            "AnthropicProvider does not support embeddings. "
            "Use --provider together or --provider openai for semantic mode."
        )

    async def judge_divergence(
        self, output_a: str, output_b: str, session: aiohttp.ClientSession
    ) -> int:
        raise NotImplementedError(
            "AnthropicProvider does not support judge divergence. "
            "Use --provider together or --provider openai for judge mode."
        )


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

_provider: GenerationProvider | None = None
_embedding_provider: EmbeddingProvider | None = None
_judge_provider: JudgeProvider | None = None


def configure_provider(
    provider: GenerationProvider,
    embedding_provider: EmbeddingProvider | None = None,
    judge_provider: JudgeProvider | None = None,
) -> None:
    """Configure the active LLM provider(s).

    Args:
        provider: Primary provider for generation.
        embedding_provider: Override for embedding calls. Falls back to provider
            if it implements EmbeddingProvider.
        judge_provider: Override for judge calls. Falls back to provider
            if it implements JudgeProvider.
    """
    global _provider, _embedding_provider, _judge_provider
    _provider = provider
    _embedding_provider = embedding_provider or (
        provider if isinstance(provider, EmbeddingProvider) else None
    )
    _judge_provider = judge_provider or (
        provider if isinstance(provider, JudgeProvider) else None
    )


def get_configured_provider() -> GenerationProvider:
    """Return current generation provider, defaulting to TogetherProvider."""
    global _provider
    if _provider is None:
        _provider = TogetherProvider()
    return _provider


def get_configured_embedding_provider() -> EmbeddingProvider:
    """Return current embedding provider with validation."""
    global _embedding_provider
    if _embedding_provider is None:
        _embedding_provider = TogetherProvider()
    if not isinstance(_embedding_provider, EmbeddingProvider):
        raise RuntimeError(
            f"Semantic mode requires a provider with embedding support. "
            f"Current provider ({type(_embedding_provider).__name__}) does not support embeddings."
        )
    return _embedding_provider


def get_configured_judge_provider() -> JudgeProvider:
    """Return current judge provider with validation."""
    global _judge_provider
    if _judge_provider is None:
        _judge_provider = TogetherProvider()
    if not isinstance(_judge_provider, JudgeProvider):
        raise RuntimeError(
            f"Judge mode requires a provider with judge support. "
            f"Current provider ({type(_judge_provider).__name__}) does not support judging."
        )
    return _judge_provider
