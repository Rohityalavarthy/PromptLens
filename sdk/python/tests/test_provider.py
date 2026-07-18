"""
Tests for the provider abstraction layer.
"""
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from contextlib import asynccontextmanager

import aiohttp

from promptlens.provider import (
    GenerationProvider,
    EmbeddingProvider,
    JudgeProvider,
    LLMProvider,
    TogetherProvider,
    OpenAIProvider,
    AnthropicProvider,
    configure_provider,
    get_configured_provider,
    get_configured_embedding_provider,
    get_configured_judge_provider,
)
from promptlens import generator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeResponse:
    """Fake aiohttp response context manager for testing."""

    def __init__(self, data: dict, status: int = 200):
        self._data = data
        self.status = status

    async def json(self):
        return self._data

    def raise_for_status(self):
        if self.status >= 400:
            raise aiohttp.ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=self.status,
            )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class FakeContextManager:
    """Wraps FakeResponse to act as both a context manager and awaitable return."""

    def __init__(self, response: FakeResponse):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *args):
        pass


def make_chat_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def make_embedding_response(embedding: list[float]) -> dict:
    return {"data": [{"embedding": embedding}]}


def make_mock_session(response_data: dict, captured: dict | None = None):
    """Create a mock session whose .post() returns an async context manager."""
    if captured is None:
        captured = {}

    fake_resp = FakeResponse(response_data)

    def mock_post(url, **kwargs):
        captured.update(kwargs)
        captured["url"] = url
        return FakeContextManager(fake_resp)

    session = MagicMock()
    session.post = mock_post
    return session, captured


def reset_provider_state():
    """Reset module-level provider state between tests."""
    import promptlens.provider as pmod
    pmod._provider = None
    pmod._embedding_provider = None
    pmod._judge_provider = None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_providers():
    """Reset provider state before each test."""
    reset_provider_state()
    yield
    reset_provider_state()


@pytest.mark.asyncio
async def test_together_provider_builds_correct_payload():
    """TogetherProvider sends correct model and messages format."""
    provider = TogetherProvider(api_key="test-key")

    session, captured = make_mock_session(make_chat_response("hello world"))

    result = await provider.generate(
        "test prompt", system_prompt="be helpful", temperature=0.5, max_tokens=512, session=session
    )

    assert result == "hello world"
    assert captured["url"] == "https://api.together.xyz/v1/chat/completions"
    payload = captured["json"]
    assert payload["model"] == "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    assert payload["temperature"] == 0.5
    assert payload["max_tokens"] == 512
    assert payload["messages"] == [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "test prompt"},
    ]
    assert "Bearer test-key" in captured["headers"]["Authorization"]


@pytest.mark.asyncio
async def test_openai_provider_uses_openai_base_url():
    """OpenAIProvider uses the correct OpenAI base URL."""
    provider = OpenAIProvider(api_key="openai-test-key")
    assert provider.base_url == "https://api.openai.com/v1"

    session, captured = make_mock_session(make_chat_response("openai response"))

    result = await provider.generate("hello", session=session)
    assert result == "openai response"
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"


@pytest.mark.asyncio
async def test_anthropic_provider_raises_on_get_embedding():
    """AnthropicProvider raises NotImplementedError on get_embedding."""
    provider = AnthropicProvider(api_key="anthropic-key")
    session = MagicMock()

    with pytest.raises(NotImplementedError) as exc_info:
        await provider.get_embedding("test text", session)

    assert "does not support embeddings" in str(exc_info.value)
    assert "--provider together" in str(exc_info.value) or "--provider openai" in str(exc_info.value)


@pytest.mark.asyncio
async def test_anthropic_provider_raises_on_judge_divergence():
    """AnthropicProvider raises NotImplementedError on judge_divergence."""
    provider = AnthropicProvider(api_key="anthropic-key")
    session = MagicMock()

    with pytest.raises(NotImplementedError) as exc_info:
        await provider.judge_divergence("output a", "output b", session)

    assert "does not support judge divergence" in str(exc_info.value)


@pytest.mark.asyncio
async def test_default_provider_is_together():
    """Module-level generate() without configure uses TogetherProvider."""
    session, captured = make_mock_session(make_chat_response("together response"))

    with patch.dict("os.environ", {"TOGETHER_API_KEY": "fake-key"}):
        result = await generator.generate("test", session=session)

    assert result == "together response"
    assert "together.xyz" in captured["url"]


@pytest.mark.asyncio
async def test_configure_provider_changes_backend():
    """After configure_provider(mock), module-level generate() delegates to mock."""

    class MockProvider:
        async def generate(self, prompt, system_prompt=None, temperature=0.0, max_tokens=1024, session=None):
            return f"mock:{prompt}"

    mock = MockProvider()
    configure_provider(mock)

    result = await generator.generate("hello")
    assert result == "mock:hello"


@pytest.mark.asyncio
async def test_configure_separate_embedding_provider():
    """Generation goes to one provider, embedding to another."""

    class GenOnly:
        async def generate(self, prompt, system_prompt=None, temperature=0.0, max_tokens=1024, session=None):
            return "gen-only"

    class EmbedOnly:
        async def get_embedding(self, text, session):
            return [1.0, 2.0, 3.0]

    configure_provider(GenOnly(), embedding_provider=EmbedOnly())

    result = await generator.generate("test")
    assert result == "gen-only"

    session = MagicMock()
    embedding = await generator.get_embedding("test", session)
    assert embedding == [1.0, 2.0, 3.0]


@pytest.mark.asyncio
async def test_anthropic_provider_generate():
    """AnthropicProvider uses Anthropic messages API format."""
    provider = AnthropicProvider(api_key="anthropic-key")

    session, captured = make_mock_session({"content": [{"text": "claude response"}]})

    result = await provider.generate("hello", system_prompt="be nice", session=session)
    assert result == "claude response"
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["json"]["system"] == "be nice"
    assert captured["headers"]["x-api-key"] == "anthropic-key"
    assert "anthropic-version" in captured["headers"]


@pytest.mark.asyncio
async def test_openai_provider_embedding():
    """OpenAIProvider supports embedding."""
    provider = OpenAIProvider(api_key="openai-key")

    session, captured = make_mock_session(make_embedding_response([0.1, 0.2, 0.3]))

    result = await provider.get_embedding("test", session)
    assert result == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_together_provider_judge_divergence():
    """TogetherProvider judge_divergence parses score correctly."""
    provider = TogetherProvider(api_key="test-key")

    session, captured = make_mock_session(make_chat_response("7"))

    result = await provider.judge_divergence("hello", "world", session)
    assert result == 7


@pytest.mark.asyncio
async def test_protocol_isinstance_checks():
    """Verify runtime_checkable protocols work correctly."""
    together = TogetherProvider(api_key="k")
    openai = OpenAIProvider(api_key="k")
    anthropic = AnthropicProvider(api_key="k")

    assert isinstance(together, GenerationProvider)
    assert isinstance(together, EmbeddingProvider)
    assert isinstance(together, JudgeProvider)

    assert isinstance(openai, GenerationProvider)
    assert isinstance(openai, EmbeddingProvider)
    assert isinstance(openai, JudgeProvider)

    # Anthropic has the methods but embedding/judge raise NotImplementedError
    assert isinstance(anthropic, GenerationProvider)
