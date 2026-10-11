"""Unit tests for src/generation/local_llm.py."""
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.generation.local_llm import generate_local
from src.llm_client import LocalLLMConfig

_OLLAMA = LocalLLMConfig(
    provider="ollama", model="llama3", ollama_url="http://localhost:11434"
)
_OPENROUTER = LocalLLMConfig(
    provider="openrouter",
    model="mistralai/mistral-nemo",
    api_key="fake-key",
    base_url="https://openrouter.ai/api/v1",
)


@pytest.mark.asyncio
async def test_happy_path_returns_string(httpx_mock):
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:11434/api/generate",
        json={"response": "The answer is 42."},
    )
    result = await generate_local("What is 6 times 7?", _OLLAMA)
    assert isinstance(result, str)
    assert "42" in result


@pytest.mark.asyncio
async def test_error_wrapping(httpx_mock):
    httpx_mock.add_exception(httpx.ConnectError("refused"))
    result = await generate_local("test prompt", _OLLAMA)
    assert isinstance(result, str)
    assert "generation_failed" in result


@pytest.mark.asyncio
async def test_empty_prompt_raises():
    with pytest.raises(ValueError, match="empty"):
        await generate_local("", _OLLAMA)


@pytest.mark.asyncio
async def test_openrouter_provider_generates_without_ollama():
    """The deployed path must generate without any Ollama endpoint present."""
    with patch(
        "src.generation.local_llm.complete", AsyncMock(return_value="cloud answer")
    ):
        result = await generate_local("a simple question", _OPENROUTER)
    assert result == "cloud answer"


@pytest.mark.asyncio
async def test_streaming_rejected_for_non_ollama_provider():
    with pytest.raises(NotImplementedError):
        await generate_local("prompt", _OPENROUTER, stream=True)
