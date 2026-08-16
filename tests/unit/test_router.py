"""Unit tests for src/generation/router.py."""
from unittest.mock import AsyncMock, patch

import pytest

from src.generation.router import classify_intent
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
async def test_classify_vector_search(httpx_mock):
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:11434/api/generate",
        json={"response": '{"tool": "vector_search", "complexity": "simple"}'},
    )
    result = await classify_intent("What does the safety SOP say?", _OLLAMA)
    assert result["tool"] == "vector_search"
    assert result["complexity"] == "simple"


@pytest.mark.asyncio
async def test_classify_text_to_sql(httpx_mock):
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:11434/api/generate",
        json={"response": '{"tool": "text_to_sql", "complexity": "simple"}'},
    )
    result = await classify_intent("How many staff worked last Saturday?", _OLLAMA)
    assert result["tool"] == "text_to_sql"
    assert result["complexity"] == "simple"


@pytest.mark.asyncio
async def test_classify_hybrid(httpx_mock):
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:11434/api/generate",
        json={"response": '{"tool": "hybrid", "complexity": "complex"}'},
    )
    result = await classify_intent("Did staffing meet SOP thresholds on Saturday?", _OLLAMA)
    assert result["tool"] == "hybrid"
    assert result["complexity"] == "complex"


@pytest.mark.asyncio
async def test_error_defaults_to_hybrid_complex(httpx_mock):
    import httpx

    httpx_mock.add_exception(httpx.ConnectError("refused"))
    result = await classify_intent("any query", _OLLAMA)
    assert result == {"tool": "hybrid", "complexity": "complex"}


@pytest.mark.asyncio
async def test_openrouter_provider_routes_without_ollama():
    """The deployed path must classify without any Ollama endpoint present."""
    with patch(
        "src.generation.router.complete",
        AsyncMock(return_value='{"tool": "text_to_sql", "complexity": "simple"}'),
    ):
        result = await classify_intent("How many incidents last week?", _OPENROUTER)
    assert result == {"tool": "text_to_sql", "complexity": "simple"}
