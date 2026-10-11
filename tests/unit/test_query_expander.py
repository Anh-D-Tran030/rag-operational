"""Unit tests for src/retrieval/query_expander.py."""
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.llm_client import LocalLLMConfig
from src.retrieval.query_expander import expand_query

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
async def test_happy_path_returns_original_plus_expansions(httpx_mock):
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:11434/api/generate",
        json={"response": "expansion one\nexpansion two\nexpansion three"},
    )
    result = await expand_query(query="sales report for Q1", config=_OLLAMA, n_expansions=3)
    assert result[0] == "sales report for Q1"
    assert len(result) >= 2


@pytest.mark.asyncio
async def test_error_fallback_returns_original_only(httpx_mock):
    httpx_mock.add_exception(httpx.ConnectError("connection refused"))
    result = await expand_query(query="fallback query", config=_OLLAMA)
    assert result == ["fallback query"]


@pytest.mark.asyncio
async def test_http_error_fallback(httpx_mock):
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:11434/api/generate",
        status_code=500,
    )
    result = await expand_query(query="another query", config=_OLLAMA)
    assert result == ["another query"]


@pytest.mark.asyncio
async def test_openrouter_provider_expands_without_ollama():
    """The deployed path must expand without any Ollama endpoint present."""
    with patch(
        "src.retrieval.query_expander.complete",
        AsyncMock(return_value="alt one\nalt two"),
    ):
        result = await expand_query(query="original", config=_OPENROUTER, n_expansions=2)
    assert result == ["original", "alt one", "alt two"]


@pytest.mark.asyncio
@pytest.mark.parametrize("config", [_OLLAMA, _OPENROUTER], ids=["local", "openrouter"])
async def test_prompt_requests_terminology_and_output_is_bounded(config):
    mock = AsyncMock(return_value="SOP 12 staff\nsop 12 STAFF\nSOP-12 minimum\nextra\nmore")
    with patch("src.retrieval.query_expander.complete", mock):
        result = await expand_query(query="SOP-12 minimum", config=config, n_expansions=2)
    prompt = mock.await_args.args[0]
    assert "abbreviations" in prompt and "section headings" in prompt
    assert "named entity" in prompt and "Query: SOP-12 minimum" in prompt
    assert mock.await_args.args[1] is config
    assert result == ["SOP-12 minimum", "SOP 12 staff", "extra"]


@pytest.mark.asyncio
@pytest.mark.parametrize("config", [_OLLAMA, _OPENROUTER], ids=["local", "openrouter"])
async def test_provider_error_falls_back_to_original(config):
    mock = AsyncMock(side_effect=RuntimeError("provider down"))
    with patch("src.retrieval.query_expander.complete", mock):
        result = await expand_query(query="original", config=config)
    assert result == ["original"]
