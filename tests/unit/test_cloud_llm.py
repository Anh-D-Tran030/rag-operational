"""Unit tests for src/generation/cloud_llm.py."""
from unittest.mock import AsyncMock, patch

import pytest

from src.generation.cloud_llm import generate_cloud

_BASE_URL = "https://openrouter.ai/api/v1"


@pytest.mark.asyncio
async def test_returns_generated_text():
    with patch(
        "src.generation.cloud_llm.chat", AsyncMock(return_value="Here is the answer.")
    ):
        result = await generate_cloud(
            "What is 2+2?", "openai/gpt-oss-120b", "fake-key", _BASE_URL
        )
    assert result == "Here is the answer."


@pytest.mark.asyncio
async def test_passes_model_and_base_url_through():
    mock_chat = AsyncMock(return_value="ok")
    with patch("src.generation.cloud_llm.chat", mock_chat):
        await generate_cloud("Hello", "mistralai/mistral-nemo", "fake-key", _BASE_URL)
    kwargs = mock_chat.await_args.kwargs
    assert kwargs["model"] == "mistralai/mistral-nemo"
    assert kwargs["base_url"] == _BASE_URL


@pytest.mark.asyncio
async def test_error_wrapping():
    with patch(
        "src.generation.cloud_llm.chat", AsyncMock(side_effect=Exception("API down"))
    ):
        with pytest.raises(RuntimeError, match="cloud_generation_failed"):
            await generate_cloud("test", "openai/gpt-oss-120b", "fake-key", _BASE_URL)


@pytest.mark.asyncio
async def test_empty_prompt_raises():
    with pytest.raises(ValueError):
        await generate_cloud("", "model", "key", _BASE_URL)


@pytest.mark.asyncio
async def test_empty_api_key_raises():
    with pytest.raises(ValueError):
        await generate_cloud("prompt", "model", "", _BASE_URL)
