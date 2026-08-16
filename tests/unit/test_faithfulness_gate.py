"""Unit tests for the faithfulness gate."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.generation.faithfulness_gate import check_faithfulness

_MODEL = "mistralai/mistral-nemo"
_BASE_URL = "https://openrouter.ai/api/v1"


@pytest.mark.asyncio
async def test_gate_skipped_when_no_key():
    result = await check_faithfulness(
        answer="some answer",
        context_chunks=[{"content": "context"}],
        threshold=0.9,
        model=_MODEL,
        api_key="",
        base_url=_BASE_URL,
    )
    assert result is True


@pytest.mark.asyncio
async def test_gate_passes_on_yes():
    with patch(
        "src.generation.faithfulness_gate.chat", AsyncMock(return_value="yes")
    ):
        result = await check_faithfulness(
            answer="The venue opened at 9pm.",
            context_chunks=[{"content": "The venue opened at 9pm."}],
            threshold=0.9,
            model=_MODEL,
            api_key="test-key",
            base_url=_BASE_URL,
        )
    assert result is True


@pytest.mark.asyncio
async def test_gate_fails_on_no():
    with (
        patch("src.generation.faithfulness_gate.chat", AsyncMock(return_value="no")),
        patch(
            "src.generation.faithfulness_gate.rag_faithfulness_gate_failures"
        ) as mock_counter,
    ):
        result = await check_faithfulness(
            answer="hallucinated answer",
            context_chunks=[{"content": "unrelated context"}],
            threshold=0.9,
            model=_MODEL,
            api_key="test-key",
            base_url=_BASE_URL,
        )
    assert result is False
    mock_counter.inc.assert_called_once()


@pytest.mark.asyncio
async def test_gate_open_on_api_error():
    with patch(
        "src.generation.faithfulness_gate.chat",
        AsyncMock(side_effect=Exception("API error")),
    ):
        result = await check_faithfulness(
            answer="some answer",
            context_chunks=[{"content": "context"}],
            threshold=0.9,
            model=_MODEL,
            api_key="test-key",
            base_url=_BASE_URL,
        )
    assert result is True
