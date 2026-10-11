"""Unit tests for src/ingestion/context_generator.py."""
from unittest.mock import AsyncMock, patch

import pytest

from src.ingestion.context_generator import generate_contextual_prefix

_MODEL = "mistralai/mistral-nemo"
_BASE_URL = "https://openrouter.ai/api/v1"


@pytest.mark.asyncio
async def test_no_op_when_key_empty():
    result = await generate_contextual_prefix(
        chunk_content="Some content",
        doc_type="procedure",
        venue="Fairfield RSL",
        date="2026-05-28",
        api_key="",
        model=_MODEL,
        base_url=_BASE_URL,
    )
    assert result == ""


@pytest.mark.asyncio
async def test_prefix_returned_from_api():
    with patch(
        "src.ingestion.context_generator.chat",
        AsyncMock(return_value="  This chunk describes RSA procedures.  "),
    ):
        result = await generate_contextual_prefix(
            chunk_content="RSA policy content",
            doc_type="procedure",
            venue="Fairfield RSL",
            date="2026-05-28",
            api_key="fake-key",
            model=_MODEL,
            base_url=_BASE_URL,
        )
    assert result == "This chunk describes RSA procedures."


@pytest.mark.asyncio
async def test_api_error_returns_empty():
    with patch(
        "src.ingestion.context_generator.chat",
        AsyncMock(side_effect=Exception("API error")),
    ):
        result = await generate_contextual_prefix(
            chunk_content="Some content",
            doc_type="procedure",
            venue="Fairfield RSL",
            date="2026-05-28",
            api_key="fake-key",
            model=_MODEL,
            base_url=_BASE_URL,
        )
    assert result == ""
