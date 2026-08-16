"""Unit tests for src/ingestion/qdrant_store.py."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.ingestion.qdrant_store import ensure_collection, upsert_chunks

REQUIRED_PAYLOAD_FIELDS = [
    "page_num", "doc_type", "date", "venue", "shift_id",
    "table_flag", "content_hash", "content", "contextual_prefix", "bm25_text",
]


@pytest.mark.asyncio
async def test_upsert_all_payload_fields():
    captured_points = []

    async def mock_upsert(collection_name, points):
        captured_points.extend(points)

    mock_client = MagicMock()
    mock_client.upsert = mock_upsert

    chunks = [{
        "content": "Test content",
        "page_num": 1,
        "table_flag": False,
        "content_hash": "abc123",
        "contextual_prefix": "This is context.",
        "bm25_text": "This is context. Test content",
        "embedding": [0.1] * 384,
        "doc_type": "procedure",
    }]
    doc_metadata = {
        "doc_type": "procedure", "date": "2026-05-28",
        "venue": "Fairfield RSL", "shift_id": "N/A",
    }

    count = await upsert_chunks(mock_client, "test_col", chunks, doc_metadata)
    assert count == 1
    assert len(captured_points) == 1
    payload = captured_points[0].payload
    for field in REQUIRED_PAYLOAD_FIELDS:
        assert field in payload, f"Missing payload field: {field}"


@pytest.mark.asyncio
async def test_ensure_collection_creates():
    mock_client = MagicMock()
    mock_collections = MagicMock()
    mock_collections.collections = []
    mock_client.get_collections = AsyncMock(return_value=mock_collections)
    mock_client.create_collection = AsyncMock()

    await ensure_collection(mock_client, "new_collection", 384)
    mock_client.create_collection.assert_called_once()
