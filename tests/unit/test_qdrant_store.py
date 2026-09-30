"""Unit tests for src/ingestion/qdrant_store.py."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.ingestion.qdrant_store import (
    deterministic_point_id,
    ensure_collection,
    upsert_chunks,
)

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


def test_deterministic_point_id_is_stable_and_source_scoped():
    chunk = {"page_num": 1, "chunk_index": 2, "content_hash": "abc"}

    first = deterministic_point_id("v1", "procedure", chunk)
    second = deterministic_point_id("v1", "procedure", chunk)

    assert first == second
    assert first != deterministic_point_id("v1", "compliance", chunk)


@pytest.mark.asyncio
async def test_repeat_manifest_upsert_reuses_point_id():
    captured = []
    client = MagicMock()
    client.upsert = AsyncMock(side_effect=lambda **kwargs: captured.append(kwargs["points"]))
    chunks = [{
        "content": "stable", "page_num": 1, "chunk_index": 0,
        "content_hash": "abc", "embedding": [0.1] * 384,
    }]

    for _ in range(2):
        await upsert_chunks(
            client, "rag_operational_v1", chunks, {}, corpus_version="v1",
            manifest_digest="digest", source_id="procedure_pdf",
        )

    assert captured[0][0].id == captured[1][0].id
    assert captured[0][0].payload["manifest_digest"] == "digest"


@pytest.mark.asyncio
async def test_existing_collection_rejects_different_manifest():
    client = MagicMock()
    collections = MagicMock()
    collection = MagicMock()
    collection.name = "rag_operational_v1"
    collections.collections = [collection]
    client.get_collections = AsyncMock(return_value=collections)
    info = MagicMock()
    info.config.params.vectors = {"dense": MagicMock(size=384)}
    client.get_collection = AsyncMock(return_value=info)
    point = MagicMock(payload={"corpus_version": "v1", "manifest_digest": "old"})
    client.scroll = AsyncMock(return_value=([point], None))

    with pytest.raises(ValueError, match="different corpus"):
        await ensure_collection(
            client, "rag_operational_v1", 384, corpus_version="v1",
            manifest_digest="new",
        )


@pytest.mark.asyncio
async def test_existing_collection_rejects_wrong_vector_size():
    client = MagicMock()
    collections = MagicMock()
    collection = MagicMock()
    collection.name = "rag_operational_v1"
    collections.collections = [collection]
    client.get_collections = AsyncMock(return_value=collections)
    info = MagicMock()
    info.config.params.vectors = {"dense": MagicMock(size=768)}
    client.get_collection = AsyncMock(return_value=info)

    with pytest.raises(ValueError, match="incompatible vectors"):
        await ensure_collection(client, "rag_operational_v1", 384)
