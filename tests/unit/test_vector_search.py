"""Unit tests for src/retrieval/vector_search.py."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.retrieval.vector_search import hybrid_search


def _make_hit(id_: str, score: float, payload: dict):
    hit = MagicMock()
    hit.id = id_
    hit.score = score
    hit.payload = payload
    return hit


def _query_response(hits: list):
    """Mimic the QueryResponse wrapper `query_points` returns."""
    response = MagicMock()
    response.points = hits
    return response


def _make_payload(content: str, content_hash: str, bm25_text: str = "") -> dict:
    return {
        "content": content,
        "content_hash": content_hash,
        "bm25_text": bm25_text or content,
        "page_num": 1,
        "doc_type": "report",
        "date": "2024-01-01",
        "venue": None,
        "shift_id": None,
        "table_flag": False,
        "contextual_prefix": "",
    }


@pytest.fixture
def mock_client():
    client = AsyncMock()

    dense_hits = [
        _make_hit("1", 0.95, _make_payload("Q1 sales report", "hash1", "Q1 sales report")),
        _make_hit("2", 0.80, _make_payload("Revenue grew 15%", "hash2", "Revenue grew 15%")),
        _make_hit("3", 0.60, _make_payload("Office supply order", "hash3", "Office supply order")),
    ]
    client.query_points = AsyncMock(return_value=_query_response(dense_hits))

    scroll_points = [
        _make_hit("1", 0.0, _make_payload("Q1 sales report", "hash1", "Q1 sales report")),
        _make_hit("2", 0.0, _make_payload("Revenue grew 15%", "hash2", "Revenue grew 15%")),
        _make_hit("3", 0.0, _make_payload("Office supply order", "hash3", "Office supply order")),
    ]
    client.scroll = AsyncMock(return_value=(scroll_points, None))

    return client


@pytest.mark.asyncio
async def test_rrf_merges_results(mock_client):
    result = await hybrid_search(
        query="Q1 sales",
        query_embedding=[0.1] * 384,
        client=mock_client,
        collection="docs",
        top_k=3,
    )
    assert len(result) > 0
    assert mock_client.query_points.called
    assert mock_client.scroll.called


@pytest.mark.asyncio
async def test_top_k_enforced(mock_client):
    result = await hybrid_search(
        query="sales",
        query_embedding=[0.1] * 384,
        client=mock_client,
        collection="docs",
        top_k=2,
    )
    assert len(result) <= 2


@pytest.mark.asyncio
async def test_dedup_by_content_hash(mock_client):
    """Duplicate content_hash should appear only once."""
    dup_payload = _make_payload("Q1 sales report", "hash1", "Q1 sales report")
    dense_hits = [
        _make_hit("1", 0.95, dup_payload),
        _make_hit("1b", 0.90, dup_payload),  # same hash1
        _make_hit("2", 0.80, _make_payload("Revenue grew 15%", "hash2", "Revenue grew")),
    ]
    mock_client.query_points = AsyncMock(return_value=_query_response(dense_hits))
    scroll_points = [
        _make_hit("1", 0.0, dup_payload),
        _make_hit("1b", 0.0, dup_payload),
        _make_hit("2", 0.0, _make_payload("Revenue grew 15%", "hash2", "Revenue grew")),
    ]
    mock_client.scroll = AsyncMock(return_value=(scroll_points, None))

    result = await hybrid_search(
        query="Q1 sales",
        query_embedding=[0.1] * 384,
        client=mock_client,
        collection="docs",
        top_k=5,
    )
    hashes = [r["content_hash"] for r in result]
    assert len(hashes) == len(set(hashes))


@pytest.mark.asyncio
async def test_empty_scroll_still_returns_dense(mock_client):
    mock_client.scroll = AsyncMock(return_value=([], None))
    result = await hybrid_search(
        query="Q1 sales",
        query_embedding=[0.1] * 384,
        client=mock_client,
        collection="docs",
        top_k=3,
    )
    assert isinstance(result, list)
