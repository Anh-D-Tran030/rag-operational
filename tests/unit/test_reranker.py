"""Unit tests for src/retrieval/reranker.py."""
import pytest

from src.retrieval.reranker import rerank

CHUNKS = [
    {
        "content": "The Q1 sales report shows strong growth in the North region.",
        "content_hash": "a1",
    },
    {"content": "Employee handbook section on leave policy.", "content_hash": "a2"},
    {"content": "Q1 revenue increased by 15% compared to last year.", "content_hash": "a3"},
    {"content": "Office supply order submitted on Monday.", "content_hash": "a4"},
]


@pytest.mark.asyncio
async def test_top_n_enforced():
    result = await rerank(
        query="Q1 sales",
        chunks=CHUNKS,
        top_n=2,
        score_threshold=0.0,
        trace_id="trace-001",
    )
    assert len(result) <= 2


@pytest.mark.asyncio
async def test_rerank_score_field_present():
    result = await rerank(
        query="Q1 sales revenue",
        chunks=CHUNKS,
        top_n=4,
        score_threshold=0.0,
        trace_id="trace-002",
    )
    for chunk in result:
        assert "rerank_score" in chunk
        assert isinstance(chunk["rerank_score"], float)


@pytest.mark.asyncio
async def test_hard_reject_below_threshold():
    # Use a threshold of 1.0 which no chunk should exceed
    result = await rerank(
        query="completely unrelated gibberish xyzzy",
        chunks=CHUNKS,
        top_n=4,
        score_threshold=1.0,
        trace_id="trace-003",
    )
    assert result == []


@pytest.mark.asyncio
async def test_trace_id_param_accepted():
    # Ensures trace_id is a named param (not relying on contextvars)
    result = await rerank(
        query="sales",
        chunks=CHUNKS[:2],
        top_n=2,
        score_threshold=0.0,
        trace_id="explicit-trace-id",
    )
    assert isinstance(result, list)
