"""Unit tests for src/retrieval/cache.py."""

import pytest

from src.retrieval.cache import SemanticCache


@pytest.mark.asyncio
async def test_disabled_when_cache_url_empty():
    cache = SemanticCache("", "all-MiniLM-L6-v2")
    result = await cache.get("any query")
    assert result is None


@pytest.mark.asyncio
async def test_disabled_set_is_noop():
    cache = SemanticCache("", "all-MiniLM-L6-v2")
    await cache.set("query", {"answer": "hello"})  # should not raise


@pytest.mark.asyncio
async def test_hit_counter_incremented(monkeypatch):
    cache = SemanticCache("memory://test", "all-MiniLM-L6-v2")
    await cache.set("what is peak staffing?", {"answer": "5 staff"})

    hit_count = 0
    miss_count = 0

    import src.observability.metrics as m
    original_labels = m.rag_cache_hit_total.labels

    def tracking_labels(result):
        nonlocal hit_count, miss_count
        if result == "hit":
            hit_count += 1
        elif result == "miss":
            miss_count += 1
        return original_labels(result=result)

    monkeypatch.setattr(m.rag_cache_hit_total, "labels", tracking_labels)

    result = await cache.get("what is peak staffing?")
    assert result == {"answer": "5 staff"}
    assert hit_count == 1


@pytest.mark.asyncio
async def test_miss_counter_incremented(monkeypatch):
    cache = SemanticCache("memory://test", "all-MiniLM-L6-v2")

    miss_count = 0

    import src.observability.metrics as m
    original_labels = m.rag_cache_hit_total.labels

    def tracking_labels(result):
        nonlocal miss_count
        if result == "miss":
            miss_count += 1
        return original_labels(result=result)

    monkeypatch.setattr(m.rag_cache_hit_total, "labels", tracking_labels)

    result = await cache.get("a completely unseen query xyz")
    assert result is None
    assert miss_count == 1
