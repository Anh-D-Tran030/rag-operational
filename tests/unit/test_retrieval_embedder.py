"""Unit tests for src/retrieval/embedder.py."""
import pytest

from src.retrieval.embedder import _models, embed_query


@pytest.mark.asyncio
async def test_embed_query_returns_384_dim():
    result = await embed_query("What is the sales report?", "all-MiniLM-L6-v2")
    assert isinstance(result, list)
    assert len(result) == 384
    assert all(isinstance(v, float) for v in result)


@pytest.mark.asyncio
async def test_singleton_not_reloaded_on_second_call():
    _models.clear()
    await embed_query("first call", "all-MiniLM-L6-v2")
    model_id_first = id(_models["all-MiniLM-L6-v2"])
    await embed_query("second call", "all-MiniLM-L6-v2")
    model_id_second = id(_models["all-MiniLM-L6-v2"])
    assert model_id_first == model_id_second
