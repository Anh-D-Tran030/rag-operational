"""Unit tests for src/ingestion/embedder.py."""
import pytest

from src.ingestion.embedder import Embedder, embed_chunks


def test_embed_batch_shape():
    embedder = Embedder("all-MiniLM-L6-v2")
    result = embedder.embed_batch(["hello", "world"])
    assert len(result) == 2
    assert len(result[0]) == 384
    assert len(result[1]) == 384


@pytest.mark.asyncio
async def test_embed_chunks_async():
    embedder = Embedder("all-MiniLM-L6-v2")
    chunks = [
        {"content": "First chunk content", "page_num": 1},
        {"content": "Second chunk content", "page_num": 2},
    ]
    result = await embed_chunks(chunks, embedder)
    assert len(result) == 2
    for chunk in result:
        assert "embedding" in chunk
        assert isinstance(chunk["embedding"], list)
        assert len(chunk["embedding"]) == 384
