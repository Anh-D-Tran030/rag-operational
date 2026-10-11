"""Unit tests for src/ingestion/dedup.py."""
from src.ingestion.dedup import deduplicate


def test_dedup_removes_duplicates():
    chunks = [
        {"content": "Hello World", "page_num": 1},
        {"content": "Unique content", "page_num": 2},
        {"content": "hello world", "page_num": 3},
    ]
    result = deduplicate(chunks)
    assert len(result) == 2
    assert all("content_hash" in c for c in result)


def test_dedup_preserves_unique():
    chunks = [
        {"content": "First chunk", "page_num": 1},
        {"content": "Second chunk", "page_num": 2},
        {"content": "Third chunk", "page_num": 3},
    ]
    result = deduplicate(chunks)
    assert len(result) == 3
    hashes = [c["content_hash"] for c in result]
    assert len(set(hashes)) == 3
