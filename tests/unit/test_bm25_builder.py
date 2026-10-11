"""Unit tests for src/ingestion/bm25_builder.py."""
from src.ingestion.bm25_builder import add_bm25_text, build_bm25_text


def test_bm25_text_with_prefix():
    chunk = {
        "contextual_prefix": "This describes RSA procedures.",
        "content": "Staff must check IDs.",
    }
    result = build_bm25_text(chunk)
    assert "This describes RSA procedures." in result
    assert "Staff must check IDs." in result


def test_bm25_text_without_prefix():
    chunk = {"content": "Staff must check IDs."}
    result = build_bm25_text(chunk)
    assert result == "Staff must check IDs."


def test_add_bm25_text_field_present():
    chunks = [
        {"content": "First chunk", "contextual_prefix": "Prefix one"},
        {"content": "Second chunk"},
    ]
    result = add_bm25_text(chunks)
    assert len(result) == 2
    assert all("bm25_text" in c for c in result)
    assert result[0]["bm25_text"] == "Prefix one First chunk"
    assert result[1]["bm25_text"] == "Second chunk"
