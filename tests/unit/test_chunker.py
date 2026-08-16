"""Unit tests for src/ingestion/chunker.py."""
from src.ingestion.chunker import chunk_pages


def test_table_chunk_not_split():
    pages = [{
        "page_num": 1, "content": "col1 | col2\n---|---\nval1 | val2",
        "table_flag": True, "doc_type": "procedure",
    }]
    chunks = chunk_pages(pages, chunk_size=10, overlap=2)
    assert len(chunks) == 1
    assert chunks[0]["table_flag"] is True


def test_overlap_applied():
    words = " ".join([f"word{i}" for i in range(20)])
    pages = [{"page_num": 1, "content": words, "table_flag": False, "doc_type": "procedure"}]
    chunks = chunk_pages(pages, chunk_size=10, overlap=3)
    assert len(chunks) >= 2
    first_end_words = chunks[0]["content"].split()[-3:]
    second_start_words = chunks[1]["content"].split()[:3]
    assert first_end_words == second_start_words


def test_doc_type_passthrough():
    pages = [{
        "page_num": 1, "content": "some content here for shift log",
        "table_flag": False, "doc_type": "shift_log",
    }]
    chunks = chunk_pages(pages, chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0]["doc_type"] == "shift_log"
