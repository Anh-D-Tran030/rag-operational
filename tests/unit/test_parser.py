"""Unit tests for src/ingestion/parser.py."""
import pytest

from src.ingestion.parser import parse_pdf, parse_text

FIXTURE_DIR = "tests/fixtures"


@pytest.mark.asyncio
async def test_parse_pdf_valid():
    pages = await parse_pdf(f"{FIXTURE_DIR}/sample_procedure.pdf")
    assert isinstance(pages, list)
    assert len(pages) > 0
    for page in pages:
        assert "page_num" in page
        assert "content" in page
        assert "table_flag" in page
        assert isinstance(page["table_flag"], bool)


@pytest.mark.asyncio
async def test_parse_pdf_missing_file():
    with pytest.raises(FileNotFoundError):
        await parse_pdf(f"{FIXTURE_DIR}/nonexistent.pdf")


@pytest.mark.asyncio
async def test_parse_text_valid():
    pages = await parse_text(f"{FIXTURE_DIR}/sample_shift_log.txt")
    assert isinstance(pages, list)
    assert len(pages) > 0
    for page in pages:
        assert "page_num" in page
        assert "content" in page
        assert page["table_flag"] is False
        assert "doc_type" in page


@pytest.mark.asyncio
async def test_parse_text_missing_file():
    with pytest.raises(FileNotFoundError):
        await parse_text(f"{FIXTURE_DIR}/nonexistent.txt")
