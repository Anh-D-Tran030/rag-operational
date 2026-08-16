"""Integration tests for the full ingestion pipeline."""
import pytest

from src.config import Settings
from src.ingestion.embedder import Embedder
from src.ingestion.pipeline import ingest_document

FIXTURE_DIR = "tests/fixtures"
REQUIRED_PAYLOAD_FIELDS = [
    "page_num", "doc_type", "date", "venue", "shift_id",
    "table_flag", "content_hash", "content", "contextual_prefix", "bm25_text",
]


@pytest.mark.asyncio
async def test_ingest_pdf_happy_path(qdrant_client, test_collection_name):
    settings = Settings(qdrant_collection=test_collection_name, anthropic_api_key="")
    embedder = Embedder(settings.embedding_model)

    result = await ingest_document(
        file_path=f"{FIXTURE_DIR}/sample_procedure.pdf",
        doc_metadata={
            "doc_type": "procedure", "date": "2026-05-01",
            "venue": "Fairfield RSL", "shift_id": "N/A",
        },
        qdrant_client=qdrant_client,
        embedder=embedder,
        settings=settings,
    )

    assert result["chunks_ingested"] > 0
    assert result["doc_type"] == "procedure"

    scroll_result = await qdrant_client.scroll(
        collection_name=test_collection_name, limit=10, with_payload=True
    )
    assert len(scroll_result[0]) > 0
    for point in scroll_result[0]:
        for field in REQUIRED_PAYLOAD_FIELDS:
            assert field in point.payload, f"Missing payload field: {field}"


@pytest.mark.asyncio
async def test_ingest_text_happy_path(qdrant_client, test_collection_name):
    settings = Settings(qdrant_collection=test_collection_name, anthropic_api_key="")
    embedder = Embedder(settings.embedding_model)

    result = await ingest_document(
        file_path=f"{FIXTURE_DIR}/sample_shift_log.txt",
        doc_metadata={
            "doc_type": "shift_log", "date": "2026-05-28",
            "venue": "Fairfield RSL", "shift_id": "SH-001",
        },
        qdrant_client=qdrant_client,
        embedder=embedder,
        settings=settings,
    )
    assert result["chunks_ingested"] > 0
    assert result["doc_type"] == "shift_log"


@pytest.mark.asyncio
async def test_ingest_missing_file(qdrant_client, test_collection_name):
    settings = Settings(qdrant_collection=test_collection_name, anthropic_api_key="")
    embedder = Embedder(settings.embedding_model)

    with pytest.raises(FileNotFoundError):
        await ingest_document(
            file_path="nonexistent/path/doc.pdf",
            doc_metadata={
                "doc_type": "procedure", "date": "2026-05-01",
                "venue": "Fairfield RSL", "shift_id": "N/A",
            },
            qdrant_client=qdrant_client,
            embedder=embedder,
            settings=settings,
        )
