from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.ingest_corpus import bootstrap
from src.config import Settings
from src.ingestion.pipeline import ingest_document


@pytest.mark.asyncio
async def test_bootstrap_uses_exact_manifest_sources_and_summarises():
    client = MagicMock()
    embedder = MagicMock()
    ingest = AsyncMock(return_value={"chunks_ingested": 2, "chunks_deduplicated": 0})
    schema = AsyncMock(return_value=5)
    seed = AsyncMock(return_value={"shift_records": 20, "incident_logs": 10,
                                  "staffing_history": 20})
    validated = {"document_points": 6, "schema_points": 5}
    settings = Settings(
        _env_file=None, db_url="sqlite+aiosqlite:///:memory:",
        openrouter_api_key="test-key",
    )

    with (
        patch("scripts.ingest_corpus.ingest_document", ingest),
        patch("scripts.ingest_corpus.index_schema_metadata", schema),
        patch("scripts.ingest_corpus.seed_database_sources", seed),
        patch("scripts.ingest_corpus._validate", new=AsyncMock(return_value=validated)),
    ):
        summary = await bootstrap(
            "corpus/v1.yaml", runtime_settings=settings,
            qdrant_client=client, embedder=embedder,
        )

    assert ingest.await_count == 3
    assert summary["corpus_version"] == "v1"
    assert summary["source_documents"] == 3
    assert summary["chunks_upserted"] == 6
    assert summary["document_points"] == 6
    assert summary["schema_points"] == 5
    assert {call.kwargs["source_id"] for call in ingest.await_args_list} == {
        "procedure_pdf", "compliance_pdf", "shift_log.txt",
    }
    assert all(call.kwargs["manifest_digest"] for call in ingest.await_args_list)
    assert all(call.kwargs["require_contextual_prefix"] for call in ingest.await_args_list)


@pytest.mark.asyncio
async def test_bootstrap_requires_openrouter_before_writes():
    with patch("scripts.ingest_corpus.seed_database_sources", new=AsyncMock()) as seed:
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY is required"):
            await bootstrap(
                "corpus/v1.yaml",
                runtime_settings=Settings(_env_file=None, openrouter_api_key=""),
                qdrant_client=MagicMock(), embedder=MagicMock(),
            )
    seed.assert_not_awaited()


@pytest.mark.asyncio
async def test_strict_manifest_ingest_rejects_empty_context_before_upsert():
    settings = Settings(_env_file=None, openrouter_api_key="test-key")
    with patch(
        "src.ingestion.pipeline.generate_contextual_prefix",
        new=AsyncMock(return_value=""),
    ):
        with pytest.raises(RuntimeError, match="contextual prefix generation failed"):
            await ingest_document(
                file_path="tests/fixtures/sample_shift_log.txt",
                doc_metadata={"doc_type": "shift_log"},
                qdrant_client=MagicMock(), embedder=MagicMock(), settings=settings,
                corpus_version="v1", manifest_digest="digest", source_id="shift_log",
                require_contextual_prefix=True,
            )
