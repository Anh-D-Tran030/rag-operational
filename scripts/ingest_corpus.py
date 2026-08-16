#!/usr/bin/env python3
"""Ingest the mock operational corpus into Qdrant and seed the relational DB."""
import asyncio

from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import create_async_engine

from src.config import settings
from src.ingestion.embedder import Embedder
from src.ingestion.pipeline import ingest_document
from src.ingestion.schema_indexer import index_schema_metadata
from src.ingestion.structured_loader import create_views, seed_database

FIXTURE_DIR = "tests/fixtures"

DOCUMENTS = [
    {
        "file_path": f"{FIXTURE_DIR}/sample_procedure.pdf",
        "doc_metadata": {
            "doc_type": "procedure",
            "date": "2026-05-01",
            "venue": "Fairfield_RSL",
            "shift_id": "N/A",
        },
    },
    {
        "file_path": f"{FIXTURE_DIR}/sample_compliance.pdf",
        "doc_metadata": {
            "doc_type": "compliance",
            "date": "2026-04-01",
            "venue": "Fairfield_RSL",
            "shift_id": "N/A",
        },
    },
    {
        "file_path": f"{FIXTURE_DIR}/sample_shift_log.txt",
        "doc_metadata": {
            "doc_type": "shift_log",
            "date": "2026-05-28",
            "venue": "Fairfield_RSL",
            "shift_id": "SH-001",
        },
    },
]


async def main() -> None:
    print("=== Fairfield RSL Corpus Ingestion ===\n")

    print(f"[1/4] Seeding relational DB at {settings.db_url}...")
    counts = await seed_database(settings.db_url, FIXTURE_DIR)
    for table, n in counts.items():
        print(f"      {table}: {n} rows")

    engine = create_async_engine(settings.db_url, echo=False)
    print("[2/4] Creating curated views...")
    await create_views(engine)
    await engine.dispose()
    print("      5 views created")

    qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    embedder = Embedder(settings.embedding_model)

    print(f"[3/4] Indexing schema metadata into '{settings.qdrant_schema_collection}'...")
    schema_count = await index_schema_metadata(
        qdrant, settings.qdrant_schema_collection, embedder, settings.db_url
    )
    print(f"      {schema_count} schema entries indexed")

    print("[4/4] Ingesting documents...")
    for doc in DOCUMENTS:
        print(f"      → {doc['file_path']} ({doc['doc_metadata']['doc_type']})")
        result = await ingest_document(
            file_path=doc["file_path"],
            doc_metadata=doc["doc_metadata"],
            qdrant_client=qdrant,
            embedder=embedder,
            settings=settings,
        )
        print(
            f"        chunks_ingested={result['chunks_ingested']}"
            f"  deduped={result['chunks_deduplicated']}"
        )

    await qdrant.close()
    print("\nSUCCESS — corpus ingestion complete")


if __name__ == "__main__":
    asyncio.run(main())
