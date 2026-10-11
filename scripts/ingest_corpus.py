#!/usr/bin/env python3
"""Bootstrap one manifest-defined SQLite + Qdrant corpus."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from qdrant_client import AsyncQdrantClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from src.config import Settings
from src.ingestion.corpus_manifest import (
    CorpusManifest,
    load_corpus_manifest,
    resolve_manifest_source,
)
from src.ingestion.embedder import Embedder
from src.ingestion.pipeline import ingest_document
from src.ingestion.schema_indexer import index_schema_metadata
from src.ingestion.structured_loader import seed_database_sources


async def _validate(client: AsyncQdrantClient, manifest: CorpusManifest,
                    db_url: str, expected_sources: set[str]) -> dict[str, int]:
    collections = {item.name for item in (await client.get_collections()).collections}
    missing = {manifest.document_collection, manifest.schema_collection} - collections
    if missing:
        raise RuntimeError(f"bootstrap validation failed; missing collections: {sorted(missing)}")
    document_info = await client.get_collection(manifest.document_collection)
    schema_info = await client.get_collection(manifest.schema_collection)
    document_points = document_info.points_count or 0
    schema_points = schema_info.points_count or 0
    if document_points < 1 or schema_points < 1:
        raise RuntimeError("bootstrap validation failed; expected non-empty collections")
    points, _ = await client.scroll(
        collection_name=manifest.document_collection, limit=max(document_points, 1),
        with_payload=True, with_vectors=False,
    )
    represented = {(point.payload or {}).get("source_id") for point in points}
    if not expected_sources.issubset(represented):
        absent = sorted(expected_sources - represented)
        raise RuntimeError(f"bootstrap validation failed; missing document sources: {absent}")
    engine = create_async_engine(db_url, echo=False)
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
    finally:
        await engine.dispose()
    return {"document_points": document_points, "schema_points": schema_points}


async def bootstrap(
    manifest_path: str | Path, *, runtime_settings: Settings | None = None,
    qdrant_client: AsyncQdrantClient | None = None, embedder: Embedder | None = None,
) -> dict[str, Any]:
    """Provision exactly one manifest without deleting any existing collection."""
    manifest_path = Path(manifest_path)
    manifest = load_corpus_manifest(manifest_path)
    settings = runtime_settings or Settings()
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for corpus bootstrap")
    corpus_settings = settings.model_copy(update={
        "qdrant_collection": manifest.document_collection,
        "qdrant_schema_collection": manifest.schema_collection,
        "embedding_model": manifest.embedding.model,
        "context_model": manifest.contextualization_model,
        "chunk_size": manifest.chunking.chunk_size,
        "chunk_overlap": manifest.chunking.overlap,
    })
    print(f"Corpus version: {manifest.corpus_version}")
    print(f"Document collection: {manifest.document_collection}")
    print(f"Schema collection: {manifest.schema_collection}")
    structured = [
        {"table": item.table, "path": str(resolve_manifest_source(manifest_path, item.path))}
        for item in manifest.structured_data
    ]
    relational_counts = await seed_database_sources(corpus_settings.db_url, structured)
    client = qdrant_client or AsyncQdrantClient(
        url=corpus_settings.qdrant_url, api_key=corpus_settings.qdrant_api_key or None,
    )
    owns_client = qdrant_client is None
    model = embedder or Embedder(manifest.embedding.model)
    chunk_count = 0
    try:
        await index_schema_metadata(
            client, manifest.schema_collection, model, corpus_settings.db_url,
            schema_views=[item.model_dump() for item in manifest.schema_views],
            corpus_version=manifest.corpus_version, manifest_digest=manifest.digest,
            embedding_dimension=manifest.embedding.dimension,
        )
        for document in manifest.documents:
            result = await ingest_document(
                file_path=str(resolve_manifest_source(manifest_path, document.path)),
                doc_metadata=document.metadata(), qdrant_client=client, embedder=model,
                settings=corpus_settings, corpus_version=manifest.corpus_version,
                manifest_digest=manifest.digest, source_id=document.source_id,
                embedding_dimension=manifest.embedding.dimension,
                require_contextual_prefix=True,
            )
            chunk_count += result["chunks_ingested"]
        state = await _validate(client, manifest, corpus_settings.db_url,
                                {item.source_id for item in manifest.documents})
    finally:
        if owns_client:
            await client.close()
    summary: dict[str, Any] = {
        "corpus_version": manifest.corpus_version,
        "document_collection": manifest.document_collection,
        "schema_collection": manifest.schema_collection,
        "source_documents": len(manifest.documents),
        "chunks_upserted": chunk_count,
        "schema_points": state["schema_points"],
        "document_points": state["document_points"],
        "relational_rows_processed": relational_counts,
    }
    print("Bootstrap summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Path to corpus YAML manifest")
    asyncio.run(bootstrap(parser.parse_args().manifest))


if __name__ == "__main__":
    main()
