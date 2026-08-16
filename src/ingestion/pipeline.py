"""Orchestrate the full document ingestion pipeline."""
from __future__ import annotations

from qdrant_client import AsyncQdrantClient

from src.config import Settings
from src.ingestion.anchors import annotate_chunks
from src.ingestion.bm25_builder import add_bm25_text
from src.ingestion.chunker import chunk_pages
from src.ingestion.context_generator import generate_contextual_prefix
from src.ingestion.dedup import deduplicate
from src.ingestion.embedder import Embedder, embed_chunks
from src.ingestion.parser import parse_pdf, parse_text
from src.ingestion.qdrant_store import ensure_collection, upsert_chunks

_EMBEDDING_DIM = 384


async def ingest_document(
    file_path: str,
    doc_metadata: dict,
    qdrant_client: AsyncQdrantClient,
    embedder: Embedder,
    settings: Settings,
) -> dict:
    """Run the full ingestion pipeline for a single document.

    doc_metadata: {doc_type, date, venue, shift_id}.
    Returns {"doc_type", "chunks_ingested", "chunks_deduplicated"}.
    Raises descriptively on any step failure.
    """
    doc_type = doc_metadata.get("doc_type", "unknown")

    if doc_type == "shift_log":
        pages = await parse_text(file_path, doc_type=doc_type)
    else:
        pages = await parse_pdf(file_path)
        for page in pages:
            if "doc_type" not in page:
                page["doc_type"] = doc_type

    raw_chunks = annotate_chunks(
        chunk_pages(pages, settings.chunk_size, settings.chunk_overlap), doc_type
    )
    deduped = deduplicate(raw_chunks)
    chunks_deduplicated = len(raw_chunks) - len(deduped)

    for chunk in deduped:
        prefix = await generate_contextual_prefix(
            chunk_content=chunk["content"],
            doc_type=doc_type,
            venue=doc_metadata.get("venue", ""),
            date=doc_metadata.get("date", ""),
            api_key=settings.openrouter_api_key,
            model=settings.context_model,
            base_url=settings.openrouter_base_url,
        )
        chunk["contextual_prefix"] = prefix

    deduped = add_bm25_text(deduped)
    deduped = await embed_chunks(deduped, embedder)

    await ensure_collection(qdrant_client, settings.qdrant_collection, _EMBEDDING_DIM)
    count = await upsert_chunks(qdrant_client, settings.qdrant_collection, deduped, doc_metadata)

    return {
        "doc_type": doc_type,
        "chunks_ingested": count,
        "chunks_deduplicated": chunks_deduplicated,
    }
