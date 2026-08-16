"""Qdrant collection management and chunk upsert."""
from __future__ import annotations

import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PointStruct,
    VectorParams,
)


async def ensure_collection(
    client: AsyncQdrantClient, collection: str, vector_size: int
) -> None:
    """Create the Qdrant collection if it does not exist.

    Dense vector: 'dense', Cosine, HNSW m=16 ef_construct=100.
    BM25 is implemented client-side via rank_bm25 over the bm25_text payload field.
    """
    existing = {c.name for c in (await client.get_collections()).collections}
    if collection in existing:
        return

    await client.create_collection(
        collection_name=collection,
        vectors_config={
            "dense": VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
                hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
            )
        },
    )


async def upsert_chunks(
    client: AsyncQdrantClient,
    collection: str,
    chunks: list[dict],
    doc_metadata: dict,
) -> int:
    """Upsert chunk embeddings and payloads into Qdrant in batches of 100.

    doc_metadata must contain: doc_type, date, venue, shift_id.
    Each point payload includes all 10 required fields.
    Returns count of upserted points.
    """
    batch_size = 100
    total = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        points = []
        for chunk in batch:
            payload = {
                "page_num": chunk.get("page_num"),
                "doc_type": doc_metadata.get("doc_type", chunk.get("doc_type", "")),
                "date": doc_metadata.get("date", ""),
                "venue": doc_metadata.get("venue", ""),
                "shift_id": doc_metadata.get("shift_id", ""),
                "table_flag": chunk.get("table_flag", False),
                "content_hash": chunk.get("content_hash", ""),
                "chunk_id": chunk.get("chunk_id", ""),
                "section_ids": chunk.get("section_ids", []),
                "doc_key": chunk.get("doc_key", ""),
                "content": chunk.get("content", ""),
                "contextual_prefix": chunk.get("contextual_prefix", ""),
                "bm25_text": chunk.get("bm25_text", ""),
            }
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={"dense": chunk["embedding"]},
                    payload=payload,
                )
            )
        await client.upsert(collection_name=collection, points=points)
        total += len(batch)

    return total
