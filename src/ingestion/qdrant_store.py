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
    client: AsyncQdrantClient,
    collection: str,
    vector_size: int,
    *,
    vector_name: str | None = "dense",
    corpus_version: str | None = None,
    manifest_digest: str | None = None,
) -> None:
    """Create a collection or verify its vector shape and corpus identity."""
    existing = {c.name for c in (await client.get_collections()).collections}
    if collection not in existing:
        params = VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
            hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
        )
        await client.create_collection(
            collection_name=collection,
            vectors_config={vector_name: params} if vector_name else params,
        )
        return

    info = await client.get_collection(collection)
    vectors = info.config.params.vectors
    configured = vectors.get(vector_name) if isinstance(vectors, dict) else vectors
    if configured is None or configured.size != vector_size:
        raise ValueError(f"collection '{collection}' has incompatible vectors")
    if corpus_version or manifest_digest:
        await assert_collection_identity(
            client, collection, corpus_version=corpus_version,
            manifest_digest=manifest_digest,
        )


async def assert_collection_identity(
    client: AsyncQdrantClient,
    collection: str,
    *,
    corpus_version: str | None,
    manifest_digest: str | None,
) -> None:
    """Reject a non-empty collection belonging to another manifest."""
    points, _ = await client.scroll(
        collection_name=collection, limit=1, with_payload=True, with_vectors=False
    )
    if not points:
        return
    payload = points[0].payload or {}
    expected = {
        "corpus_version": corpus_version,
        "manifest_digest": manifest_digest,
    }
    for key, value in expected.items():
        if value is not None and payload.get(key) != value:
            raise ValueError(
                f"collection '{collection}' contains a different corpus ({key} mismatch)"
            )


def deterministic_point_id(corpus_version: str, source_id: str, chunk: dict) -> str:
    """Return the stable ID for one immutable manifest chunk.

    A source checksum in the manifest prevents content from changing in place;
    including the content hash here also makes the identity easy to audit.
    """
    identity = ":".join(
        [
            corpus_version,
            source_id,
            str(chunk.get("page_num", "")),
            str(chunk.get("chunk_index", "")),
            chunk.get("content_hash", ""),
        ]
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"rag-operational:{identity}"))


async def upsert_chunks(
    client: AsyncQdrantClient,
    collection: str,
    chunks: list[dict],
    doc_metadata: dict,
    *,
    corpus_version: str | None = None,
    manifest_digest: str | None = None,
    source_id: str | None = None,
) -> int:
    """Upsert chunks in batches; manifest callers receive stable point IDs."""
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
                "chunk_index": chunk.get("chunk_index"),
                "content": chunk.get("content", ""),
                "contextual_prefix": chunk.get("contextual_prefix", ""),
                "bm25_text": chunk.get("bm25_text", ""),
            }
            if corpus_version is not None:
                payload.update(
                    corpus_version=corpus_version,
                    manifest_digest=manifest_digest,
                    source_id=source_id,
                )
            point_id = (
                deterministic_point_id(corpus_version, source_id, chunk)
                if corpus_version and source_id
                else str(uuid.uuid4())
            )
            points.append(
                PointStruct(
                    id=point_id,
                    vector={"dense": chunk["embedding"]},
                    payload=payload,
                )
            )
        await client.upsert(collection_name=collection, points=points)
        total += len(batch)

    return total
