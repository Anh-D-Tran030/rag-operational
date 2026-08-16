"""Index SQL view schema metadata into Qdrant for schema-linking retrieval."""
from __future__ import annotations

import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.ingestion.embedder import Embedder

_SCHEMA_DESCRIPTIONS = [
    {
        "view_name": "v_shift_summary",
        "columns": ["venue", "date", "shift_type", "total_staff"],
        "description": (
            "Aggregated staff counts per venue, date, and shift type."
            " Use for staffing level queries."
        ),
        "example_query": (
            "SELECT venue, date, total_staff FROM v_shift_summary"
            " WHERE venue = 'Fairfield RSL'"
        ),
    },
    {
        "view_name": "v_incident_by_venue",
        "columns": ["venue", "date", "incident_type", "incident_count"],
        "description": (
            "Count of incidents grouped by venue, date, and incident type."
            " Use for incident frequency analysis."
        ),
        "example_query": (
            "SELECT venue, incident_type, incident_count FROM v_incident_by_venue"
            " WHERE date = '2026-05-28'"
        ),
    },
    {
        "view_name": "v_staffing_gap",
        "columns": ["venue", "date", "role", "count", "required_count", "gap"],
        "description": (
            "Staffing gap analysis: actual count minus required count per role."
            " Negative gap means understaffed."
        ),
        "example_query": "SELECT venue, role, gap FROM v_staffing_gap WHERE gap < 0",
    },
    {
        "view_name": "v_incident_severity",
        "columns": [
            "incident_id", "shift_id", "venue", "date",
            "incident_type", "description", "severity",
        ],
        "description": "High-severity incidents only. Use for safety and compliance queries.",
        "example_query": "SELECT venue, date, description FROM v_incident_severity",
    },
    {
        "view_name": "v_peak_staffing",
        "columns": ["shift_id", "venue", "date", "staff_count", "shift_type", "avg_staff"],
        "description": (
            "Shifts where staff count exceeded the venue average."
            " Use for peak demand analysis."
        ),
        "example_query": "SELECT venue, date, staff_count, avg_staff FROM v_peak_staffing",
    },
]


async def index_schema_metadata(
    client: AsyncQdrantClient,
    schema_collection: str,
    embedder: Embedder,
    db_url: str,
) -> int:
    """Embed and upsert schema metadata for each view into schema_collection.

    Returns count of schema entries indexed.
    """
    existing = {c.name for c in (await client.get_collections()).collections}
    if schema_collection not in existing:
        sample_emb = embedder.embed_batch(["test"])
        vector_size = len(sample_emb[0])
        await client.create_collection(
            collection_name=schema_collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    texts = [
        f"{s['view_name']}: {s['description']} Columns: {', '.join(s['columns'])}"
        for s in _SCHEMA_DESCRIPTIONS
    ]
    embeddings = embedder.embed_batch(texts)

    # Derive the id from the view name so re-indexing overwrites rather than
    # accumulates. With random ids, repeat runs filled the top-k with duplicates of
    # one view and starved the schema linker of the others.
    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, schema["view_name"])),
            vector=emb,
            payload=schema,
        )
        for schema, emb in zip(_SCHEMA_DESCRIPTIONS, embeddings)
    ]
    await client.upsert(collection_name=schema_collection, points=points)
    return len(points)
