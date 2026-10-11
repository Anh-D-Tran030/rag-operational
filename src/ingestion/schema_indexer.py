"""Index SQL view schema metadata into Qdrant for schema-linking retrieval."""
from __future__ import annotations

import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.ingestion.embedder import Embedder
from src.ingestion.qdrant_store import ensure_collection

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
    {
        "view_name": "v_shift_details",
        "columns": ["shift_id", "venue", "date", "staff_count", "shift_type", "supervisor"],
        "description": (
            "One record per operational shift, including its supervisor. "
            "Use for shift-specific detail and supervisor queries."
        ),
        "example_query": (
            "SELECT shift_id, supervisor FROM v_shift_details "
            "WHERE shift_id = 'SH-001'"
        ),
    },
    {
        "view_name": "v_incident_by_shift",
        "columns": ["shift_id", "venue", "date", "incident_count"],
        "description": "Count of all logged incidents per shift. Use for incident totals by shift.",
        "example_query": (
            "SELECT shift_id, incident_count FROM v_incident_by_shift "
            "WHERE shift_id = 'SH-006'"
        ),
    },
    {
        "view_name": "v_shift_staffing_evidence",
        "columns": ["shift_id", "venue", "date", "role", "count", "required_count", "gap"],
        "description": (
            "Staffing evidence joined to shifts by venue and date, only where a staffing "
            "record exists. Negative gap means understaffed; absence of a row is not evidence "
            "that staffing met requirements."
        ),
        "example_query": "SELECT shift_id, role, gap FROM v_shift_staffing_evidence WHERE gap < 0",
    },
]


async def index_schema_metadata(
    client: AsyncQdrantClient,
    schema_collection: str,
    embedder: Embedder,
    db_url: str,
    *,
    schema_views: list[dict] | None = None,
    corpus_version: str | None = None,
    manifest_digest: str | None = None,
    embedding_dimension: int | None = None,
) -> int:
    """Embed and upsert schema metadata for each view into schema_collection.

    Returns count of schema entries indexed.
    """
    views = schema_views or _SCHEMA_DESCRIPTIONS
    sample_emb = embedder.embed_batch(["test"])
    vector_size = embedding_dimension or len(sample_emb[0])
    if corpus_version:
        await ensure_collection(
            client,
            schema_collection,
            vector_size,
            vector_name=None,
            corpus_version=corpus_version,
            manifest_digest=manifest_digest,
        )
    else:
        existing = {c.name for c in (await client.get_collections()).collections}
        if schema_collection not in existing:
            await client.create_collection(
                collection_name=schema_collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    texts = [
        f"{s['view_name']}: {s['description']} Columns: {', '.join(s['columns'])}"
        for s in views
    ]
    embeddings = embedder.embed_batch(texts)

    points = []
    for schema, emb in zip(views, embeddings):
        identity = schema["view_name"]
        payload = dict(schema)
        if corpus_version:
            identity = f"{corpus_version}:{manifest_digest}:{identity}"
            payload.update(
                corpus_version=corpus_version,
                manifest_digest=manifest_digest,
            )
        points.append(
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, identity)),
                vector=emb,
                payload=payload,
            )
        )
    await client.upsert(collection_name=schema_collection, points=points)
    return len(points)
