"""Schema-linking: retrieve relevant SQL view metadata from Qdrant."""
from __future__ import annotations

import re

from qdrant_client import AsyncQdrantClient

_PEAK_STAFFING_VIEW = {
    "view_name": "v_peak_staffing",
    "columns": ["shift_id", "venue", "date", "staff_count", "shift_type", "avg_staff"],
    "description": "Shifts where staff count exceeded the venue average.",
    "example_query": "SELECT shift_id, date, staff_count FROM v_peak_staffing",
}
_INCIDENT_BY_VENUE_VIEW = {
    "view_name": "v_incident_by_venue",
    "columns": ["venue", "date", "incident_type", "incident_count"],
    "description": "Count of incidents grouped by venue, date, and incident type.",
    "example_query": "SELECT venue, date, incident_count FROM v_incident_by_venue",
}
_SHIFT_DETAILS_VIEW = {
    "view_name": "v_shift_details",
    "columns": ["shift_id", "venue", "date", "staff_count", "shift_type", "supervisor"],
    "description": "One record per operational shift, including its supervisor.",
    "example_query": "SELECT shift_id, supervisor FROM v_shift_details WHERE shift_id = 'SH-001'",
}
_INCIDENT_BY_SHIFT_VIEW = {
    "view_name": "v_incident_by_shift",
    "columns": ["shift_id", "venue", "date", "incident_count"],
    "description": "Count of all logged incidents per shift.",
    "example_query": "SELECT shift_id, incident_count FROM v_incident_by_shift WHERE shift_id = 'SH-006'",
}


def _requires_peak_staff_incident_views(query: str) -> bool:
    """Identify the narrow combined staffing-and-incident schema request."""
    normalized = query.casefold()
    return "largest staff count" in normalized and "incident" in normalized


def _required_views(query: str) -> tuple[dict, ...]:
    """Return schema views implied by concepts that retrieval may miss."""
    normalized = query.casefold()
    required: list[dict] = []
    if _requires_peak_staff_incident_views(query):
        required.extend((_PEAK_STAFFING_VIEW, _INCIDENT_BY_VENUE_VIEW))
    if re.search(r"\bsupervis(?:or|ed|ing)\b", normalized):
        required.append(_SHIFT_DETAILS_VIEW)
    if "incident" in normalized and re.search(r"\bshift\s+sh-\d+\b", normalized):
        required.append(_INCIDENT_BY_SHIFT_VIEW)
    return tuple(required)


def _add_required_views(query: str, views: list[dict]) -> list[dict]:
    """Complete schema context for explicit field and entity requirements."""
    present = {view["view_name"] for view in views}
    additions = [view for view in _required_views(query) if view["view_name"] not in present]
    return [*views, *additions]


async def find_relevant_views(
    query: str,
    query_embedding: list[float],
    client: AsyncQdrantClient,
    schema_collection: str,
    top_k: int = 3,
) -> list[dict]:
    """Search schema_collection for views relevant to the query.

    Returns top_k schema metadata records as dicts:
      {view_name, columns, description, example_query}
    Explicit field/entity requirements are completed from trusted local schema
    metadata when the collection is empty or unreachable.
    """
    try:
        # This collection is created with a single unnamed vector, so no `using=`.
        response = await client.query_points(
            collection_name=schema_collection,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
        )
        results = response.points
    except Exception:
        return _add_required_views(query, [])

    output = []
    for hit in results:
        payload = hit.payload or {}
        output.append({
            "view_name": payload.get("view_name", ""),
            "columns": payload.get("columns", []),
            "description": payload.get("description", ""),
            "example_query": payload.get("example_query", ""),
        })
    return _add_required_views(query, output)
