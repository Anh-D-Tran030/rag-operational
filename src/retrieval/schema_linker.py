"""Schema-linking: retrieve relevant SQL view metadata from Qdrant."""
from __future__ import annotations

from qdrant_client import AsyncQdrantClient


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
    Returns [] if the collection is empty or unreachable.
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
        return []

    output = []
    for hit in results:
        payload = hit.payload or {}
        output.append({
            "view_name": payload.get("view_name", ""),
            "columns": payload.get("columns", []),
            "description": payload.get("description", ""),
            "example_query": payload.get("example_query", ""),
        })
    return output
