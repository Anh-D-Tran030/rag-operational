"""Demo: wrong-but-runnable SQL is captured by the audit trail (PRD §4.3).

Sends an ambiguous quantitative query through text_to_sql_tool. The cloud LLM
may generate SQL that is syntactically valid AND executes cleanly, but produces
a result that does not answer the question (e.g. counts the wrong entity,
groups by the wrong dimension, or silently omits rows).

The demo shows what the audit trail actually captures on the returned response:

  * `sql_used` — the exact SQL string that was executed against the DB
  * `view_names` — the schema-linked views the LLM was allowed to query
  * `citations[0].sql_used` — the same SQL propagated to the caller-visible
    citation, so a human reviewer can compare it against the natural-language
    question after the fact

We also independently run three "oracle" queries directly against SQLite so a
reader can compare the LLM's row count against the ground truth.

Usage:
    .venv/bin/python scripts/demo_bad_sql_audit.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

from qdrant_client import AsyncQdrantClient

from src.config import Settings
from src.generation.query_handler import handle_query
from src.observability.tracer import RAGTracer
from src.retrieval.cache import SemanticCache
from src.retrieval.embedder import embed_query
from src.retrieval.schema_linker import find_relevant_views
from src.retrieval.text_to_sql import text_to_sql_tool

# Ambiguous count-style query. The corpus covers Fairfield RSL only, dates in
# 2026-05-09..2026-05-28, so "last week" has no natural anchor and there is no
# "seniority" column — good conditions for a plausible-looking but wrong SQL.
QUERY = (
    "How many incidents involving senior staff members happened last week "
    "at Fairfield RSL, and which shift type had the most?"
)

# Oracle probes against the local SQLite DB (not through the LLM).
ORACLE_SQL = [
    (
        "distinct_incident_types_all_time",
        "SELECT incident_type, COUNT(*) as n FROM incident_logs "
        "GROUP BY incident_type ORDER BY n DESC",
    ),
    (
        "total_incidents_by_shift_type",
        "SELECT s.shift_type, COUNT(i.incident_id) AS incidents "
        "FROM incident_logs i JOIN shift_records s USING (shift_id) "
        "GROUP BY s.shift_type ORDER BY incidents DESC",
    ),
    (
        "columns_that_could_encode_senior_staff",
        "SELECT name FROM pragma_table_info('shift_records') "
        "UNION SELECT name FROM pragma_table_info('incident_logs') "
        "UNION SELECT name FROM pragma_table_info('staffing_history')",
    ),
]


def oracle_probes(db_path: str) -> dict:
    """Run oracle SQL directly against the SQLite file for ground truth."""
    out: dict = {}
    con = sqlite3.connect(db_path)
    try:
        for name, sql in ORACLE_SQL:
            cur = con.execute(sql)
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            out[name] = {"sql": sql, "rows": rows}
    finally:
        con.close()
    return out


async def main() -> int:
    settings = Settings(
        qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        ollama_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        semantic_cache_url="",
    )
    if not settings.openrouter_api_key:
        print("OPENROUTER_API_KEY not set; cannot run demo.", file=sys.stderr)
        return 1

    qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    tracer = RAGTracer(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
    cache = SemanticCache(cache_url="", embedding_model=settings.embedding_model)

    print("=" * 72)
    print("DEMO: wrong-but-runnable SQL captured by audit trail")
    print("=" * 72)
    print(f"query: {QUERY}\n")

    # (1) direct text_to_sql_tool invocation — this is the surface an operator
    #     would inspect when reviewing what the LLM actually did.
    print("-- text_to_sql_tool (direct) --")
    query_embedding = await embed_query(QUERY, settings.embedding_model)
    relevant_views = await find_relevant_views(
        query=QUERY,
        query_embedding=query_embedding,
        client=qdrant,
        schema_collection=settings.qdrant_schema_collection,
    )
    print(f"schema-linked views:  {[v.get('view_name') for v in relevant_views]}")
    t0 = time.monotonic()
    sql_result = await text_to_sql_tool(
        query=QUERY,
        relevant_views=relevant_views,
        settings=settings,
        trace_id="demo-bad-sql-audit",
    )
    dt = time.monotonic() - t0
    print(f"latency_s:            {dt:.2f}")
    print(f"sql_used (executed):\n{sql_result.get('sql_used', '<none — error>')}")
    print(f"view_names:           {sql_result.get('view_names', [])}")
    rows = sql_result.get("rows", [])
    print(f"row_count:            {len(rows)}")
    if rows:
        print(f"rows (first 5):       {json.dumps(rows[:5], default=str)}")
    if sql_result.get("error"):
        print(f"error:                {sql_result['error']}")
    print()

    # (2) end-to-end via handle_query — shows how the audit trail surfaces to
    #     the caller as a citation entry with sql_used populated.
    print("-- handle_query end-to-end (audit surface = citation.sql_used) --")
    response = await handle_query(
        query=QUERY,
        qdrant_client=qdrant,
        settings=settings,
        tracer=tracer,
        cache=cache,
    )
    citations = response.get("citations", [])
    sql_citations = [c for c in citations if c.get("sql_used")]
    print(f"tool routed:          {response.get('tool')}")
    print(f"n_sql_citations:      {len(sql_citations)}")
    for i, c in enumerate(sql_citations, 1):
        print(f"  citation[{i}].view_name: {c.get('view_name')}")
        print(f"  citation[{i}].row_count: {c.get('row_count')}")
        print(f"  citation[{i}].sql_used:\n{c.get('sql_used')}")
    print(f"answer:               {response.get('answer', '')[:400]}")
    print()

    # (3) oracle probes for ground truth comparison.
    db_path = settings.db_url.split("///")[-1]
    print(f"-- oracle probes against {db_path} --")
    oracle = oracle_probes(db_path)
    for name, payload in oracle.items():
        print(f"[{name}]")
        for r in payload["rows"][:10]:
            print(f"  {r}")
    print()

    Path(".logs").mkdir(exist_ok=True)
    out = Path(".logs/demo_bad_sql_audit.json")
    out.write_text(
        json.dumps(
            {
                "query": QUERY,
                "schema_linked_views": [v.get("view_name") for v in relevant_views],
                "sql_tool_result": {
                    "sql_used": sql_result.get("sql_used"),
                    "view_names": sql_result.get("view_names"),
                    "row_count": len(rows),
                    "rows": rows,
                    "error": sql_result.get("error"),
                },
                "handle_query_response": {
                    "tool": response.get("tool"),
                    "answer": response.get("answer"),
                    "sql_citations": sql_citations,
                },
                "oracle": oracle,
            },
            indent=2,
            default=str,
        )
    )
    print(f"evidence written to: {out}")

    await qdrant.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
