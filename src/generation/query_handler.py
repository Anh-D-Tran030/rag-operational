"""End-to-end query handler: cache → route → retrieve/sql → generate → respond."""
from __future__ import annotations

import time

from qdrant_client import AsyncQdrantClient

from src.config import Settings
from src.generation.cloud_llm import generate_cloud
from src.generation.faithfulness_gate import check_faithfulness
from src.generation.local_llm import generate_local
from src.generation.prompts import build_hybrid_prompt, build_vector_rag_prompt
from src.generation.router import classify_intent
from src.guardrails.pii_redactor import redact_pii
from src.ingestion.anchors import row_anchor
from src.observability.metrics import (
    rag_latency_by_component,
    rag_query_cost_usd,
    rag_route_decision,
)
from src.observability.tracer import RAGTracer
from src.retrieval.cache import SemanticCache
from src.retrieval.embedder import embed_query
from src.retrieval.pipeline import vector_search_pipeline
from src.retrieval.schema_linker import find_relevant_views
from src.retrieval.text_to_sql import text_to_sql_tool


async def handle_query(
    query: str,
    qdrant_client: AsyncQdrantClient,
    settings: Settings,
    tracer: RAGTracer,
    cache: SemanticCache,
    filters: dict | None = None,
) -> dict:
    """Handle a user query end-to-end and return a structured response.

    Steps:
      1. Check semantic cache — return cached response on hit.
      2. Start Langfuse trace.
      3. Classify intent (vector_search / text_to_sql / hybrid).
      4. Run vector retrieval and/or SQL depending on route.
      5. Build prompt and generate answer (local for simple, cloud for complex/hybrid).
      6. Record metrics and close trace.
      7. Cache and return response.

    Returns:
      {answer, citations, trace_id, route, tool, latency_ms}
    """
    t_total = time.monotonic()

    query = redact_pii(query, label="input")

    cached = await cache.get(query)
    if cached is not None:
        return cached

    trace_id, trace_obj = tracer.start_trace(name="query", user_query=query)

    t0 = time.monotonic()
    intent = await classify_intent(query, settings.local_llm(settings.router_model))
    rag_latency_by_component.labels(component="route").observe(time.monotonic() - t0)

    tool = intent["tool"]
    complexity = intent["complexity"]
    rag_route_decision.labels(route=tool).inc()

    chunks: list[dict] = []
    sql_result: dict | None = None
    vector_citations: list[dict] = []
    sql_citations: list[dict] = []

    if tool in {"vector_search", "hybrid"}:
        t0 = time.monotonic()
        chunks, _ = await vector_search_pipeline(
            query=query,
            client=qdrant_client,
            settings=settings,
            tracer=tracer,
            trace_id=trace_id,
            complexity=complexity,
            filters=filters,
        )
        rag_latency_by_component.labels(component="retrieval").observe(time.monotonic() - t0)
        vector_citations = [
            {
                "doc_type": c.get("doc_type"),
                "venue": c.get("venue"),
                "date": c.get("date"),
                "shift_id": c.get("shift_id"),
                "page_num": c.get("page_num"),
                "chunk_id": c.get("chunk_id", ""),
                "section_ids": c.get("section_ids", []) or [],
                "snippet": c.get("content", "")[:200],
                "score": c.get("rerank_score", c.get("score", 0.0)),
            }
            for c in chunks
        ]

    if tool in {"text_to_sql", "hybrid"}:
        t0 = time.monotonic()
        query_embedding = await embed_query(query, settings.embedding_model)
        relevant_views = await find_relevant_views(
            query=query,
            query_embedding=query_embedding,
            client=qdrant_client,
            schema_collection=settings.qdrant_schema_collection,
        )
        sql_result = await text_to_sql_tool(
            query=query,
            relevant_views=relevant_views,
            settings=settings,
            trace_id=trace_id,
        )
        rag_latency_by_component.labels(component="sql").observe(time.monotonic() - t0)
        if sql_result and sql_result.get("rows") is not None:
            rows = sql_result.get("rows", [])
            sql_citations = [
                {
                    "view_name": ", ".join(sql_result.get("view_names", [])),
                    "sql_used": sql_result.get("sql_used", ""),
                    "row_count": len(rows),
                    "section_ids": [a for a in (row_anchor(r) for r in rows) if a],
                }
            ]

    if not chunks and (not sql_result or not sql_result.get("rows")):
        response = {
            "answer": "insufficient context",
            "citations": [],
            "trace_id": trace_id,
            "route": complexity,
            "tool": tool,
            "latency_ms": (time.monotonic() - t_total) * 1000,
        }
        tracer.end_trace(trace_obj, output="insufficient context", usage={})
        return response

    if tool == "hybrid" or (sql_result and sql_result.get("rows")):
        prompt = build_hybrid_prompt(query, chunks, sql_result)
    else:
        prompt = build_vector_rag_prompt(query, chunks)

    t0 = time.monotonic()
    use_cloud = complexity == "complex" or tool == "hybrid"
    if use_cloud and settings.openrouter_api_key:
        answer = await generate_cloud(
            prompt=prompt,
            model=settings.cloud_model,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
        )
    else:
        answer = await generate_local(prompt=prompt, config=settings.local_llm())
    rag_latency_by_component.labels(component="generation").observe(time.monotonic() - t0)

    faithful = await check_faithfulness(
        answer=answer,
        context_chunks=chunks,
        threshold=0.9,
        model=settings.gate_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )
    if not faithful:
        answer = (
            "The generated answer could not be verified against the retrieved context. "
            "Please rephrase your question or consult the source documents directly."
        )
    answer = redact_pii(str(answer), label="output")

    est_cost = len(prompt.split()) * 0.000003 + len(str(answer).split()) * 0.000015
    rag_query_cost_usd.observe(est_cost)

    latency_ms = (time.monotonic() - t_total) * 1000
    tracer.end_trace(trace_obj, output=answer, usage={"cost_usd": est_cost})

    response = {
        "answer": answer,
        "citations": vector_citations + sql_citations,
        "trace_id": trace_id,
        "route": complexity,
        "tool": tool,
        "latency_ms": latency_ms,
    }
    await cache.set(query, response)
    return response
