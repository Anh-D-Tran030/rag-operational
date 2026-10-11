"""End-to-end query handler: cache → route → retrieve/sql → generate → respond."""
from __future__ import annotations

import asyncio
import logging
import re
import time

from qdrant_client import AsyncQdrantClient

from src.config import Settings
from src.generation.alignment_gate import (
    DETERMINISTIC_REJECT_REASONS,
    AlignmentDecision,
    check_alignment,
)
from src.generation.cloud_llm import generate_cloud
from src.generation.faithfulness_gate import (
    GATE_ABSTENTION,
    GateDecision,
    check_faithfulness,
    record_gate_decision,
    render_evidence,
)
from src.generation.generation_result import GenerationResult, parse_generation_result
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
from src.retrieval.schema_linker import complete_schema_views, find_relevant_views
from src.retrieval.text_to_sql import text_to_sql_tool

logger = logging.getLogger(__name__)

ROUTER_TIMEOUT_S = 15.0
SCHEMA_EMBED_TIMEOUT_S = 15.0
SCHEMA_SEARCH_TIMEOUT_S = 15.0
SQL_FINAL_GENERATION_TIMEOUT_S = 60.0
# One judge budget on every route (formerly SQL-only). Expiry fails closed.
FAITHFULNESS_TIMEOUT_S = 30.0
ALIGNMENT_TIMEOUT_S = 15.0
_SCALAR_COLUMN = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
# Disjoint stages whose sum, subtracted from the total, gives unattributed time.
# schema_linking_ms already contains schema embedding and schema search.
_ATTRIBUTED_MS = (
    "router_ms", "retrieval_ms", "reranking_ms", "schema_linking_ms",
    "final_generation_ms", "faithfulness_ms", "alignment_ms",
)
_SQL_ATTRIBUTED_MS = ("generation_ms", "validation_ms", "execution_ms")
# A bare "Yes."/"No." omits the subject/date/role/count the question asked
# for; the prompt asks for a full sentence, but the model does not always
# comply, so this catches what the prompt missed.
_BARE_YES_NO = re.compile(r"^\s*(yes|no)[.!]?\s*$", re.IGNORECASE)


def _is_bare_yes_no(answer: str) -> bool:
    return bool(_BARE_YES_NO.fullmatch(answer))


def _direct_sql_answer(rows: list[dict]) -> str | None:
    """Render one scalar SQL value without introducing claims beyond the row."""
    if len(rows) != 1 or len(rows[0]) != 1:
        return None
    key, value = next(iter(rows[0].items()))
    if not isinstance(key, str) or not _SCALAR_COLUMN.fullmatch(key):
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    rendered = str(value).strip()
    if not rendered or len(rendered) > 160 or "\n" in rendered:
        return None
    return f"{key.replace('_', ' ').capitalize()}: {rendered}."


def _with_unattributed(components: dict, latency_ms: float) -> dict:
    """Reconcile stage timings against the total so unmeasured time stays visible."""
    sql = components.get("sql") or {}
    attributed = sum(float(components.get(k) or 0.0) for k in _ATTRIBUTED_MS)
    attributed += sum(float(sql.get(k) or 0.0) for k in _SQL_ATTRIBUTED_MS)
    return {**components, "unattributed_ms": round(latency_ms - attributed, 2)}


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
      6. Judge faithfulness; anything but `passed` becomes GATE_ABSTENTION.
      7. Record metrics, close trace, cache (unless the judge could not run), return.

    Returns:
      {answer, citations, trace_id, route, tool, latency_ms, latency_components}
      plus the internal `faithfulness_gate` decision, which is never cached and
      never exposed by the public /query schema.
    """
    t_total = time.monotonic()

    # Redact PII before cache lookup so cached entries never contain raw PII.
    # Timing is captured here; the Langfuse span is logged after start_trace
    # because trace_obj doesn't exist until then.
    t_pii_in = time.monotonic()
    query = redact_pii(query, label="input")
    _pii_in_ms = (time.monotonic() - t_pii_in) * 1000

    # A pre-contract entry with no `abstained` key at all cannot be served: it
    # may predate the gate this code now depends on, so it is a miss, not a
    # guess. `cached.get("abstained")` used to treat "key absent" the same as
    # "key present and false", silently serving a legacy response as if it had
    # been through the current gate.
    cached = await cache.get(query, filters=filters)
    # Only an explicit boolean counts: a missing or non-boolean flag (truthy
    # strings, 0/1, None) is a miss that re-enters routing and the gates, never
    # inferred from the answer's prose.
    if isinstance(cached, dict) and isinstance(cached.get("abstained"), bool):
        # Use the explicit flag recorded when this entry was written, never a
        # prose guess: a stored answer that happens to start with "insufficient"
        # is not necessarily an abstention, and vice versa.
        if cached["abstained"]:
            return {**cached, "citations": []}
        return cached

    trace_id, trace_obj = tracer.start_trace(name="query", user_query=query)
    tracer.log_span(
        trace_obj=trace_obj,
        name="pii_redaction",
        input={"label": "input"},
        output={"processed": True},
        latency_ms=_pii_in_ms,
        metadata={"stage": "input"},
    )

    t0 = time.monotonic()
    try:
        intent = await asyncio.wait_for(
            classify_intent(query, settings.local_llm(settings.router_model)),
            timeout=ROUTER_TIMEOUT_S,
        )
        router_timed_out = False
    except TimeoutError:
        intent = {"tool": "hybrid", "complexity": "complex"}
        router_timed_out = True
    router_ms = (time.monotonic() - t0) * 1000
    rag_latency_by_component.labels(component="route").observe(router_ms / 1000)

    tool = intent["tool"]
    complexity = intent["complexity"]
    rag_route_decision.labels(route=tool).inc()

    chunks: list[dict] = []
    sql_result: dict | None = None
    vector_citations: list[dict] = []
    sql_citations: list[dict] = []
    # Every return path reports the same keys, so early exits keep stage timings.
    lc: dict = {
        "router_ms": round(router_ms, 2), "router_timed_out": router_timed_out,
        "retrieval_ms": 0.0, "reranking_ms": 0.0, "schema_linking_ms": 0.0,
        "schema_embedding_ms": 0.0, "schema_embedding_timed_out": False,
        "schema_search_ms": 0.0, "schema_search_timed_out": False, "sql": {},
        "final_generation_ms": 0.0, "final_generation_timed_out": False,
        "faithfulness_ms": 0.0, "faithfulness_timed_out": False, "alignment_ms": 0.0,
        "direct_sql_response": False, "sql_schema_fallback": False,
        "generation_malformed": False,
    }

    def _respond(answer: str, citations: list[dict], **extra: object) -> dict:
        latency_ms = (time.monotonic() - t_total) * 1000
        return {
            "answer": answer, "citations": citations, **extra,
            "trace_id": trace_id, "route": complexity, "tool": tool,
            "latency_ms": latency_ms, "latency_components": _with_unattributed(lc, latency_ms),
        }

    async def _retrieve() -> tuple[list[dict], list[dict]]:
        t0 = time.monotonic()
        chunks, retrieval_metadata = await vector_search_pipeline(
            query=query,
            client=qdrant_client,
            settings=settings,
            tracer=tracer,
            trace_id=trace_id,
            complexity=complexity,
            filters=filters,
        )
        rag_latency_by_component.labels(component="retrieval").observe(time.monotonic() - t0)
        lc["retrieval_ms"] = retrieval_metadata.get("retrieval_ms", 0.0)
        lc["reranking_ms"] = retrieval_metadata.get("reranking_ms", 0.0)
        citations = [
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
        return chunks, citations

    if tool in {"vector_search", "hybrid"}:
        chunks, vector_citations = await _retrieve()

    if tool in {"text_to_sql", "hybrid"}:
        t0 = time.monotonic()
        schema_started = time.monotonic()
        embedding_started = time.monotonic()
        try:
            query_embedding = await asyncio.wait_for(
                embed_query(query, settings.embedding_model), timeout=SCHEMA_EMBED_TIMEOUT_S
            )
        except TimeoutError:
            query_embedding = []
            lc["schema_embedding_timed_out"] = True
        lc["schema_embedding_ms"] = round((time.monotonic() - embedding_started) * 1000, 2)
        rag_latency_by_component.labels(component="sql_schema_embedding").observe(
            lc["schema_embedding_ms"] / 1000
        )
        search_started = time.monotonic()
        if query_embedding:
            try:
                relevant_views = await asyncio.wait_for(
                    find_relevant_views(
                        query=query,
                        query_embedding=query_embedding,
                        client=qdrant_client,
                        schema_collection=settings.qdrant_schema_collection,
                    ), timeout=SCHEMA_SEARCH_TIMEOUT_S,
                )
            except TimeoutError:
                relevant_views = []
                lc["schema_search_timed_out"] = True
        else:
            relevant_views = []
        lc["schema_search_ms"] = round((time.monotonic() - search_started) * 1000, 2)
        rag_latency_by_component.labels(component="sql_schema_search").observe(
            lc["schema_search_ms"] / 1000
        )
        lc["schema_linking_ms"] = round((time.monotonic() - schema_started) * 1000, 2)
        # The same trusted concept completion the linker applies, so a bypassed or
        # timed-out lookup still offers the views the question explicitly needs.
        relevant_views = complete_schema_views(query, relevant_views)
        sql_result = await text_to_sql_tool(
            query=query,
            relevant_views=relevant_views,
            settings=settings,
            trace_id=trace_id,
            simple_sql=tool == "text_to_sql" and complexity == "simple",
        )
        rag_latency_by_component.labels(component="sql").observe(time.monotonic() - t0)
        lc["sql"] = sql_result.get("latency", {})
        if sql_result.get("schema_status") in {
            "unsupported_by_schema", "partial_schema_support",
        }:
            answer = f"insufficient context: {sql_result['explanation']}"
            tracer.end_trace(trace_obj, output=answer, usage={})
            return _respond(
                answer, [], rows=[], schema_status=sql_result["schema_status"],
                missing_schema=sql_result["missing_schema"],
            )
        if sql_result and sql_result.get("rows") is not None:
            rows = sql_result.get("rows", [])
            sql_citations = [
                {
                    "view_name": ", ".join(sql_result.get("view_names", [])),
                    "sql_used": sql_result.get("sql_used", ""),
                    "row_count": len(rows), "rows": rows[: settings.rerank_top_n],
                    "section_ids": [a for a in (row_anchor(r) for r in rows) if a],
                }
            ]
        # The SQL needed a fact no curated view stores, so the documents are the
        # only source. A valid query that found zero rows is not this signal.
        # Both fields are required: failure_kind is authoritative, and an alias-only
        # rejection (candidate_alias_rejected) keeps fallback_reason solely for
        # compatibility, so that field alone or a missing failure_kind never falls back.
        if (
            tool == "text_to_sql"
            and sql_result.get("fallback_reason") == "schema_capability_gap"
            and sql_result.get("failure_kind") == "schema_capability_gap"
        ):
            lc["sql_schema_fallback"] = True
            chunks, vector_citations = await _retrieve()

    if not chunks and (not sql_result or not sql_result.get("rows")):
        gate = GateDecision("skipped_no_context")  # the generator is never called
        record_gate_decision(gate, tracer, trace_obj, 0.0)
        tracer.end_trace(trace_obj, output="insufficient context", usage={})
        return {**_respond("insufficient context", []), "faithfulness_gate": gate.as_dict()}

    direct_answer = (
        _direct_sql_answer(sql_result["rows"])
        if tool == "text_to_sql" and complexity == "simple" and sql_result
        else None
    )
    prompt = ""
    # Set only when the generator itself signalled/failed to answer: an explicit
    # is_abstention=true, or a malformed/timed-out reply. Distinct from a later
    # alignment/faithfulness rejection, which produces GATE_ABSTENTION instead.
    explicit_generated_abstention = False
    generation_cacheable: bool | None = None
    if direct_answer is not None:
        answer = direct_answer
        lc["direct_sql_response"] = True
    else:
        if tool == "hybrid" or (sql_result and sql_result.get("rows")):
            prompt = build_hybrid_prompt(query, chunks, sql_result)
        else:
            prompt = build_vector_rag_prompt(query, chunks)

        t0 = time.monotonic()
        use_cloud = complexity == "complex" or tool == "hybrid"

        def _call() -> object:
            if use_cloud and settings.openrouter_api_key:
                return generate_cloud(
                    prompt=prompt,
                    model=settings.cloud_model,
                    api_key=settings.openrouter_api_key,
                    base_url=settings.openrouter_base_url,
                )
            return generate_local(prompt=prompt, config=settings.local_llm())

        async def _generate() -> str:
            operation = _call()
            if tool == "text_to_sql":
                return await asyncio.wait_for(operation, timeout=SQL_FINAL_GENERATION_TIMEOUT_S)
            return await operation

        lc["final_generation_retried_bare_yes_no"] = False
        lc["generation_malformed"] = False
        final_generation_exception_class: str | None = None
        generation_result: GenerationResult | None = None
        try:
            generation_result = parse_generation_result(str(await _generate()))
            # A bare yes/no is only a retry trigger for an explicit substantive
            # answer; an explicit abstention or a malformed reply is terminal.
            if (
                generation_result is not None
                and not generation_result.is_abstention
                and _is_bare_yes_no(generation_result.answer)
            ):
                lc["final_generation_retried_bare_yes_no"] = True
                generation_result = parse_generation_result(str(await _generate()))
                if (
                    generation_result is not None
                    and not generation_result.is_abstention
                    and _is_bare_yes_no(generation_result.answer)
                ):
                    generation_result = GenerationResult(is_abstention=True, answer="")
        except TimeoutError as exc:
            # A timeout is a provider/infra failure, not a real abstention: it
            # must be distinguishable in the raw record from is_abstention=true
            # or malformed JSON, the same way a RuntimeError is below.
            lc["final_generation_timed_out"] = True
            final_generation_exception_class = type(exc).__name__
            generation_result = None
        except RuntimeError as exc:
            # generate_cloud wraps every provider failure (after its own
            # bounded response_format fallback) as RuntimeError. This must
            # become a safe refusal here, never propagate to the API route as
            # an unhandled exception (which would surface as HTTP 500). Only
            # the exception class is logged, never its message, so a raw
            # provider error never reaches logs or the response.
            logger.warning("final generation provider failure: %s", type(exc).__name__)
            final_generation_exception_class = type(exc).__name__
            generation_result = None
        lc["final_generation_ms"] = round((time.monotonic() - t0) * 1000, 2)
        rag_latency_by_component.labels(component="generation").observe(
            lc["final_generation_ms"] / 1000
        )

        if generation_result is None:
            # Malformed JSON, a missing/non-bool flag, an empty false-answer, a
            # provider error, or a timeout: a safe refusal, never the raw error,
            # never cached.
            lc["generation_malformed"] = True
            answer = "insufficient context"
            explicit_generated_abstention = True
            generation_cacheable = False
        elif generation_result.is_abstention:
            answer = "insufficient context"
            explicit_generated_abstention = True
            generation_cacheable = True
        else:
            answer = generation_result.answer

    # Query-evidence alignment: does the evidence concern the requested entity,
    # date, role and operation, not just a fluent-sounding neighbor of it? Only
    # runs for a non-abstention draft; an abstention has no evidence to align.
    # A reject/indeterminate/judge_error/timeout skips faithfulness entirely so
    # it is never mislabeled "passed", and cacheability is decided separately
    # below since skipped_due_to_alignment is not in faithfulness_gate's
    # cacheable set.
    alignment_cacheable: bool | None = None
    # Internal-only alignment diagnostic (outcome, reason, exception class name):
    # never cached and not part of the public /query schema.
    alignment_info: dict[str, str] | None = None
    lc["alignment_ms"] = 0.0
    if explicit_generated_abstention:
        # The generator itself already said it cannot answer (or failed to say
        # anything parseable); there is no factual claim left to align or judge.
        gate = GateDecision("skipped_generated_abstention", final_generation_exception_class)
        record_gate_decision(gate, tracer, trace_obj, 0.0)
    else:
        # explicit_generated_abstention is False here, so `answer` is a genuine
        # draft the generator claims is substantive; alignment must check every
        # such draft, never guess from its prose that it was secretly a refusal.
        align_started = time.monotonic()
        try:
            alignment = await asyncio.wait_for(
                check_alignment(
                    query=query,
                    answer=str(answer),
                    chunks=chunks,
                    sql_result=sql_result,
                    filters=filters,
                    model=settings.gate_model,
                    api_key=settings.openrouter_api_key,
                    base_url=settings.openrouter_base_url,
                    token_budget=settings.token_budget_per_query,
                ),
                timeout=ALIGNMENT_TIMEOUT_S,
            )
        except TimeoutError as exc:
            alignment = AlignmentDecision("timeout", exception_class=type(exc).__name__)
        alignment_info = alignment.as_dict()
        lc["alignment_ms"] = round((time.monotonic() - align_started) * 1000, 2)

        if not alignment.passed:
            gate = GateDecision("skipped_due_to_alignment")
            # Allowlist, not an exclusion list: only a reason that is a known
            # deterministic hard-constraint mismatch is safe to replay. A
            # semantic-judge rejection, an outage (judge_error/timeout), an
            # unresolved indeterminate, or any reason not in the allowlist
            # (including None/unknown) must not, per the alignment contract's
            # deterministic-only cache rule -- a `Literal` type hint alone does
            # not enforce this at runtime.
            alignment_cacheable = (
                alignment.outcome == "rejected"
                and alignment.reason in DETERMINISTIC_REJECT_REASONS
            )
            record_gate_decision(gate, tracer, trace_obj, lc["alignment_ms"])
        else:
            # Even a direct SQL answer is judged: only `passed` may deliver an answer.
            gate_started = time.monotonic()
            try:
                gate = await asyncio.wait_for(
                    check_faithfulness(
                        answer=str(answer),
                        evidence=render_evidence(
                            chunks, sql_result, settings.token_budget_per_query
                        ),
                        model=settings.gate_model,
                        api_key=settings.openrouter_api_key,
                        base_url=settings.openrouter_base_url,
                    ),
                    timeout=FAITHFULNESS_TIMEOUT_S,
                )
            except TimeoutError as exc:
                gate = GateDecision("timeout", type(exc).__name__)
            lc["faithfulness_ms"] = round((time.monotonic() - gate_started) * 1000, 2)
            lc["faithfulness_timed_out"] = gate.outcome == "timeout"
            rag_latency_by_component.labels(component="faithfulness").observe(
                lc["faithfulness_ms"] / 1000
            )
            record_gate_decision(gate, tracer, trace_obj, lc["faithfulness_ms"])
    # An explicit generated abstention already reads as plain "insufficient
    # context"; only an alignment/faithfulness rejection gets the generic
    # GATE_ABSTENTION wording.
    if not gate.passed and not explicit_generated_abstention:
        answer = GATE_ABSTENTION
    t_pii_out = time.monotonic()
    # Names copied verbatim from queried rows or retrieved text are the answer,
    # not leaked PII; only spans the generator introduced are redacted.
    evidence = [str(v) for r in (sql_result or {}).get("rows") or [] for v in r.values()]
    evidence += [str(c.get("content", "")) for c in chunks]
    answer = redact_pii(str(answer), label="output", evidence=evidence)
    # An abstention is a safety response, not evidence for an answer.  Generated
    # abstentions can occur after retrieval, so clear provisional evidence here.
    # `gate.passed` alone is authoritative: explicit_generated_abstention already
    # forces gate to a non-passing outcome above, so no prose guess is needed.
    if not gate.passed:
        vector_citations = []
        sql_citations = []
    tracer.log_span(
        trace_obj=trace_obj,
        name="pii_redaction",
        input={"label": "output"},
        output={"processed": True},
        latency_ms=(time.monotonic() - t_pii_out) * 1000,
        metadata={"stage": "output"},
    )

    est_cost = len(prompt.split()) * 0.000003 + len(str(answer).split()) * 0.000015
    rag_query_cost_usd.observe(est_cost)

    tracer.end_trace(trace_obj, output=answer, usage={"cost_usd": est_cost})

    # Stored explicitly so a future cache hit can tell abstention from a real
    # answer without guessing from the answer's wording (D-006/PLAN-010).
    response = _respond(
        answer, vector_citations + sql_citations,
        abstained=not gate.passed or explicit_generated_abstention,
    )
    # A refusal caused by a judge/generator that could not run must not replay.
    should_cache = (
        generation_cacheable if generation_cacheable is not None
        else alignment_cacheable if alignment_cacheable is not None
        else gate.cacheable
    )
    if should_cache:
        await cache.set(query, response, filters=filters)
    internal = {"faithfulness_gate": gate.as_dict()}
    if alignment_info is not None:
        internal["alignment_gate"] = alignment_info
    return {**response, **internal}
