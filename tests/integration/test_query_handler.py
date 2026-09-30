"""Integration tests for src/generation/query_handler.py."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.routes.query import QueryRequest
from src.api.routes.query import query as query_route
from src.config import Settings
from src.evaluation.sql_ground_truth import load_golden_cases
from src.generation.alignment_gate import AlignmentDecision
from src.generation.faithfulness_gate import GateDecision
from src.generation.query_handler import _direct_sql_answer, handle_query
from src.ingestion.schema_indexer import _SCHEMA_DESCRIPTIONS
from src.observability.tracer import RAGTracer
from src.retrieval.cache import SemanticCache
from src.retrieval.text_to_sql import CURATED_VIEW_COLUMNS

_SQL_GOLDEN = load_golden_cases("eval_data/golden_dataset.json")
# The judge is always mocked explicitly: an environment with no key abstains.
_GATE = "src.generation.query_handler.check_faithfulness"
_PASSED = GateDecision("passed")
# The alignment gate runs before faithfulness and is mocked the same way, so a
# real (network) judge call never happens in this file's unit-scoped tests.
_ALIGN = "src.generation.query_handler.check_alignment"
_ALIGNED = AlignmentDecision("passed")


def _wire(answer: str, is_abstention: bool = False) -> str:
    """Build a raw generator reply matching the GenerationResult JSON contract."""
    import json

    return json.dumps({"is_abstention": is_abstention, "answer": answer})


def _make_chunk(content: str = "The minimum staffing is 5.", doc_type: str = "compliance") -> dict:
    return {
        "content": content,
        "doc_type": doc_type,
        "venue": "RSL",
        "date": "2026-01-01",
        "shift_id": None,
        "page_num": 1,
        "rerank_score": 0.85,
        "content_hash": "abc123",
        "score": 0.85,
    }


def test_direct_sql_answer_only_accepts_one_safe_scalar() -> None:
    assert _direct_sql_answer([{"staff_count": 17}]) == "Staff count: 17."
    assert _direct_sql_answer([{"shift_id": "SH-006", "staff_count": 20}]) is None
    assert _direct_sql_answer([{"staff_count": 17}, {"staff_count": 18}]) is None
    assert _direct_sql_answer([{"met": True}]) is None
    assert _direct_sql_answer([{"answer": "first\nsecond"}]) is None


@pytest.fixture
def settings():
    return Settings(
        anthropic_api_key="",
        openai_api_key="",
        ollama_url="http://localhost:11434",
        db_url="sqlite+aiosqlite:///./operational.db",
        semantic_cache_url="",
    )


@pytest.fixture
def tracer():
    return RAGTracer(public_key="", secret_key="", host="")


@pytest.fixture
def cache(settings):
    return SemanticCache("", settings.embedding_model)


@pytest.mark.asyncio
async def test_vector_query_happy_path(settings, tracer, cache):
    mock_client = AsyncMock()
    mock_chunk = _make_chunk()

    with (
        patch(
            "src.generation.query_handler.classify_intent",
            return_value={"tool": "vector_search", "complexity": "simple"},
        ),
        patch(
            "src.generation.query_handler.vector_search_pipeline",
            return_value=([mock_chunk], {}),
        ),
        patch(
            "src.generation.query_handler.generate_local",
            return_value=_wire("The answer is 5 staff."),
        ),
        patch(_ALIGN, return_value=_ALIGNED),
        patch(_GATE, return_value=_PASSED),
    ):
        result = await handle_query(
            "What is minimum staffing?", mock_client, settings, tracer, cache
        )

    assert "answer" in result
    assert "trace_id" in result
    assert result["tool"] == "vector_search"
    assert len(result["citations"]) >= 1
    assert result["citations"][0]["doc_type"] == "compliance"


@pytest.mark.asyncio
async def test_bare_yes_no_draft_is_retried_once_then_kept_if_complete(settings, tracer, cache):
    mock_client = AsyncMock()
    mock_chunk = _make_chunk()

    with (
        patch(
            "src.generation.query_handler.classify_intent",
            return_value={"tool": "vector_search", "complexity": "simple"},
        ),
        patch(
            "src.generation.query_handler.vector_search_pipeline",
            return_value=([mock_chunk], {}),
        ),
        patch(
            "src.generation.query_handler.generate_local",
            side_effect=[
                _wire("No."),
                _wire("No — minimum staffing is 5, and only 3 were on shift."),
            ],
        ) as local_answer,
        patch(_ALIGN, return_value=_ALIGNED),
        patch(_GATE, return_value=_PASSED),
    ):
        result = await handle_query(
            "Did they meet minimum staffing?", mock_client, settings, tracer, cache
        )

    assert local_answer.call_count == 2
    assert result["answer"] == "No — minimum staffing is 5, and only 3 were on shift."
    assert result["latency_components"]["final_generation_retried_bare_yes_no"] is True
    assert len(result["citations"]) >= 1


@pytest.mark.asyncio
async def test_bare_yes_no_draft_still_bare_after_retry_abstains(settings, tracer, cache):
    mock_client = AsyncMock()
    mock_chunk = _make_chunk()

    with (
        patch(
            "src.generation.query_handler.classify_intent",
            return_value={"tool": "vector_search", "complexity": "simple"},
        ),
        patch(
            "src.generation.query_handler.vector_search_pipeline",
            return_value=([mock_chunk], {}),
        ),
        patch(
            "src.generation.query_handler.generate_local",
            side_effect=[_wire("No."), _wire("Yes.")],
        ) as local_answer,
        patch(_ALIGN, return_value=_ALIGNED),
        patch(_GATE, return_value=_PASSED),
    ):
        result = await handle_query(
            "Did they meet minimum staffing?", mock_client, settings, tracer, cache
        )

    assert local_answer.call_count == 2
    assert result["answer"] == "insufficient context"
    assert result["citations"] == []


@pytest.mark.asyncio
async def test_tabular_query_path(settings, tracer, cache):
    mock_client = AsyncMock()
    sql_result = {
        "rows": [{"venue": "RSL", "staff_count": 10}],
        "sql_used": "SELECT venue, staff_count FROM v_shift_summary",
        "view_names": ["v_shift_summary"],
        "trace_id": "t1",
    }

    with (
        patch(
            "src.generation.query_handler.classify_intent",
            return_value={"tool": "text_to_sql", "complexity": "simple"},
        ),
        patch("src.generation.query_handler.embed_query", return_value=[0.1] * 384),
        patch(
            "src.generation.query_handler.find_relevant_views",
            return_value=[{
                "view_name": "v_shift_summary",
                "columns": ["venue", "staff_count"],
                "description": "x",
            }],
        ),
        patch("src.generation.query_handler.text_to_sql_tool", return_value=sql_result),
        patch(
            "src.generation.query_handler.generate_local",
            return_value=_wire("10 staff worked."),
        ),
        patch(_ALIGN, return_value=_ALIGNED),
        patch(_GATE, return_value=_PASSED),
    ):
        result = await handle_query("How many staff?", mock_client, settings, tracer, cache)

    assert result["tool"] == "text_to_sql"
    assert any("sql_used" in c for c in result["citations"])
    assert result["answer"] == "10 staff worked."
    assert result["latency_components"]["direct_sql_response"] is False


@pytest.mark.asyncio
async def test_simple_scalar_sql_skips_answer_llm_but_is_judged(settings, tracer, cache):
    sql_result = {
        "rows": [{"staff_count": 17}],
        "sql_used": "SELECT staff_count FROM v_peak_staffing WHERE shift_id = 'SH-010'",
        "view_names": ["v_peak_staffing"],
        "latency": {"candidate_llm_calls": [{"latency_ms": 4.0, "timed_out": False}]},
    }
    with (
        patch("src.generation.query_handler.classify_intent", return_value={
            "tool": "text_to_sql", "complexity": "simple",
        }),
        patch("src.generation.query_handler.embed_query", return_value=[0.1] * 384),
        patch("src.generation.query_handler.find_relevant_views", return_value=[]),
        patch("src.generation.query_handler.text_to_sql_tool", return_value=sql_result) as sql_tool,
        patch("src.generation.query_handler.generate_local") as answer_llm,
        patch(_ALIGN, return_value=_ALIGNED),
        patch(_GATE, return_value=_PASSED) as gate,
    ):
        result = await handle_query(
            "What was the staff count?", AsyncMock(), settings, tracer, cache
        )
    assert result["answer"] == "Staff count: 17."
    assert result["latency_components"]["direct_sql_response"] is True
    assert result["latency_components"]["final_generation_ms"] == 0
    assert result["latency_components"]["sql"] == sql_result["latency"]
    assert sql_tool.call_args.kwargs["simple_sql"] is True
    assert result["citations"][0]["sql_used"] == sql_result["sql_used"]
    answer_llm.assert_not_called()
    assert "staff_count" in gate.call_args.kwargs["evidence"] and result["faithfulness_gate"] == {
        "outcome": "passed"}


@pytest.mark.asyncio
@pytest.mark.parametrize("case_id,status", [
    (18, "partial_schema_support"),
    (23, "unsupported_by_schema"),
])
async def test_schema_guard_returns_no_sql_rows_or_citations(
    settings, tracer, cache, case_id: int, status: str
):
    with (
        patch("src.generation.query_handler.classify_intent", return_value={
            "tool": "hybrid", "complexity": "complex",
        }),
        patch("src.generation.query_handler.vector_search_pipeline", return_value=(
            [_make_chunk()], {"retrieval_ms": 7.0, "reranking_ms": 3.0},
        )),
        patch("src.generation.query_handler.embed_query", return_value=[0.1] * 384),
        patch("src.generation.query_handler.find_relevant_views", return_value=[]),
        patch("src.retrieval.text_to_sql.generate_sql", new_callable=AsyncMock) as generate,
        patch("src.retrieval.text_to_sql.validate_sql", new_callable=AsyncMock) as validate,
        patch("src.retrieval.text_to_sql.execute_sql", new_callable=AsyncMock) as execute,
        patch("src.generation.query_handler.generate_local") as local_answer,
        patch("src.generation.query_handler.generate_cloud") as cloud_answer,
    ):
        result = await handle_query(
            _SQL_GOLDEN[case_id]["question"], AsyncMock(), settings, tracer, cache
        )
    assert result["schema_status"] == status
    assert result["rows"] == []
    assert result["citations"] == []
    assert result["missing_schema"]
    assert result["answer"].startswith("insufficient context:")
    assert result["latency_components"]["retrieval_ms"] == 7.0
    assert result["latency_components"]["reranking_ms"] == 3.0
    assert "unattributed_ms" in result["latency_components"]
    generate.assert_not_awaited()
    validate.assert_not_awaited()
    execute.assert_not_awaited()
    local_answer.assert_not_called()
    cloud_answer.assert_not_called()


def test_curated_schema_exposes_supervisor_and_shift_incident_count() -> None:
    assert "supervisor" in CURATED_VIEW_COLUMNS["v_shift_details"]
    assert {"shift_id", "incident_count"} <= CURATED_VIEW_COLUMNS["v_incident_by_shift"]
    indexed = {d["view_name"]: set(d["columns"]) for d in _SCHEMA_DESCRIPTIONS}
    assert "supervisor" in indexed["v_shift_details"]
    assert {"shift_id", "incident_count"} <= indexed["v_incident_by_shift"]


@pytest.mark.asyncio
@pytest.mark.parametrize("case_id", [10, 11, 20])
async def test_supported_schema_cases_reach_sql_execution(settings, tracer, cache, case_id: int):
    rows = [{"shift_id": "SH-003", "supervisor": "A. Lee", "incident_count": 2}]
    with (
        patch("src.generation.query_handler.classify_intent", return_value={
            "tool": "hybrid" if case_id == 20 else "text_to_sql", "complexity": "complex",
        }),
        patch("src.generation.query_handler.vector_search_pipeline", return_value=([], {})),
        patch("src.generation.query_handler.embed_query", return_value=[0.1] * 384),
        patch("src.generation.query_handler.find_relevant_views", return_value=[]),
        patch("src.retrieval.text_to_sql.generate_sql", new_callable=AsyncMock,
              return_value=["SELECT shift_id, supervisor FROM v_shift_details"]) as generate,
        patch("src.retrieval.text_to_sql.validate_sql", new_callable=AsyncMock,
              return_value=(True, "")) as validate,
        patch("src.retrieval.text_to_sql.execute_sql", new_callable=AsyncMock,
              return_value=rows) as execute,
        patch("src.generation.query_handler.generate_local", return_value=_wire("A. Lee; 2.")),
        patch("src.generation.query_handler.generate_cloud", return_value=_wire("A. Lee; 2.")),
        patch(_ALIGN, return_value=_ALIGNED),
        patch(_GATE, return_value=_PASSED),
    ):
        result = await handle_query(
            _SQL_GOLDEN[case_id]["question"], AsyncMock(), settings, tracer, cache
        )
    assert "schema_status" not in result
    generate.assert_awaited()
    validate.assert_awaited()
    execute.assert_awaited()
    assert result["citations"][-1]["row_count"] == 1


@pytest.mark.asyncio
async def test_query_api_exposes_structured_schema_guard_result() -> None:
    request = MagicMock()
    guarded = {
        "answer": "insufficient context: No curated view exposes supervisor.",
        "citations": [], "rows": [], "schema_status": "unsupported_by_schema",
        "missing_schema": [{
            "kind": "field", "name": "supervisor",
            "detail": "No curated view exposes supervisor.",
        }],
        "trace_id": "trace", "route": "simple", "tool": "text_to_sql", "latency_ms": 1.0,
    }
    with patch("src.generation.query_handler.handle_query", return_value=guarded):
        response = await query_route(
            request, QueryRequest(query="Who supervised shift SH-003?"), MagicMock()
        )
    payload = response.model_dump(exclude_none=True)
    assert payload["schema_status"] == "unsupported_by_schema"
    assert payload["rows"] == []
    assert payload["missing_schema"][0]["name"] == "supervisor"
    assert payload["sql_citations"] == []
    assert payload["vector_citations"] == []


@pytest.mark.asyncio
async def test_cache_hit_skips_retrieval(settings, tracer):
    # This is a handler control-flow test. Semantic similarity is covered by
    # test_cache; avoiding model construction keeps the bypass assertion local.
    cache = SemanticCache("", settings.embedding_model)
    cached_resp = {
        "answer": "cached answer",
        "citations": [],
        "trace_id": "cached-trace",
        "route": "simple",
        "tool": "vector_search",
        "latency_ms": 10.0,
    }
    mock_client = AsyncMock()

    with (
        patch.object(cache, "get", return_value=cached_resp),
        patch("src.generation.query_handler.classify_intent") as route,
        patch("src.generation.query_handler.vector_search_pipeline") as retrieval,
    ):
        result = await handle_query("same query again", mock_client, settings, tracer, cache)

    assert result["answer"] == "cached answer"
    assert result["trace_id"] == "cached-trace"
    route.assert_not_called()
    retrieval.assert_not_called()


@pytest.mark.asyncio
async def test_empty_results_returns_insufficient_context(settings, tracer, cache):
    mock_client = AsyncMock()

    with (
        patch(
            "src.generation.query_handler.classify_intent",
            return_value={"tool": "vector_search", "complexity": "simple"},
        ),
        patch("src.generation.query_handler.vector_search_pipeline", return_value=([], {})),
    ):
        result = await handle_query("unknowable question", mock_client, settings, tracer, cache)

    assert result["answer"] == "insufficient context"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [
        "insufficient context",
        "I cannot find any information about the price of a schooner of beer at "
        "Fairfield RSL in the provided context.",
        "I'm unable to provide the price of a schooner of beer at Fairfield RSL as the "
        "information is not included in the provided context.",
        "The price of a schooner of beer at Fairfield RSL is not provided in the given context.",
        "No information about a dress code for entry to Fairfield RSL is "
        "provided in the given context.",
        "I cannot find information about the cost to book a function room at "
        "Fairfield RSL in the provided context.",
        "The provided context does not contain information about when Fairfield RSL was founded.",
        "The provided context does not include information about when Fairfield RSL was founded.",
        "No information provided in the given context about a dress code for entry to "
        "Fairfield RSL.",
        "No information provided about a dress code for entry to Fairfield RSL in the "
        "given context.",
        "Based on the provided context, there is no information about the cost to book "
        "a function room at Fairfield RSL.",
        "I'm unable to provide the cost to book a function room at Fairfield RSL as the "
        "information is not present in the given context.",
        "The context provided does not contain information about a member loyalty "
        "program at Fairfield RSL.",
        "The provided context does not contain any information about a member loyalty "
        "program at Fairfield RSL.",
        "The context provided does not include any information about a member loyalty "
        "program at Fairfield RSL. Therefore, I cannot provide an answer to this "
        "question based on the given context.",
    ],
)
async def test_generated_abstention_clears_citations(settings, tracer, cache, answer):
    """An explicit is_abstention=true must never present unrelated evidence, and
    the prose in the ignored `answer` field must never leak into the response —
    only the canonical 'insufficient context' wording is ever shown (D-006: no
    free-text regex classifies new generation output)."""
    mock_client = AsyncMock()

    with (
        patch(
            "src.generation.query_handler.classify_intent",
            return_value={"tool": "vector_search", "complexity": "simple"},
        ),
        patch(
            "src.generation.query_handler.vector_search_pipeline",
            return_value=([_make_chunk()], {}),
        ),
        patch(
            "src.generation.query_handler.generate_local",
            return_value=_wire(answer, is_abstention=True),
        ),
        patch(_ALIGN, return_value=_ALIGNED),
        patch(_GATE, return_value=_PASSED),
    ):
        result = await handle_query("unsupported question", mock_client, settings, tracer, cache)

    assert result["answer"] == "insufficient context"
    assert result["citations"] == []
    assert result["faithfulness_gate"] == {"outcome": "skipped_generated_abstention"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [
        "No information about a dress code for entry to Fairfield RSL is "
        "provided in the given context.",
        "I cannot find information about the cost to book a function room at "
        "Fairfield RSL in the provided context.",
    ],
)
async def test_cached_live_abstention_clears_stale_citations(settings, tracer, cache, answer):
    cached = {
        "answer": answer,
        "citations": [{"doc_type": "procedure"}],
        "trace_id": "cached-trace",
        "route": "simple",
        "tool": "vector_search",
        "latency_ms": 1.0,
    }
    with patch.object(cache, "get", return_value=cached):
        result = await handle_query("member loyalty", AsyncMock(), settings, tracer, cache)
    assert result["citations"] == []


@pytest.mark.asyncio
async def test_cached_supported_answer_keeps_citations(settings, tracer, cache):
    cached = {
        "answer": "The procedure requires two signatures.",
        "citations": [{"doc_type": "procedure"}],
        "trace_id": "cached-trace",
        "route": "simple",
        "tool": "vector_search",
        "latency_ms": 1.0,
    }
    with patch.object(cache, "get", return_value=cached):
        result = await handle_query("signature requirement", AsyncMock(), settings, tracer, cache)
    assert result["citations"] == cached["citations"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["input", "output"])
async def test_pii_span_logged(settings, cache, stage):
    """Input- and output-path redaction must each emit a pii_redaction Langfuse span."""
    mock_tracer = MagicMock()
    mock_tracer.start_trace.return_value = ("trace-pii", MagicMock())

    with (
        patch(
            "src.generation.query_handler.classify_intent",
            return_value={"tool": "vector_search", "complexity": "simple"},
        ),
        patch(
            "src.generation.query_handler.vector_search_pipeline",
            return_value=([_make_chunk()], {}),
        ),
        patch("src.generation.query_handler.generate_local", return_value=_wire("answer text")),
        patch(_ALIGN, return_value=_ALIGNED),
        patch(_GATE, return_value=_PASSED),
    ):
        await handle_query("hello world", AsyncMock(), settings, mock_tracer, cache)

    spans = [
        c for c in mock_tracer.log_span.call_args_list
        if c.kwargs.get("name") == "pii_redaction"
        and c.kwargs.get("metadata", {}).get("stage") == stage
    ]
    assert spans, f"expected a pii_redaction span with stage={stage}"


# PLAN-009: schema-capability fallback boundary and evidence-aware redaction.
def _sql_route_patches(sql_result: dict):
    return (
        patch("src.generation.query_handler.classify_intent", return_value={
            "tool": "text_to_sql", "complexity": "simple",
        }),
        patch("src.generation.query_handler.embed_query", return_value=[0.1] * 384),
        patch("src.generation.query_handler.find_relevant_views", return_value=[]),
        patch("src.generation.query_handler.text_to_sql_tool", return_value=sql_result),
    )


@pytest.mark.asyncio
async def test_schema_capability_gap_falls_back_to_document_retrieval(settings, tracer, cache):
    sql_result = {"rows": [], "error": "no valid candidates",
                  "fallback_reason": "schema_capability_gap"}
    chunk = _make_chunk("Total patron count (peak): 312", doc_type="shift_log")
    route, embed, views, sql_tool = _sql_route_patches(sql_result)
    with (
        route, embed, views, sql_tool,
        patch("src.generation.query_handler.vector_search_pipeline",
              return_value=([chunk], {})) as retrieval,
        patch("src.generation.query_handler.generate_local", return_value=_wire("312 patrons.")),
        patch(_ALIGN, return_value=_ALIGNED),
        patch(_GATE, return_value=_PASSED),
    ):
        result = await handle_query(
            "What was the peak patron count during shift SH-001?", AsyncMock(),
            settings, tracer, cache,
        )
    retrieval.assert_awaited_once()
    assert result["tool"] == "text_to_sql"
    assert result["answer"] == "312 patrons."
    assert result["citations"][0]["snippet"] == "Total patron count (peak): 312"
    assert result["latency_components"]["sql_schema_fallback"] is True


@pytest.mark.asyncio
async def test_valid_sql_with_zero_rows_does_not_fall_back(settings, tracer, cache):
    sql_result = {"rows": [], "sql_used": "SELECT supervisor FROM v_shift_details "
                  "WHERE shift_id = 'SH-999'", "view_names": ["v_shift_details"]}
    route, embed, views, sql_tool = _sql_route_patches(sql_result)
    with (
        route, embed, views, sql_tool,
        patch("src.generation.query_handler.vector_search_pipeline") as retrieval,
    ):
        result = await handle_query(
            "Who supervised shift SH-999?", AsyncMock(), settings, tracer, cache
        )
    retrieval.assert_not_called()
    assert result["answer"] == "insufficient context"
    assert result["latency_components"]["sql_schema_fallback"] is False


def _name_flagging_engines(names: tuple[str, ...]):
    """Fake Presidio engines that flag every occurrence of `names` as PERSON."""
    def analyze(text, **_):
        results = []
        for name in names:
            start = text.find(name)
            while start != -1:
                results.append(MagicMock(start=start, end=start + len(name), entity_type="PERSON"))
                start = text.find(name, start + 1)
        return results

    def anonymize(text, analyzer_results):
        for r in sorted(analyzer_results, key=lambda r: r.start, reverse=True):
            text = text[:r.start] + "<PERSON>" + text[r.end:]
        return MagicMock(text=text)

    return MagicMock(analyze=analyze), MagicMock(anonymize=anonymize)


@pytest.mark.asyncio
async def test_supervisor_name_from_sql_row_is_not_redacted(settings, tracer, cache):
    sql_result = {"rows": [{"supervisor": "J. Thompson"}],
                  "sql_used": "SELECT supervisor FROM v_shift_details WHERE shift_id = 'SH-003'",
                  "view_names": ["v_shift_details"]}
    route, embed, views, sql_tool = _sql_route_patches(sql_result)
    with (
        route, embed, views, sql_tool,
        patch("src.guardrails.pii_redactor._get_engines",
              return_value=_name_flagging_engines(("J. Thompson",))),
        patch(_ALIGN, return_value=_ALIGNED),
        patch(_GATE, return_value=_PASSED),
    ):
        result = await handle_query(
            "Who supervised shift SH-003?", AsyncMock(), settings, tracer, cache
        )
    assert result["answer"] == "Supervisor: J. Thompson."
