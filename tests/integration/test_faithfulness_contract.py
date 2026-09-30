"""Handler-level faithfulness contract (PLAN-003 section 3): only `passed` delivers."""
from __future__ import annotations

import asyncio
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import json

from src.api.routes.query import QueryRequest
from src.api.routes.query import query as query_route
from src.config import Settings
from src.generation.alignment_gate import AlignmentDecision
from src.generation.faithfulness_gate import GATE_ABSTENTION, GateDecision
from src.generation.query_handler import handle_query
from src.observability.tracer import RAGTracer
from src.retrieval.cache import SemanticCache

_H = "src.generation.query_handler"
_CHUNK = {"content": "Two signatures are required.", "doc_type": "procedure", "score": 0.9}
_ROWS = [{"shift_id": "SH-006", "incident_count": 2}]
_SQL = {"rows": _ROWS, "sql_used": "SELECT 1", "view_names": ["v_incident_by_shift"], "latency": {}}
_ANSWER = "Two signatures are required on shift SH-006."


def _settings(api_key: str = "test-key") -> Settings:
    return Settings(_env_file=None, openrouter_api_key=api_key, semantic_cache_url="")


def _wire(answer: str, is_abstention: bool = False) -> str:
    """Build a raw generator reply matching the GenerationResult JSON contract."""
    return json.dumps({"is_abstention": is_abstention, "answer": answer})


async def _run(
    tool="vector_search", gate=None, chat=None, api_key="test-key", answer=_ANSWER,
    is_abstention=False,
):
    """Run handle_query with mocked routing/retrieval/SQL; return (result, cache.set mock)."""
    cache = SemanticCache("", "all-MiniLM-L6-v2")
    chunks = [] if tool == "text_to_sql" else [_CHUNK]
    wired = _wire(answer, is_abstention=is_abstention)
    with ExitStack() as stack:
        enter = stack.enter_context
        enter(patch(f"{_H}.classify_intent", return_value={"tool": tool, "complexity": "complex"}))
        enter(patch(f"{_H}.vector_search_pipeline", return_value=(chunks, {})))
        enter(patch(f"{_H}.embed_query", return_value=[0.1] * 384))
        enter(patch(f"{_H}.find_relevant_views", return_value=[]))
        enter(patch(f"{_H}.text_to_sql_tool", return_value=_SQL))
        enter(patch(f"{_H}.generate_cloud", return_value=wired))
        enter(patch(f"{_H}.generate_local", return_value=wired))
        # Out of scope here: this file exercises faithfulness, not alignment.
        enter(patch(f"{_H}.check_alignment", AsyncMock(return_value=AlignmentDecision("passed"))))
        if gate is not None:
            enter(patch(f"{_H}.check_faithfulness", gate))
        if chat is not None:
            enter(patch("src.generation.faithfulness_gate.chat", chat))
        cache_set = enter(patch.object(cache, "set", AsyncMock()))
        tracer = RAGTracer("", "", "")
        result = await handle_query("q", AsyncMock(), _settings(api_key), tracer, cache)
    return result, cache_set


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ["vector_search", "text_to_sql", "hybrid"])
async def test_passed_delivers_answer_with_citations_and_caches(tool):
    result, cache_set = await _run(tool, chat=AsyncMock(return_value="yes"))
    assert result["answer"] == _ANSWER
    assert result["faithfulness_gate"] == {"outcome": "passed"}
    assert result["citations"]
    cached = cache_set.await_args.args[1]
    assert "faithfulness_gate" not in cached and cached["answer"] == _ANSWER


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ["vector_search", "text_to_sql", "hybrid"])
async def test_rejected_abstains_clears_vector_and_sql_citations_and_caches(tool):
    result, cache_set = await _run(tool, chat=AsyncMock(return_value="no"))
    assert result["answer"] == GATE_ABSTENTION
    assert result["citations"] == []
    assert result["faithfulness_gate"] == {"outcome": "rejected"}
    cache_set.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ["text_to_sql", "hybrid"])
async def test_sql_evidence_is_judged_not_skipped(tool):
    chat = AsyncMock(return_value="yes")
    await _run(tool, chat=chat)
    evidence = chat.await_args.kwargs["prompt"].split("\n\nAnswer:")[0]
    assert "shift_id | incident_count" in evidence and "SH-006 | 2" in evidence
    assert ("Two signatures" in evidence) == (tool == "hybrid")


async def _slow_judge(**_: object) -> GateDecision:
    await asyncio.sleep(1)
    return GateDecision("passed")


@pytest.mark.asyncio
@pytest.mark.parametrize("case,expected", [
    ({"chat": AsyncMock(side_effect=ConnectionError("x"))},
     {"outcome": "judge_error", "exception_class": "ConnectionError"}),
    ({"gate": _slow_judge}, {"outcome": "timeout", "exception_class": "TimeoutError"}),
    ({"api_key": "", "chat": AsyncMock()}, {"outcome": "skipped_no_api_key"}),
])
@pytest.mark.parametrize("tool", ["vector_search", "text_to_sql"])
async def test_judge_unavailable_fails_closed_and_is_never_cached(case, expected, tool):
    with patch(f"{_H}.FAITHFULNESS_TIMEOUT_S", 0.05):
        result, cache_set = await _run(tool, **case)
    assert result["answer"] == GATE_ABSTENTION
    assert result["citations"] == []
    assert result["faithfulness_gate"] == expected
    cache_set.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_evidence_skips_generator_and_judge():
    cache = SemanticCache("", "all-MiniLM-L6-v2")
    route = {"tool": "vector_search", "complexity": "simple"}
    with (
        patch(f"{_H}.classify_intent", return_value=route),
        patch(f"{_H}.vector_search_pipeline", return_value=([], {})),
        patch(f"{_H}.generate_local") as generator,
        patch(f"{_H}.check_faithfulness") as judge,
    ):
        result = await handle_query("q", AsyncMock(), _settings(), RAGTracer("", "", ""), cache)
    assert result["answer"] == "insufficient context" and result["citations"] == []
    assert result["faithfulness_gate"] == {"outcome": "skipped_no_context"}
    generator.assert_not_called()
    judge.assert_not_called()


@pytest.mark.asyncio
async def test_case_25_generated_abstention_has_no_citations():
    # The model explicitly flags is_abstention=true; the prose in `answer` is
    # ignored in favor of the canonical wording (D-006: no free-text regex
    # classifies new generation output).
    answer = "The provided context does not contain information about the current CEO."
    result, _ = await _run(
        "vector_search", chat=AsyncMock(return_value="yes"), answer=answer, is_abstention=True,
    )
    assert result["answer"] == "insufficient context"
    assert result["citations"] == []


@pytest.mark.asyncio
async def test_cache_hit_never_replays_an_outcome():
    cache = SemanticCache("", "all-MiniLM-L6-v2")
    cached = {"answer": _ANSWER, "citations": [], "trace_id": "t", "route": "simple",
              "tool": "vector_search", "latency_ms": 1.0}
    with patch.object(cache, "get", AsyncMock(return_value=cached)):
        result = await handle_query("q", AsyncMock(), _settings(), RAGTracer("", "", ""), cache)
    assert "faithfulness_gate" not in result


@pytest.mark.asyncio
async def test_public_schema_never_exposes_gate_diagnostics():
    internal = {
        "answer": GATE_ABSTENTION, "citations": [], "trace_id": "t", "route": "complex",
        "tool": "hybrid", "latency_ms": 1.0,
        "faithfulness_gate": {"outcome": "judge_error", "exception_class": "RuntimeError"},
    }
    with patch(f"{_H}.handle_query", AsyncMock(return_value=internal)):
        response = await query_route(MagicMock(), QueryRequest(query="CEO?"), MagicMock())
    payload = response.model_dump_json()
    assert not any(s in payload for s in ("faithfulness_gate", "judge_error", "RuntimeError"))
