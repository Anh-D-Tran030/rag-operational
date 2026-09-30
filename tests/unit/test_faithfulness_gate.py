"""Unit tests for the faithfulness gate."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.generation.faithfulness_gate import (
    GateDecision,
    check_faithfulness,
    record_gate_decision,
    render_evidence,
)
from src.observability.metrics import rag_faithfulness_gate_total

_MODEL = "mistralai/mistral-nemo"
_BASE_URL = "https://openrouter.ai/api/v1"
_OUTCOMES = (
    "passed", "rejected", "skipped_no_api_key", "skipped_no_context", "judge_error", "timeout",
)


async def _judge(evidence: str = "The venue opened at 9pm.", api_key: str = "test-key"):
    return await check_faithfulness(
        answer="The venue opened at 9pm.",
        evidence=evidence,
        model=_MODEL,
        api_key=api_key,
        base_url=_BASE_URL,
    )


@pytest.mark.asyncio
async def test_no_key_fails_closed_without_calling_judge():
    with patch("src.generation.faithfulness_gate.chat", AsyncMock()) as chat:
        decision = await _judge(api_key="")
    assert decision == GateDecision("skipped_no_api_key")
    assert not decision.passed
    chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_evidence_is_skipped_no_context():
    with patch("src.generation.faithfulness_gate.chat", AsyncMock()) as chat:
        decision = await _judge(evidence="  ")
    assert decision.outcome == "skipped_no_context"
    chat.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", ["yes", "Yes.", "**yes**"])
async def test_gate_passes_on_yes(raw):
    with patch("src.generation.faithfulness_gate.chat", AsyncMock(return_value=raw)):
        decision = await _judge()
    assert decision == GateDecision("passed")
    assert decision.passed


@pytest.mark.asyncio
async def test_gate_rejects_on_no_and_keeps_failure_counter():
    with (
        patch("src.generation.faithfulness_gate.chat", AsyncMock(return_value="no")),
        patch("src.generation.faithfulness_gate.rag_faithfulness_gate_failures") as failures,
    ):
        decision = await _judge()
    assert decision == GateDecision("rejected")
    failures.inc.assert_called_once()


@pytest.mark.asyncio
async def test_judge_error_fails_closed_with_class_name_only():
    error = RuntimeError("upstream said: sk-secret-token body")
    with patch("src.generation.faithfulness_gate.chat", AsyncMock(side_effect=error)):
        decision = await _judge()
    assert decision == GateDecision("judge_error", "RuntimeError")
    assert decision.as_dict() == {"outcome": "judge_error", "exception_class": "RuntimeError"}
    assert "sk-secret" not in repr(decision)


def test_only_working_judge_outcomes_are_cacheable():
    cacheable = {o for o in _OUTCOMES if GateDecision(o).cacheable}
    assert cacheable == {"passed", "rejected", "skipped_no_context"}
    assert [o for o in _OUTCOMES if GateDecision(o).passed] == ["passed"]


def test_evidence_includes_every_chunk_and_exact_sql_rows():
    chunks = [{"content": f"chunk {i}"} for i in range(7)]
    sql = {"rows": [{"shift_id": "SH-006", "incident_count": 2}]}
    evidence = render_evidence(chunks, sql, token_budget=4000)
    assert all(f"chunk {i}" in evidence for i in range(7))  # no 5-chunk truncation
    assert "shift_id | incident_count" in evidence
    assert "SH-006 | 2" in evidence


def test_sql_only_evidence_is_not_empty():
    evidence = render_evidence([], {"rows": [{"staff_count": 17}]}, token_budget=4000)
    assert "staff_count" in evidence and "17" in evidence


def test_evidence_is_bounded_by_token_budget():
    evidence = render_evidence([{"content": "x" * 10_000}], None, token_budget=100)
    assert len(evidence) == 400


def test_record_counts_one_bounded_label_and_logs_trace_span():
    tracer = MagicMock()
    before = rag_faithfulness_gate_total.labels(outcome="timeout")._value.get()
    record_gate_decision(GateDecision("timeout", "TimeoutError"), tracer, "trace", 12.5)
    after = rag_faithfulness_gate_total.labels(outcome="timeout")._value.get()
    assert after == before + 1
    span = tracer.log_span.call_args.kwargs
    assert span["name"] == "faithfulness_gate"
    assert span["output"] == {"outcome": "timeout", "exception_class": "TimeoutError"}
    assert rag_faithfulness_gate_total._labelnames == ("outcome",)
