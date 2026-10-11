"""Unit tests for src/generation/alignment_gate.py."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.generation.alignment_gate import AlignmentDecision, check_alignment

_MODEL = "mistralai/mistral-nemo"
_BASE_URL = "https://openrouter.ai/api/v1"


def _sql(rows, sql_used="SELECT * FROM v_peak_staffing"):
    return {"rows": rows, "sql_used": sql_used}


async def _check(**kwargs):
    defaults = dict(
        query="Did Fairfield RSL meet its security staffing requirement on 2026-05-25?",
        answer="Yes, Fairfield RSL met its security staffing requirement on 2026-05-25.",
        chunks=[],
        sql_result=None,
        filters=None,
        model=_MODEL,
        api_key="test-key",
        base_url=_BASE_URL,
    )
    defaults.update(kwargs)
    return await check_alignment(**defaults)


@pytest.mark.asyncio
async def test_role_mismatch_rejected_before_judge_is_called():
    # The known slip-through: question names "security", SQL rows carry floor_staff.
    rows = [
        {
            "venue": "Fairfield RSL",
            "date": "2026-05-25",
            "role": "floor_staff",
            "required_count": 3,
            "count": 5,
            "gap": 2,
        }
    ]
    with patch("src.generation.alignment_gate.chat", AsyncMock()) as chat:
        decision = await _check(sql_result=_sql(rows))
    assert decision == AlignmentDecision("rejected", "role_mismatch")
    chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_matching_security_role_defers_to_judge_and_passes():
    rows = [
        {
            "venue": "Fairfield RSL",
            "date": "2026-05-25",
            "role": "security",
            "required_count": 3,
            "count": 2,
            "gap": -1,
        }
    ]
    with patch("src.generation.alignment_gate.chat", AsyncMock(return_value="yes")):
        decision = await _check(sql_result=_sql(rows))
    assert decision == AlignmentDecision("passed")


@pytest.mark.asyncio
async def test_date_mismatch_rejected_before_judge_is_called():
    rows = [{"venue": "Fairfield RSL", "date": "2026-05-24", "role": "security"}]
    with patch("src.generation.alignment_gate.chat", AsyncMock()) as chat:
        decision = await _check(sql_result=_sql(rows))
    assert decision == AlignmentDecision("rejected", "date_mismatch")
    chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_venue_filter_mismatch_rejected():
    rows = [{"venue": "Other Venue", "date": "2026-05-25", "role": "security"}]
    with patch("src.generation.alignment_gate.chat", AsyncMock()) as chat:
        decision = await _check(
            sql_result=_sql(rows), filters={"venue": "Fairfield RSL"}
        )
    assert decision == AlignmentDecision("rejected", "venue_mismatch")
    chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_role_column_with_provable_sql_predicate_defers_to_judge():
    rows = [{"venue": "Fairfield RSL", "date": "2026-05-25", "required_count": 3}]
    sql_used = "SELECT * FROM v_peak_staffing WHERE role = 'security'"
    with patch("src.generation.alignment_gate.chat", AsyncMock(return_value="yes")):
        decision = await _check(sql_result=_sql(rows, sql_used))
    assert decision == AlignmentDecision("passed")


@pytest.mark.asyncio
async def test_missing_role_column_without_provable_predicate_is_indeterminate_not_silent_pass():
    rows = [{"venue": "Fairfield RSL", "date": "2026-05-25", "required_count": 3}]
    sql_used = "SELECT * FROM v_peak_staffing"
    with patch("src.generation.alignment_gate.chat", AsyncMock()) as chat:
        decision = await _check(sql_result=_sql(rows, sql_used), api_key="")
    assert decision == AlignmentDecision("indeterminate", "unknown")
    chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_vector_only_no_explicit_role_date_defers_to_semantic_judge():
    chunks = [{"content": "The evacuation procedure names two assembly points."}]
    with patch("src.generation.alignment_gate.chat", AsyncMock(return_value="yes")):
        decision = await _check(
            query="What does the evacuation procedure say about assembly points?",
            answer="Two assembly points are named.",
            chunks=chunks,
        )
    assert decision == AlignmentDecision("passed")


@pytest.mark.asyncio
async def test_semantic_judge_no_is_rejected():
    chunks = [{"content": "Unrelated leave policy text."}]
    with patch("src.generation.alignment_gate.chat", AsyncMock(return_value="no")):
        decision = await _check(
            query="What does the evacuation procedure say about assembly points?",
            answer="Two assembly points are named.",
            chunks=chunks,
        )
    assert decision == AlignmentDecision("rejected", "semantic_mismatch")


@pytest.mark.asyncio
async def test_malformed_judge_reply_is_indeterminate():
    chunks = [{"content": "Some evidence."}]
    with patch("src.generation.alignment_gate.chat", AsyncMock(return_value="maybe")):
        decision = await _check(
            query="What does the evacuation procedure say?",
            answer="Some answer.",
            chunks=chunks,
        )
    assert decision == AlignmentDecision("indeterminate", "unknown")


@pytest.mark.asyncio
async def test_judge_error_fails_closed():
    chunks = [{"content": "Some evidence."}]
    with patch(
        "src.generation.alignment_gate.chat", AsyncMock(side_effect=RuntimeError("boom"))
    ):
        decision = await _check(
            query="What does the evacuation procedure say?",
            answer="Some answer.",
            chunks=chunks,
        )
    assert decision == AlignmentDecision("judge_error", "unknown")


@pytest.mark.asyncio
async def test_no_api_key_is_indeterminate_without_calling_judge():
    chunks = [{"content": "Some evidence."}]
    with patch("src.generation.alignment_gate.chat", AsyncMock()) as chat:
        decision = await _check(
            query="What does the evacuation procedure say?",
            answer="Some answer.",
            chunks=chunks,
            api_key="",
        )
    assert decision == AlignmentDecision("indeterminate", "unknown")
    chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_evidence_at_all_is_indeterminate():
    with patch("src.generation.alignment_gate.chat", AsyncMock()) as chat:
        decision = await _check(
            query="What does the evacuation procedure say?", answer="Some answer.", chunks=[]
        )
    assert decision == AlignmentDecision("indeterminate", "unknown")
    chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_multiple_roles_in_rows_one_matching_one_not_is_rejected():
    # Caller filters must apply to every answer-bearing row, not just some.
    rows = [
        {"venue": "Fairfield RSL", "date": "2026-05-25", "role": "security"},
        {"venue": "Fairfield RSL", "date": "2026-05-25", "role": "floor_staff"},
    ]
    with patch("src.generation.alignment_gate.chat", AsyncMock()) as chat:
        decision = await _check(sql_result=_sql(rows))
    assert decision == AlignmentDecision("rejected", "role_mismatch")
    chat.assert_not_awaited()


def test_decision_as_dict_never_leaks_evidence_or_raw_judge_output():
    decision = AlignmentDecision("rejected", "semantic_mismatch")
    assert decision.as_dict() == {"outcome": "rejected", "reason": "semantic_mismatch"}
    assert set(decision.as_dict()) == {"outcome", "reason"}
