"""Unit tests for src/retrieval/text_to_sql.py."""
import asyncio
import json
import sqlite3
from unittest.mock import AsyncMock, patch

import httpx
import openai
import pytest

from src.evaluation.sql_ground_truth import load_golden_cases
from src.llm_client import ChatCompletionResult
from src.retrieval.text_to_sql import (
    CURATED_VIEW_COLUMNS,
    SQL_FALLBACK_MODEL,
    SQL_RESPONSE_FORMAT,
    assess_schema_coverage,
    execute_sql,
    generate_sql,
    text_to_sql_tool,
    validate_sql,
)

_DB = "sqlite+aiosqlite:///./operational.db"
_GOLDEN = load_golden_cases("eval_data/golden_dataset.json")


def _structured(sql: str) -> str:
    return json.dumps({"sql": sql})


@pytest.mark.parametrize("case_id", [9, 10, 11, 12, 13, 16, 17, 19, 20, 21])
def test_supported_golden_cases_have_full_schema_coverage(case_id: int) -> None:
    assert assess_schema_coverage(_GOLDEN[case_id]["question"]) == {
        "status": "full", "missing_schema": [],
    }


@pytest.mark.parametrize("case_id,status,missing_name", [
    (18, "partial_schema_support", "complete_incident_to_staffing_shift_evidence"),
    (23, "unsupported_by_schema", "intervention_rate_denominator"),
])
def test_blocked_golden_cases_name_the_missing_schema(
    case_id: int, status: str, missing_name: str
) -> None:
    coverage = assess_schema_coverage(_GOLDEN[case_id]["question"])
    assert coverage["status"] == status
    assert missing_name in {item["name"] for item in coverage["missing_schema"]}
    assert all(item["detail"] for item in coverage["missing_schema"])


def test_schema_guard_columns_match_curated_view_metadata() -> None:
    from src.ingestion.schema_indexer import _SCHEMA_DESCRIPTIONS

    indexed = {view["view_name"]: frozenset(view["columns"]) for view in _SCHEMA_DESCRIPTIONS}
    assert CURATED_VIEW_COLUMNS == indexed


@pytest.mark.parametrize("query,status", [
    ("Who was the supervisor for shift SH-003?", "full"),
    ("How many incidents did shift SH-006 have?", "full"),
    ("Were high-severity incidents linked to low staffing?", "partial_schema_support"),
    ("Compare RSA intervention percentages across evening shifts.", "unsupported_by_schema"),
])
def test_schema_guard_covers_live_question_paraphrases(query: str, status: str) -> None:
    assert assess_schema_coverage(query)["status"] == status


@pytest.mark.asyncio
@pytest.mark.parametrize("case_id", [18, 23])
async def test_direct_sql_generator_rejects_incomplete_schema(case_id: int) -> None:
    with patch("src.retrieval.text_to_sql.chat", new_callable=AsyncMock) as chat:
        candidates = await generate_sql(
            _GOLDEN[case_id]["question"], [], "model", "fake", "https://example.test"
        )
    assert candidates == []
    chat.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("case_id", [9, 10, 11, 12, 13, 16, 17, 19, 20, 21])
async def test_supported_cases_reach_sql_execution(case_id: int) -> None:
    from src.config import Settings

    settings = Settings(_env_file=None, openrouter_api_key="fake", db_url=_DB)
    with (
        patch("src.retrieval.text_to_sql.generate_sql", new_callable=AsyncMock) as generate,
        patch("src.retrieval.text_to_sql.validate_sql", new_callable=AsyncMock) as validate,
        patch("src.retrieval.text_to_sql.execute_sql", new_callable=AsyncMock) as execute,
    ):
        generate.return_value = ["SELECT 1 AS value"]
        validate.return_value = (True, "")
        execute.return_value = [{"value": 1}]
        result = await text_to_sql_tool(
            _GOLDEN[case_id]["question"], [], settings, "trace"
        )
    assert result["rows"] == [{"value": 1}]
    assert "schema_status" not in result
    generate.assert_awaited_once()
    validate.assert_awaited_once()
    execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_schema_guard_abstains_before_generation_validation_or_execution() -> None:
    from src.config import Settings

    settings = Settings(_env_file=None, openrouter_api_key="fake", db_url=_DB)
    with (
        patch("src.retrieval.text_to_sql.generate_sql", new_callable=AsyncMock) as generate,
        patch("src.retrieval.text_to_sql.validate_sql", new_callable=AsyncMock) as validate,
        patch("src.retrieval.text_to_sql.execute_sql", new_callable=AsyncMock) as execute,
        patch("src.retrieval.text_to_sql.rag_sql_unsupported_by_schema_total") as unsupported,
        patch("src.retrieval.text_to_sql.rag_sql_partial_schema_support_total") as partial,
        patch("src.retrieval.text_to_sql.rag_sql_schema_guard_abstention_total") as abstention,
    ):
        results = {
            case_id: await text_to_sql_tool(
                _GOLDEN[case_id]["question"], [], settings, "trace"
            ) for case_id in (18, 23)
        }
    assert all(result["rows"] == [] and result["sql_used"] is None for result in results.values())
    assert results[18]["schema_status"] == "partial_schema_support"
    assert results[23]["schema_status"] == "unsupported_by_schema"
    generate.assert_not_awaited()
    validate.assert_not_awaited()
    execute.assert_not_awaited()
    assert unsupported.inc.call_count == 1
    partial.inc.assert_called_once()
    assert abstention.inc.call_count == 2


@pytest.mark.asyncio
async def test_generate_sql_returns_list():
    raw = _structured("SELECT venue FROM v_shift_summary")
    with patch("src.retrieval.text_to_sql.chat", AsyncMock(return_value=raw)) as chat:
        result = await generate_sql(
            query="How many venues?",
            relevant_views=[{
                "view_name": "v_shift_summary",
                "columns": ["venue"],
                "description": "shift data",
            }],
            model="openai/gpt-oss-120b",
            api_key="fake-key",
            base_url="https://openrouter.ai/api/v1",
        )
    assert isinstance(result, list)
    assert len(result) >= 1
    assert all(isinstance(s, str) for s in result)
    assert chat.await_args.kwargs["response_format"] == SQL_RESPONSE_FORMAT


@pytest.mark.asyncio
async def test_generate_sql_normalizes_hyphenated_incident_type_literal():
    raw = (
        "SELECT supervisor FROM v_incident_severity "
        "WHERE incident_type = 'medical-event'"
    )
    with patch("src.retrieval.text_to_sql.chat", AsyncMock(return_value=_structured(raw))):
        result = await generate_sql(
            query="Who supervised the medical-event incident?",
            relevant_views=[],
            model="test-model",
            api_key="fake-key",
            base_url="https://example.test",
        )

    assert result == [
        "SELECT supervisor FROM v_incident_severity "
        "WHERE incident_type = 'medical event' COLLATE NOCASE"
    ]


@pytest.mark.asyncio
async def test_supervisor_comparison_uses_deterministic_auditable_sql():
    query = (
        "Was the supervisor on the day of the medical-event incident the same person who "
        "supervised shift SH-001?"
    )
    with patch("src.retrieval.text_to_sql.chat", AsyncMock()) as chat:
        result = await generate_sql(
            query=query,
            relevant_views=[],
            model="test-model",
            api_key="fake-key",
            base_url="https://example.test",
        )

    assert len(result) == 1
    assert "AS incident_supervisor" in result[0]
    assert "AS comparison_supervisor" in result[0]
    assert "AS same_supervisor" in result[0]
    assert "incident.incident_type = 'medical event' COLLATE NOCASE" in result[0]
    chat.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "first", "second", "rejection"),
    [
        (
            "Who supervised shift SH-003?",
            "SELECT shift_id FROM v_peak_staffing WHERE shift_id = 'SH-003'",
            "SELECT supervisor FROM v_shift_details WHERE shift_id = 'SH-003'",
            "missing_requested_supervisor_field",
        ),
        (
            "How many incidents were logged on shift SH-006?",
            "SELECT COUNT(*) AS incident_count FROM v_incident_severity WHERE shift_id = 'SH-006'",
            "SELECT incident_count FROM v_incident_by_shift WHERE shift_id = 'SH-006'",
            "shift_incident_total_requires_v_incident_by_shift",
        ),
        (
            "Who supervised shift SH-001?",
            "SELECT i.shift_id FROM v_incident_severity i JOIN v_shift_summary s "
            "ON i.shift_id = s.shift_type",
            "SELECT i.supervisor FROM v_shift_details i WHERE i.shift_id = 'SH-001'",
            "incompatible_shift_id_shift_type_join",
        ),
    ],
)
async def test_semantically_invalid_sql_is_retried_before_execution(
    query: str, first: str, second: str, rejection: str,
):
    diagnostics: list[dict] = []
    with patch(
        "src.retrieval.text_to_sql.chat",
        AsyncMock(side_effect=[_structured(first), _structured(second)]),
    ) as chat:
        result = await generate_sql(
            query=query,
            relevant_views=[],
            model="test-model",
            api_key="fake-key",
            base_url="https://example.test",
            diagnostics=diagnostics,
        )

    assert result == [second]
    assert chat.await_count == 2
    assert rejection in diagnostics[0]["candidate_diagnostics"][0]["filter_rejections"]
    assert diagnostics[0]["will_retry"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw,diagnostic",
    [
        ("not-json", "malformed_structured_output"),
        (json.dumps({"sql": ""}), "incomplete_structured_output"),
        (json.dumps({"other": "SELECT 1"}), "incomplete_structured_output"),
        (json.dumps({"sql": "SELECT 1", "extra": True}), "incomplete_structured_output"),
    ],
)
async def test_invalid_sql_envelope_is_diagnostic_without_fallback(raw, diagnostic):
    diagnostics: list[dict] = []
    with patch("src.retrieval.text_to_sql.chat", AsyncMock(return_value=raw)) as chat:
        result = await generate_sql(
            query="How many venues?", relevant_views=[], model="test-model",
            api_key="fake", base_url="https://example.test", diagnostics=diagnostics,
        )

    assert result == []
    chat.assert_awaited_once()
    assert diagnostics[0]["outcome"] == diagnostic
    assert diagnostics[0]["structured_output_error"] == diagnostic


@pytest.mark.asyncio
async def test_unsupported_structured_output_invokes_fallback():
    error = openai.BadRequestError(
        "structured output unsupported",
        response=httpx.Response(400, request=httpx.Request("POST", "https://example.test")),
        body=None,
    )
    diagnostics: list[dict] = []
    with patch(
        "src.retrieval.text_to_sql.chat",
        AsyncMock(side_effect=[error, error]),
    ) as chat:
        result = await generate_sql(
            query="How many venues?", relevant_views=[], model="test-model",
            api_key="fake", base_url="https://example.test", diagnostics=diagnostics,
        )

    assert result == []
    assert chat.await_count == 2
    assert diagnostics[0]["outcome"] == "provider_error"
    assert diagnostics[1]["phase"] == "fallback"
    assert diagnostics[1]["fallback_trigger_reason"] == "provider_error"


@pytest.mark.asyncio
async def test_sql_candidate_call_records_timeout_and_retries(monkeypatch):
    async def slow_chat(**_kwargs):
        await asyncio.sleep(0.1)
        return "SELECT 1"

    monkeypatch.setattr("src.retrieval.text_to_sql.SQL_LLM_TIMEOUT_S", 0.001)
    timings = []
    with patch("src.retrieval.text_to_sql.chat", side_effect=slow_chat) as chat:
        result = await generate_sql(
            query="How many staff?", relevant_views=[], model="test",
            api_key="fake", base_url="https://example.test", call_timings=timings,
        )
    assert result == []
    assert chat.call_count == 2
    assert len(timings) == 2
    assert all(call["timed_out"] and call["latency_ms"] >= 0 for call in timings)


@pytest.mark.asyncio
async def test_empty_primary_retries_once_then_fallback_succeeds():
    valid = "SELECT venue FROM v_shift_summary"
    diagnostics: list[dict] = []
    status: dict = {}
    mock_chat = AsyncMock(side_effect=["", "", _structured(valid)])

    with patch("src.retrieval.text_to_sql.chat", mock_chat):
        result = await generate_sql(
            query="How many venues?", relevant_views=[], model="openai/gpt-oss-120b",
            api_key="fake", base_url="https://example.test", diagnostics=diagnostics,
            generation_status=status,
        )

    assert result == [valid]
    assert [call.kwargs["model"] for call in mock_chat.await_args_list] == [
        "openai/gpt-oss-120b", "openai/gpt-oss-120b", SQL_FALLBACK_MODEL,
    ]
    assert [item["phase"] for item in diagnostics] == ["primary", "primary", "fallback"]
    assert diagnostics[-1]["fallback_trigger_reason"] == "empty_completion"
    assert status == {"status": "available", "reason": None}


@pytest.mark.asyncio
async def test_valid_primary_sql_never_invokes_fallback():
    valid = "SELECT venue FROM v_shift_summary"
    mock_chat = AsyncMock(return_value=_structured(valid))

    with patch("src.retrieval.text_to_sql.chat", mock_chat):
        result = await generate_sql(
            query="How many venues?", relevant_views=[], model="openai/gpt-oss-120b",
            api_key="fake", base_url="https://example.test",
        )

    assert result == [valid]
    mock_chat.assert_awaited_once()
    assert mock_chat.await_args.kwargs["model"] == "openai/gpt-oss-120b"


@pytest.mark.asyncio
async def test_primary_timeout_invokes_fallback_once():
    valid = "SELECT venue FROM v_shift_summary"
    diagnostics: list[dict] = []
    mock_chat = AsyncMock(side_effect=[TimeoutError(), _structured(valid)])

    with patch("src.retrieval.text_to_sql.chat", mock_chat):
        result = await generate_sql(
            query="How many venues?", relevant_views=[], model="openai/gpt-oss-120b",
            api_key="fake", base_url="https://example.test", diagnostics=diagnostics,
        )

    assert result == [valid]
    assert [call.kwargs["model"] for call in mock_chat.await_args_list] == [
        "openai/gpt-oss-120b", SQL_FALLBACK_MODEL,
    ]
    assert diagnostics[0]["outcome"] == "timeout"
    assert diagnostics[0]["timeout_error_class"] == "TimeoutError"
    assert diagnostics[1]["fallback_trigger_reason"] == "timeout"


@pytest.mark.asyncio
async def test_primary_and_fallback_unavailable_returns_no_sql_with_reason():
    diagnostics: list[dict] = []
    status: dict = {}
    with patch("src.retrieval.text_to_sql.chat", AsyncMock(side_effect=["", "", ""])):
        result = await generate_sql(
            query="How many venues?", relevant_views=[], model="openai/gpt-oss-120b",
            api_key="fake", base_url="https://example.test", diagnostics=diagnostics,
            generation_status=status,
        )

    assert result == []
    assert status == {
        "status": "sql_generation_unavailable", "reason": "empty_completion",
    }
    assert diagnostics[-1]["phase"] == "fallback"
    assert diagnostics[-1]["outcome"] == "empty_completion"


@pytest.mark.asyncio
async def test_sql_tool_returns_explicit_unavailable_after_both_models_fail():
    from src.config import Settings

    with patch("src.retrieval.text_to_sql.chat", AsyncMock(side_effect=["", "", ""])):
        result = await text_to_sql_tool(
            "How many venues?", [],
            Settings(_env_file=None, openrouter_api_key="fake", db_url=_DB), "trace",
        )

    assert result["error"] == "sql_generation_unavailable"
    assert result["generation_unavailable_reason"] == "empty_completion"
    assert result["sql_used"] is None
    assert result["rows"] == []


@pytest.mark.asyncio
async def test_sql_tool_never_falls_back_after_execution_failure():
    from src.config import Settings

    with (
        patch(
            "src.retrieval.text_to_sql.chat",
            AsyncMock(side_effect=[_structured("SELECT 1 AS value"), ""]),
        ) as mock_chat,
        patch("src.retrieval.text_to_sql.validate_sql", AsyncMock(return_value=(True, ""))),
        patch(
            "src.retrieval.text_to_sql.execute_sql",
            AsyncMock(side_effect=RuntimeError("database unavailable")),
        ),
    ):
        result = await text_to_sql_tool(
            "How many venues?", [],
            Settings(_env_file=None, openrouter_api_key="fake", db_url=_DB), "trace",
            max_attempts=2, simple_sql=True,
        )

    assert result["error"] == "database unavailable"
    assert [call.kwargs["model"] for call in mock_chat.await_args_list] == [
        "openai/gpt-oss-120b",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [
    openai.APIConnectionError(
        message="connection failed", request=httpx.Request("POST", "https://example.test")
    ),
    openai.APIStatusError(
        "upstream unavailable",
        response=httpx.Response(503, request=httpx.Request("POST", "https://example.test")),
        body=None,
    ),
])
async def test_provider_errors_are_not_classified_as_empty_completion(error):
    diagnostics: list[dict] = []
    status: dict = {}
    with patch(
        "src.retrieval.text_to_sql.chat",
        AsyncMock(side_effect=[error, error]),
    ) as mock_chat:
        result = await generate_sql(
            query="How many venues?", relevant_views=[], model="openai/gpt-oss-120b",
            api_key="fake", base_url="https://example.test", diagnostics=diagnostics,
            generation_status=status,
        )

    assert result == []
    assert mock_chat.await_count == 2
    assert status == {"status": "sql_generation_unavailable", "reason": "provider_error"}
    assert diagnostics[0]["outcome"] == "provider_error"
    assert diagnostics[0]["provider_error"]["class"] != "empty_completion"
    assert diagnostics[1]["fallback_trigger_reason"] == "provider_error"


@pytest.mark.asyncio
async def test_length_finish_reason_is_confirmed_truncation_and_falls_back():
    truncated = ChatCompletionResult(
        text="SELECT venue FROM v_shift_summary WHERE venue = '",
        response_id="primary-response",
        finish_reason="length",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        model="openai/gpt-oss-120b",
        provider="provider-a",
        configured_output_token_limit=1024,
    )
    valid = "SELECT venue FROM v_shift_summary"
    diagnostics: list[dict] = []
    with patch(
        "src.retrieval.text_to_sql.chat",
        AsyncMock(side_effect=[truncated, ChatCompletionResult(
            text=_structured(valid),
            response_id="fallback-response",
            finish_reason="stop",
            usage={"prompt_tokens": 8, "completion_tokens": 6, "total_tokens": 14},
            model=SQL_FALLBACK_MODEL,
            provider="provider-b",
            configured_output_token_limit=1024,
        )]),
    ) as mock_chat:
        result = await generate_sql(
            query="How many venues?", relevant_views=[], model="openai/gpt-oss-120b",
            api_key="fake", base_url="https://example.test", diagnostics=diagnostics,
        )

    assert result == [valid]
    assert mock_chat.await_count == 2
    assert diagnostics[0]["outcome"] == "truncated_completion"
    assert diagnostics[0]["completion_evidence"] == {
        "response_id": "primary-response",
        "finish_reason": "length",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "configured_output_token_limit": 1024,
        "model": "openai/gpt-oss-120b",
        "provider": "provider-a",
    }
    assert diagnostics[1]["fallback_trigger_reason"] == "truncated_completion"


@pytest.mark.asyncio
async def test_parser_rejection_does_not_invoke_fallback():
    raw = "SELECT venue FROM v_shift_summary WHERE venue = '"
    diagnostics: list[dict] = []
    with patch(
        "src.retrieval.text_to_sql.chat", AsyncMock(return_value=_structured(raw))
    ) as mock_chat:
        result = await generate_sql(
            query="How many venues?", relevant_views=[], model="openai/gpt-oss-120b",
            api_key="fake", base_url="https://example.test", diagnostics=diagnostics,
        )

    assert result == []
    mock_chat.assert_awaited_once()
    assert diagnostics[0]["outcome"] == "no_valid_candidates"
    assert diagnostics[0]["candidate_diagnostics"][0]["parser_rejection"] == "unbalanced_quotes"


@pytest.mark.asyncio
async def test_case_19_empty_then_timeout_uses_fallback_and_executes():
    valid = (
        "SELECT p.shift_id, p.staff_count, SUM(i.incident_count) AS incident_count "
        "FROM v_peak_staffing AS p JOIN v_incident_by_venue AS i "
        "ON i.venue = p.venue AND i.date = p.date "
        "GROUP BY p.shift_id, p.staff_count ORDER BY p.staff_count DESC LIMIT 1"
    )
    from src.config import Settings

    mock_chat = AsyncMock(side_effect=["", TimeoutError(), _structured(valid)])
    rows = [{"shift_id": "SH-006", "staff_count": 20, "incident_count": 2}]
    with (
        patch("src.retrieval.text_to_sql.chat", mock_chat),
        patch("src.retrieval.text_to_sql.validate_sql", AsyncMock(return_value=(True, ""))),
        patch("src.retrieval.text_to_sql.execute_sql", AsyncMock(return_value=rows)) as execute,
    ):
        result = await text_to_sql_tool(
            _GOLDEN[19]["question"], [],
            Settings(_env_file=None, openrouter_api_key="fake", db_url=_DB),
            "case-19",
        )

    assert result["rows"] == rows
    execute.assert_awaited_once_with(valid, _DB)
    assert [call.kwargs["model"] for call in mock_chat.await_args_list] == [
        "openai/gpt-oss-120b", "openai/gpt-oss-120b", SQL_FALLBACK_MODEL,
    ]


@pytest.mark.asyncio
async def test_simple_sql_requests_one_candidate_and_retries_validation_failure():
    from src.config import Settings

    settings = Settings(_env_file=None, openrouter_api_key="fake", db_url=_DB)
    first = "SELECT missing FROM v_peak_staffing"
    second = "SELECT 17 AS staff_count"
    with (
        patch("src.retrieval.text_to_sql.generate_sql", new_callable=AsyncMock) as generate,
        patch("src.retrieval.text_to_sql.validate_sql", new_callable=AsyncMock) as validate,
        patch("src.retrieval.text_to_sql.execute_sql", new_callable=AsyncMock) as execute,
    ):
        generate.side_effect = [[first], [second]]
        validate.side_effect = [(False, "missing column"), (True, "")]
        execute.return_value = [{"staff_count": 17}]
        result = await text_to_sql_tool(
            "How many staff?", [], settings, "trace", simple_sql=True
        )
    assert result["rows"] == [{"staff_count": 17}]
    assert result["latency"]["retries"] == 1
    assert len(result["latency"]["attempts"]) == 2
    assert generate.await_count == 2
    assert all(call.kwargs["n_candidates"] == 1 for call in generate.await_args_list)
    execute.assert_awaited_once_with(second, _DB)


@pytest.mark.asyncio
async def test_simple_sql_records_validation_timeout_then_retries(monkeypatch):
    from src.config import Settings

    async def validate(sql, _db_url):
        if sql == "SELECT slow":
            await asyncio.sleep(0.1)
        return True, ""

    monkeypatch.setattr("src.retrieval.text_to_sql.SQL_VALIDATION_TIMEOUT_S", 0.001)
    settings = Settings(_env_file=None, openrouter_api_key="fake", db_url=_DB)
    with (
        patch("src.retrieval.text_to_sql.generate_sql", new_callable=AsyncMock) as generate,
        patch("src.retrieval.text_to_sql.validate_sql", side_effect=validate),
        patch("src.retrieval.text_to_sql.execute_sql", new_callable=AsyncMock) as execute,
    ):
        generate.side_effect = [["SELECT slow"], ["SELECT 17 AS staff_count"]]
        execute.return_value = [{"staff_count": 17}]
        result = await text_to_sql_tool("How many staff?", [], settings, "trace", simple_sql=True)
    assert result["rows"] == [{"staff_count": 17}]
    assert result["latency"]["attempts"][0]["validation_timed_out"] is True
    assert result["latency"]["attempts"][1]["validation_timed_out"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "raw", "expected"),
    [
        (
            "On which shifts did staffing fall short?",
            "SELECT 1 AS value WHERE '2026-05-27' BETWEEN "
            "DATE '2026-05-01' AND DATE '2026-05-31'",
            "SELECT 1 AS value WHERE '2026-05-27' BETWEEN '2026-05-01' AND '2026-05-31'",
        ),
        (
            "Did security staffing meet the requirement?",
            "SELECT count, required_count FROM v_staffing_gap "
            "WHERE date = DATE '2026-05-25'",
            "SELECT count, required_count FROM v_staffing_gap WHERE date = '2026-05-25'",
        ),
        (
            "How many RSA interventions occurred?",
            "Here is the SQL:\nSELECT date, incident_count FROM v_incident_by_venue",
            "SELECT date, incident_count FROM v_incident_by_venue",
        ),
    ],
)
async def test_generate_sql_outputs_sqlite_sql_only(query, raw, expected):
    with patch("src.retrieval.text_to_sql.chat", AsyncMock(return_value=_structured(raw))):
        result = await generate_sql(
            query=query,
            relevant_views=[],
            model="test-model",
            api_key="fake-key",
            base_url="https://example.test",
    )
    assert result == ([] if not expected else [expected])


@pytest.mark.asyncio
async def test_generate_sql_repairs_live_case_16_dangling_limit():
    raw = (
        "SELECT count AS floor_staff_count, required_count, "
        "CASE WHEN gap >= 0 THEN true ELSE false END AS met_requirement "
        "FROM v_staffing_gap WHERE venue = 'Fairfield RSL' "
        "AND date = '2026-05-28' AND role = 'floor' LIMIT"
    )
    with patch("src.retrieval.text_to_sql.chat", AsyncMock(return_value=_structured(raw))):
        result = await generate_sql(
            query=(
                "Did floor staffing at Fairfield RSL meet the required count on the night "
                "of 2026-05-28?"
            ),
            relevant_views=[],
            model="test-model",
            api_key="fake-key",
            base_url="https://example.test",
        )
    assert result == [raw.removesuffix(" LIMIT").replace("'floor'", "'floor_staff'")]


@pytest.mark.asyncio
async def test_generate_sql_skips_null_incident_placeholder_for_correct_candidate():
    raw = (
        "SELECT p.shift_id, p.staff_count, SUM(i.incident_count) AS incident_count "
        "FROM v_peak_staffing AS p JOIN v_incident_by_venue AS i "
        "ON i.venue = p.venue AND i.date = p.date "
        "GROUP BY p.shift_id, p.staff_count ORDER BY p.staff_count DESC LIMIT 1"
    )
    with patch("src.retrieval.text_to_sql.chat", AsyncMock(return_value=_structured(raw))):
        result = await generate_sql(
            query=(
                "Which shift had the largest staff count, and how many incidents were logged on it?"
            ),
            relevant_views=[],
            model="test-model",
            api_key="fake-key",
            base_url="https://example.test",
        )
    assert result == [raw]


@pytest.mark.asyncio
async def test_generate_sql_diagnostics_capture_safe_case_19_rejections():
    diagnostics = []
    malformed = "Authorization: Bearer secret-token-value\nI cannot provide SQL."
    null_placeholder = (
        "SQL: SELECT shift_id, staff_count, NULL AS incident_count FROM v_peak_staffing"
    )
    with patch(
        "src.retrieval.text_to_sql.chat",
        AsyncMock(side_effect=[_structured(malformed), _structured(null_placeholder)]),
    ):
        result = await generate_sql(
            query=(
                "Which shift in May 2026 had the largest staff count, and how many "
                "incidents were logged on it?"
            ),
            relevant_views=[],
            model="test-model",
            api_key="fake-key",
            base_url="https://example.test",
            diagnostics=diagnostics,
            diagnostic_case_id=19,
        )

    assert result == []
    assert [item["model_attempt"] for item in diagnostics] == [1]
    assert all(item["case_id"] == 19 for item in diagnostics)
    assert diagnostics[0]["candidate_diagnostics"] == [{
        "candidate_index": 1,
        "parser_rejection": "not_select_or_with",
        "filter_rejections": [],
    }]
    assert "secret-token-value" not in diagnostics[0]["raw_response"]["excerpt"]
    assert "[REDACTED]" in diagnostics[0]["raw_response"]["excerpt"]
    assert diagnostics[0]["outcome"] == "no_valid_candidates"
    assert diagnostics[0]["will_retry"] is False


@pytest.mark.asyncio
async def test_live_case_17_does_not_retry_after_filter_rejection():
    first = (
        "SELECT date, count, required_count FROM v_staffing_gap "
        "WHERE venue = 'Fairfield RSL' AND gap < 0"
    )
    second = (
        "SELECT date, count, required_count FROM v_staffing_gap "
        "WHERE venue = 'Fairfield RSL' AND role = 'floor_staff' AND gap < 0"
    )
    mock_chat = AsyncMock(side_effect=[_structured(first), _structured(second)])
    with patch("src.retrieval.text_to_sql.chat", mock_chat):
        result = await generate_sql(
            query=(
                "On which May 2026 shifts did Fairfield RSL fall short of its required "
                "floor-staff count?"
            ),
            relevant_views=[],
            model="test-model",
            api_key="fake-key",
            base_url="https://example.test",
        )
    assert result == []
    mock_chat.assert_awaited_once()


def test_live_case_17_floor_staff_predicate_excludes_bar_and_security_rows(tmp_path):
    connection = sqlite3.connect(tmp_path / "case_17.db")
    connection.execute(
        "CREATE TABLE staffing_history (venue TEXT, date TEXT, role TEXT, count INTEGER, "
        "required_count INTEGER)"
    )
    connection.executemany(
        "INSERT INTO staffing_history VALUES (?, ?, ?, ?, ?)",
        [
            ("Fairfield RSL", "2026-05-27", "floor_staff", 6, 8),
            ("Fairfield RSL", "2026-05-27", "bar_staff", 3, 4),
            ("Fairfield RSL", "2026-05-25", "security", 2, 3),
        ],
    )
    connection.execute(
        "CREATE VIEW v_staffing_gap AS SELECT venue, date, role, count, required_count, "
        "count - required_count AS gap FROM staffing_history"
    )
    rows = connection.execute(
        "SELECT date, count, required_count FROM v_staffing_gap "
        "WHERE venue = 'Fairfield RSL' AND role = 'floor_staff' AND gap < 0 "
        "ORDER BY date DESC",
    ).fetchall()
    connection.close()
    assert rows == [
        ("2026-05-27", 6, 8),
    ]


@pytest.mark.asyncio
async def test_live_case_19_does_not_retry_after_parser_rejection():
    valid = (
        "SELECT p.shift_id, p.staff_count, SUM(i.incident_count) AS incident_count "
        "FROM v_peak_staffing AS p JOIN v_incident_by_venue AS i "
        "ON i.venue = p.venue AND i.date = p.date "
        "GROUP BY p.shift_id, p.staff_count ORDER BY p.staff_count DESC LIMIT 1"
    )
    mock_chat = AsyncMock(side_effect=[_structured("SELECT"), _structured(valid)])
    with patch("src.retrieval.text_to_sql.chat", mock_chat):
        result = await generate_sql(
            query=(
                "Which shift in May 2026 had the largest staff count, and how many "
                "incidents were logged on it?"
            ),
            relevant_views=[],
            model="test-model",
            api_key="fake-key",
            base_url="https://example.test",
        )
    assert result == []
    mock_chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_live_case_19_does_not_retry_after_truncated_sql():
    valid = (
        "SELECT p.shift_id, p.staff_count, SUM(i.incident_count) AS incident_count "
        "FROM v_peak_staffing AS p JOIN v_incident_by_venue AS i "
        "ON i.venue = p.venue AND i.date = p.date "
        "GROUP BY p.shift_id, p.staff_count ORDER BY p.staff_count DESC LIMIT 1"
    )
    mock_chat = AsyncMock(side_effect=[
        _structured("SELECT COALESCE(i.incident_count, 0"), _structured(valid)
    ])
    with patch("src.retrieval.text_to_sql.chat", mock_chat):
        result = await generate_sql(
            query=(
                "Which shift had the largest staff count, and how many incidents were "
                "logged on it?"
            ),
            relevant_views=[],
            model="test-model",
            api_key="fake-key",
            base_url="https://example.test",
        )
    assert result == []
    mock_chat.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("SELECT", ""),
        ("SELECT date FROM v_incident_by_venue WHERE incident_type = 'RSA-intervention' AND", ""),
        ("SELECT venue FROM v_staffing_gap WHERE venue = 'Fairfield R", ""),
        ("SELECT shift_id,", ""),
        (
            "SELECT date, count FROM v_staffing_gap WHERE role = 'floor-staff'",
            "SELECT date, count FROM v_staffing_gap WHERE role = 'floor_staff'",
        ),
        (
            "SELECT date, count FROM v_staffing_gap WHERE role = 'Security'",
            "SELECT date, count FROM v_staffing_gap WHERE role = 'security'",
        ),
    ],
)
async def test_generate_sql_rejects_malformed_candidates_and_normalizes_roles(raw, expected):
    with patch("src.retrieval.text_to_sql.chat", AsyncMock(return_value=_structured(raw))):
        result = await generate_sql(
            query="Did staffing meet the requirement?",
            relevant_views=[],
            model="test-model",
            api_key="fake-key",
            base_url="https://example.test",
        )
    assert result == ([] if not expected else [expected])


@pytest.mark.asyncio
async def test_generate_sql_adds_incident_aggregate_context_for_incident_questions():
    with patch("src.retrieval.text_to_sql.chat", AsyncMock(return_value=_structured("SELECT 1"))) as chat:
        await generate_sql(
            query="How many incidents occurred at Fairfield RSL?",
            relevant_views=[{"view_name": "v_peak_staffing", "columns": [], "description": "x"}],
            model="test-model",
            api_key="fake-key",
            base_url="https://example.test",
        )
    assert "View: v_incident_by_venue" in chat.call_args.kwargs["prompt"]


@pytest.mark.asyncio
async def test_case_19_prompt_requires_shift_identifier_with_linked_views():
    views = [
        {
            "view_name": "v_shift_summary",
            "columns": ["venue", "date", "shift_type", "total_staff"],
            "description": "Aggregated staff counts per venue, date, and shift type.",
        },
        {
            "view_name": "v_peak_staffing",
            "columns": ["shift_id", "venue", "date", "staff_count", "shift_type", "avg_staff"],
            "description": "Shifts where staff count exceeded the venue average.",
        },
        {
            "view_name": "v_staffing_gap",
            "columns": ["venue", "date", "role", "count", "required_count", "gap"],
            "description": "Staffing gap analysis.",
        },
        {
            "view_name": "v_incident_by_venue",
            "columns": ["venue", "date", "incident_type", "incident_count"],
            "description": "Count of incidents grouped by venue, date, and incident type.",
        },
    ]
    with patch("src.retrieval.text_to_sql.chat", AsyncMock(return_value=_structured("SELECT 1"))) as chat:
        await generate_sql(
            _GOLDEN[19]["question"], views, "test-model", "fake-key", "https://example.test"
        )

    prompt = chat.call_args.kwargs["prompt"]
    assert "'which shift' requires shift_id" in prompt
    assert "View: v_shift_summary" in prompt
    assert "View: v_peak_staffing" in prompt
    assert "View: v_staffing_gap" in prompt
    assert "View: v_incident_by_venue" in prompt


@pytest.mark.asyncio
async def test_case_19_peak_staffing_and_incident_candidate_remains_accepted():
    candidate = (
        "SELECT p.shift_id, p.staff_count, SUM(i.incident_count) AS incident_count "
        "FROM v_peak_staffing AS p JOIN v_incident_by_venue AS i "
        "ON i.venue = p.venue AND i.date = p.date "
        "GROUP BY p.shift_id, p.staff_count ORDER BY p.staff_count DESC LIMIT 1"
    )
    with patch("src.retrieval.text_to_sql.chat", AsyncMock(return_value=_structured(candidate))):
        result = await generate_sql(
            _GOLDEN[19]["question"], [
                {
                    "view_name": "v_peak_staffing",
                    "columns": ["shift_id", "venue", "date", "staff_count"],
                    "description": "Peak staffing shifts.",
                },
                {
                    "view_name": "v_incident_by_venue",
                    "columns": ["venue", "date", "incident_count"],
                    "description": "Incident counts by venue and date.",
                },
            ], "test-model", "fake-key", "https://example.test"
        )

    assert result == [candidate]


def _staffing_views() -> list[dict]:
    return [
        {
            "view_name": "v_shift_summary",
            "columns": ["venue", "date", "shift_type", "total_staff"],
            "description": "Aggregated staff counts per venue, date, and shift type.",
        },
        {
            "view_name": "v_staffing_gap",
            "columns": ["venue", "date", "role", "count", "required_count", "gap"],
            "description": "Staffing gap analysis where a negative gap means understaffed.",
        },
        {
            "view_name": "v_peak_staffing",
            "columns": ["shift_id", "venue", "date", "staff_count", "shift_type", "avg_staff"],
            "description": "Shifts where staff count exceeded the venue average.",
        },
    ]


@pytest.mark.asyncio
async def test_case_17_prompt_prefers_self_contained_views_and_compatible_joins():
    with patch("src.retrieval.text_to_sql.chat", AsyncMock(return_value=_structured("SELECT 1"))) as chat:
        await generate_sql(
            _GOLDEN[17]["question"], _staffing_views(), "test-model", "fake-key", "https://example.test"
        )

    prompt = chat.call_args.kwargs["prompt"]
    assert "Prefer one self-contained view" in prompt
    assert "only on semantically compatible entity keys" in prompt
    assert "never equate unrelated categorical domains such as shift type and staffing role" in prompt
    assert "query v_staffing_gap directly" in prompt
    assert "do not join v_peak_staffing" in prompt
    assert "role = 'floor_staff' and gap < 0" in prompt
    assert "date, count, and required_count" in prompt


@pytest.mark.asyncio
async def test_case_17_staffing_gap_candidate_remains_accepted():
    candidate = (
        "SELECT date, count AS floor_staff_count, required_count FROM v_staffing_gap "
        "WHERE venue = 'Fairfield RSL' AND date LIKE '2026-05-%' "
        "AND role = 'floor_staff' AND gap < 0"
    )
    with patch("src.retrieval.text_to_sql.chat", AsyncMock(return_value=_structured(candidate))):
        result = await generate_sql(
            _GOLDEN[17]["question"], _staffing_views(), "test-model", "fake-key", "https://example.test"
        )
    assert result == [candidate]


@pytest.mark.asyncio
async def test_floor_staff_shortfall_listing_rejects_unnecessary_peak_staffing_join():
    candidate = (
        "SELECT g.date, g.count, g.required_count FROM v_peak_staffing AS p "
        "JOIN v_staffing_gap AS g ON p.venue = g.venue AND p.date = g.date "
        "WHERE g.venue = 'Fairfield RSL' AND g.date LIKE '2026-05-%' "
        "AND g.role = 'floor_staff' AND g.gap < 0"
    )
    with patch("src.retrieval.text_to_sql.chat", AsyncMock(return_value=_structured(candidate))):
        result = await generate_sql(
            "Which May shifts had understaffed floor staff?",
            _staffing_views(), "test-model", "fake-key", "https://example.test",
        )
    assert result == []


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [
    "Which May shifts had understaffed floor staff?",
    "List May dates with floor staff below the required count.",
    "Show the May floor-staff staffing deficits.",
])
async def test_floor_staff_shortfall_listing_paraphrases_accept_direct_gap_query(query: str):
    candidate = (
        "SELECT date, count, required_count FROM v_staffing_gap "
        "WHERE venue = 'Fairfield RSL' AND date LIKE '2026-05-%' "
        "AND role = 'floor_staff' AND gap < 0"
    )
    with patch("src.retrieval.text_to_sql.chat", AsyncMock(return_value=_structured(candidate))):
        result = await generate_sql(
            query, _staffing_views(), "test-model", "fake-key", "https://example.test"
        )
    assert result == [candidate]


def test_case_17_incompatible_shift_type_to_role_join_returns_no_rows(tmp_path):
    connection = sqlite3.connect(tmp_path / "case_17_join.db")
    connection.execute(
        "CREATE TABLE staffing_history (venue TEXT, date TEXT, role TEXT, count INTEGER, "
        "required_count INTEGER)"
    )
    connection.execute(
        "CREATE TABLE shift_summary (venue TEXT, date TEXT, shift_type TEXT, total_staff INTEGER)"
    )
    connection.executemany(
        "INSERT INTO staffing_history VALUES (?, ?, ?, ?, ?)",
        [
            ("Fairfield RSL", "2026-05-22", "floor_staff", 4, 8),
            ("Fairfield RSL", "2026-05-25", "floor_staff", 5, 8),
            ("Fairfield RSL", "2026-05-27", "floor_staff", 6, 8),
        ],
    )
    connection.executemany(
        "INSERT INTO shift_summary VALUES (?, ?, ?, ?)",
        [
            ("Fairfield RSL", "2026-05-22", "Evening", 12),
            ("Fairfield RSL", "2026-05-25", "Evening", 14),
            ("Fairfield RSL", "2026-05-27", "Evening", 13),
        ],
    )
    connection.execute(
        "CREATE VIEW v_staffing_gap AS SELECT venue, date, role, count, required_count, "
        "count - required_count AS gap FROM staffing_history"
    )
    connection.execute("CREATE VIEW v_shift_summary AS SELECT * FROM shift_summary")
    faulty_rows = connection.execute(
        "SELECT s.venue, s.date, s.shift_type, s.total_staff, g.gap "
        "FROM v_shift_summary AS s JOIN v_staffing_gap AS g "
        "ON s.venue = g.venue AND s.date = g.date AND s.shift_type = g.role "
        "WHERE s.date LIKE '2026-05-%' AND s.venue = 'Fairfield RSL' "
        "AND g.role = 'floor_staff' AND g.gap < 0"
    ).fetchall()
    correct_rows = connection.execute(
        "SELECT date, count AS floor_staff_count, required_count FROM v_staffing_gap "
        "WHERE venue = 'Fairfield RSL' AND date LIKE '2026-05-%' "
        "AND role = 'floor_staff' AND gap < 0 ORDER BY date"
    ).fetchall()
    connection.close()

    assert faulty_rows == []
    assert correct_rows == [
        ("2026-05-22", 4, 8),
        ("2026-05-25", 5, 8),
        ("2026-05-27", 6, 8),
    ]


@pytest.mark.asyncio
async def test_case_21_prompt_requires_operands_for_derived_status():
    views = _staffing_views()[1:] + _staffing_views()[:1]
    with patch("src.retrieval.text_to_sql.chat", AsyncMock(return_value=_structured("SELECT 1"))) as chat:
        await generate_sql(
            _GOLDEN[21]["question"], views, "test-model", "fake-key", "https://example.test"
        )

    prompt = chat.call_args.kwargs["prompt"]
    assert "return every requested base metric" in prompt
    assert "derived comparison, boolean, or status" in prompt
    assert "return the available input metrics used to establish it" in prompt


@pytest.mark.asyncio
async def test_case_21_status_with_actual_and_required_counts_remains_accepted():
    candidate = (
        "SELECT count AS security_count, required_count, "
        "CASE WHEN gap >= 0 THEN 'Yes' ELSE 'No' END AS met_requirement "
        "FROM v_staffing_gap WHERE venue = 'Fairfield RSL' "
        "AND date = '2026-05-25' AND role = 'security'"
    )
    views = _staffing_views()[1:] + _staffing_views()[:1]
    with patch("src.retrieval.text_to_sql.chat", AsyncMock(return_value=_structured(candidate))):
        result = await generate_sql(
            _GOLDEN[21]["question"], views, "test-model", "fake-key", "https://example.test"
        )
    assert result == [candidate]


@pytest.mark.asyncio
async def test_validate_sql_rejects_markdown_or_prose():
    valid, error = await validate_sql("Here is SQL: SELECT 1", _DB)
    assert valid is False
    assert "SELECT" in error


@pytest.mark.asyncio
async def test_validate_sql_rejects_dml():
    valid, err = await validate_sql("INSERT INTO t VALUES (1)", _DB)
    assert valid is False
    assert "DML" in err or "not permitted" in err


@pytest.mark.asyncio
async def test_validate_sql_rejects_deep_join():
    sql = (
        "SELECT a.x FROM t1 a "
        "JOIN t2 b ON a.id=b.id "
        "JOIN t3 c ON b.id=c.id "
        "JOIN t4 d ON c.id=d.id "
        "JOIN t5 e ON d.id=e.id"
    )
    valid, err = await validate_sql(sql, _DB)
    assert valid is False
    assert "join" in err.lower() or "depth" in err.lower()


@pytest.mark.asyncio
async def test_execute_sql_happy_path():
    rows = await execute_sql("SELECT 1 AS val", _DB)
    assert isinstance(rows, list)
    assert rows[0]["val"] == 1


@pytest.mark.asyncio
async def test_sql_execution_error_increments_counter():
    """rag_sql_execution_errors is incremented once per failed execute_sql call."""
    raw = "```sql\nSELECT 1\n```"

    from src.config import Settings
    settings = Settings(openrouter_api_key="fake", db_url=_DB)

    with (
        patch("src.retrieval.text_to_sql.chat", AsyncMock(return_value=_structured(raw))),
        patch("src.retrieval.text_to_sql.validate_sql", AsyncMock(return_value=(True, ""))),
        patch(
            "src.retrieval.text_to_sql.execute_sql",
            AsyncMock(side_effect=RuntimeError("table not found")),
        ),
        patch("src.retrieval.text_to_sql.rag_sql_execution_errors") as mock_counter,
    ):
        result = await text_to_sql_tool(
            query="get data",
            relevant_views=[{
                "view_name": "v_shift_summary",
                "columns": ["venue"],
                "description": "shift data",
            }],
            settings=settings,
            trace_id="trace-err",
        )

    mock_counter.inc.assert_called_once()
    assert "error" in result or result.get("rows") == []


@pytest.mark.asyncio
async def test_self_correction_tries_next_candidate():
    raw = "SELECT 1 AS ok"

    from src.config import Settings
    settings = Settings(openrouter_api_key="fake", db_url=_DB)

    with patch("src.retrieval.text_to_sql.chat", AsyncMock(return_value=_structured(raw))):
        result = await text_to_sql_tool(
            query="get data",
            relevant_views=[{
                "view_name": "v_shift_summary",
                "columns": ["venue"],
                "description": "x",
            }],
            settings=settings,
            trace_id="trace-abc",
        )
    assert "rows" in result
    if "error" not in result:
        assert isinstance(result["rows"], list)


# PLAN-009 regressions. _GOLDEN is keyed by query_id (golden index + 1).
from src.retrieval.text_to_sql import (  # noqa: E402
    _filter_rejection_reasons,
    _supervisor_comparison_sql,
    _with_required_sql_views,
)

_PATRON_QUESTION = _GOLDEN[15]["question"]  # golden 14: peak patron count
_LARGEST_SHIFT_QUESTION = _GOLDEN[19]["question"]  # golden 18
_SUMMED_JOIN = (
    "SELECT p.shift_id, p.staff_count, SUM(i.incident_count) AS incident_count "
    "FROM v_peak_staffing AS p JOIN v_incident_by_venue AS i "
    "ON i.venue = p.venue AND i.date = p.date "
    "GROUP BY p.shift_id, p.staff_count ORDER BY p.staff_count DESC LIMIT 1"
)


def test_staff_column_relabelled_as_patron_metric_is_rejected() -> None:
    sql = (
        "SELECT staff_count AS peak_patron_count FROM v_peak_staffing "
        "WHERE shift_id = 'SH-001' AND date = '2026-05-28' LIMIT 1"
    )
    assert "unsupported_metric_alias" in _filter_rejection_reasons(_PATRON_QUESTION, sql)


@pytest.mark.parametrize("query,sql", [
    (
        "How many security staff were rostered on 2026-05-25?",
        "SELECT count AS security_count, required_count FROM v_staffing_gap "
        "WHERE date = '2026-05-25' AND role = 'security'",
    ),
    (
        "How many incidents were logged in total on shift SH-006?",
        "SELECT incident_count AS total_incidents FROM v_incident_by_shift "
        "WHERE shift_id = 'SH-006'",
    ),
    (
        "Was the supervisor of SH-001 the same supervisor as SH-002?",
        "SELECT a.supervisor AS incident_supervisor, "
        "CASE WHEN a.supervisor = b.supervisor THEN 1 ELSE 0 END AS same_supervisor "
        "FROM v_shift_details AS a CROSS JOIN v_shift_details AS b "
        "WHERE a.shift_id = 'SH-001' AND b.shift_id = 'SH-002'",
    ),
])
def test_curated_derived_and_filter_value_aliases_are_accepted(query: str, sql: str) -> None:
    assert "unsupported_metric_alias" not in _filter_rejection_reasons(query, sql)


def test_unsummed_incident_join_is_rejected_but_summed_join_is_accepted() -> None:
    bare = (
        "SELECT v1.shift_id, v1.staff_count, v2.incident_count FROM v_peak_staffing v1 "
        "LEFT JOIN v_incident_by_venue v2 ON v1.venue = v2.venue AND v1.date = v2.date "
        "ORDER BY v1.staff_count DESC LIMIT 1"
    )
    assert "incident_count_grain_mismatch" in _filter_rejection_reasons(
        _LARGEST_SHIFT_QUESTION, bare
    )
    assert _filter_rejection_reasons(_LARGEST_SHIFT_QUESTION, _SUMMED_JOIN) == []


def test_which_shift_question_requires_shift_id_in_select_list() -> None:
    by_type = (
        "SELECT shift_type, total_staff FROM v_shift_summary "
        "WHERE date LIKE '2026-05-%' ORDER BY total_staff DESC LIMIT 1"
    )
    assert "missing_requested_shift_id" in _filter_rejection_reasons(
        _LARGEST_SHIFT_QUESTION, by_type
    )


def test_shift_incident_question_offers_the_per_shift_incident_view() -> None:
    names = [v["view_name"] for v in _with_required_sql_views(_LARGEST_SHIFT_QUESTION, [])]
    assert names == ["v_incident_by_venue", "v_incident_by_shift"]


def test_supervisor_comparison_returns_the_identities_it_compares() -> None:
    sql = _supervisor_comparison_sql(_GOLDEN[20]["question"])
    for column in ("incident_id", "incident_shift_id", "comparison_shift_id", "same_supervisor"):
        assert column in sql


@pytest.mark.asyncio
async def test_supervisor_comparison_rows_cite_incident_and_shift_ids() -> None:
    rows = await execute_sql(_supervisor_comparison_sql(_GOLDEN[20]["question"]), _DB)
    assert rows[0]["incident_id"] == "INC-007"
    assert rows[0]["incident_shift_id"] == "SH-010"
    assert rows[0]["incident_supervisor"] == "S. Lee"


def _tool_settings():
    from src.config import Settings

    return Settings(_env_file=None, openrouter_api_key="fake", db_url=_DB)


@pytest.mark.asyncio
async def test_invented_metric_alias_reports_schema_capability_gap() -> None:
    sql = "SELECT staff_count AS peak_patron_count FROM v_peak_staffing LIMIT 1"
    with patch("src.retrieval.text_to_sql.chat", AsyncMock(return_value=_structured(sql))):
        result = await text_to_sql_tool(
            _PATRON_QUESTION, [], _tool_settings(), "t", simple_sql=True
        )
    assert result["rows"] == []
    assert result["fallback_reason"] == "schema_capability_gap"


@pytest.mark.asyncio
async def test_invented_schema_identifiers_report_schema_capability_gap() -> None:
    sql = "SELECT signoff_person FROM v_shift_summary WHERE date = '2026-05-28'"
    with (
        patch("src.retrieval.text_to_sql.chat", AsyncMock(return_value=_structured(sql))),
        patch("src.retrieval.text_to_sql.validate_sql", AsyncMock(return_value=(
            False, "(sqlite3.OperationalError) no such column: signoff_person",
        ))),
    ):
        result = await text_to_sql_tool(
            _GOLDEN[14]["question"], [], _tool_settings(), "t", simple_sql=True
        )
    assert result["rows"] == []
    assert result["fallback_reason"] == "schema_capability_gap"


@pytest.mark.asyncio
async def test_valid_query_with_zero_rows_is_not_a_schema_gap() -> None:
    sql = "SELECT supervisor FROM v_shift_details WHERE shift_id = 'SH-999'"
    with (
        patch("src.retrieval.text_to_sql.chat", AsyncMock(return_value=_structured(sql))),
        patch("src.retrieval.text_to_sql.validate_sql", AsyncMock(return_value=(True, ""))),
        patch("src.retrieval.text_to_sql.execute_sql", AsyncMock(return_value=[])),
    ):
        result = await text_to_sql_tool(
            "Who supervised shift SH-999?", [], _tool_settings(), "t", simple_sql=True
        )
    assert result["rows"] == []
    assert "fallback_reason" not in result


@pytest.mark.asyncio
async def test_syntax_error_failure_is_not_a_schema_gap() -> None:
    sql = "SELECT supervisor FROM v_shift_details WHERE shift_id = 'SH-003'"
    with (
        patch("src.retrieval.text_to_sql.chat", AsyncMock(return_value=_structured(sql))),
        patch("src.retrieval.text_to_sql.validate_sql", AsyncMock(return_value=(
            False, '(sqlite3.OperationalError) near "WHERE": syntax error',
        ))),
    ):
        result = await text_to_sql_tool(
            "Who supervised shift SH-003?", [], _tool_settings(), "t", simple_sql=True
        )
    assert result["rows"] == []
    assert "fallback_reason" not in result


@pytest.mark.asyncio
async def test_schema_gap_ends_retry_loop_before_a_wrong_view_can_answer() -> None:
    # Golden 13: a later retry once returned rows from v_peak_staffing and
    # overrode the gap signal. The first schema-gap failure must end the loop.
    invented = "SELECT signed_off_by FROM v_shift_summary WHERE date = '2026-05-28'"
    wrong_view = "SELECT shift_id, venue FROM v_peak_staffing WHERE date = '2026-05-28'"
    chat = AsyncMock(side_effect=[_structured(invented), _structured(wrong_view)])
    execute = AsyncMock(return_value=[{"shift_id": "SH-001", "venue": "Fairfield RSL"}])
    with (
        patch("src.retrieval.text_to_sql.chat", chat),
        patch("src.retrieval.text_to_sql.execute_sql", execute),
    ):
        result = await text_to_sql_tool(
            _GOLDEN[14]["question"], [], _tool_settings(), "t", simple_sql=True
        )
    assert chat.await_count == 1
    execute.assert_not_awaited()
    assert result["rows"] == []
    assert result["fallback_reason"] == "schema_capability_gap"


@pytest.mark.asyncio
async def test_non_schema_failure_still_retries() -> None:
    sql = "SELECT supervisor FROM v_shift_details WHERE shift_id = 'SH-003'"
    rows = [{"supervisor": "J. Thompson"}]
    with (
        patch("src.retrieval.text_to_sql.chat", AsyncMock(return_value=_structured(sql))) as chat,
        patch("src.retrieval.text_to_sql.validate_sql", AsyncMock(side_effect=[
            (False, '(sqlite3.OperationalError) near "WHERE": syntax error'), (True, ""),
        ])),
        patch("src.retrieval.text_to_sql.execute_sql", AsyncMock(return_value=rows)),
    ):
        result = await text_to_sql_tool(
            "Who supervised shift SH-003?", [], _tool_settings(), "t", simple_sql=True
        )
    assert chat.await_count == 2
    assert result["rows"] == rows
    assert "fallback_reason" not in result
