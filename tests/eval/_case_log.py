"""Per-case evidence log for the live DeepEval gate.

Diagnostic only: it records what each case saw and produced, and changes no
product behavior. SQL attempts are captured by wrapping text_to_sql's
validate_sql/execute_sql inside the test process, so every attempt appears,
including failed attempts and retries. Output is JSON Lines, one record per
case, at $RAG_EVAL_CASE_LOG (default .logs/deepeval_cases.jsonl).

RAG_EVAL_GATE_TRACE=1 (D-003, default off) adds a "trace" field: the pre-gate
draft answer, the faithfulness-gate judge's raw reply, per-call gate and
DeepEval-judge latency, and case wall-clock timing. The draft is unredacted
generator output, so it is written only to this log, never to a response.
The case timing exists for case 08, which was deferred before this was added:
it is observability only and blocks no gate. Unset the variable after the run.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

# Same pattern as text_to_sql._MISSING_SCHEMA_IDENTIFIER, the schema-gap signal.
_SCHEMA_GAP = re.compile(r"\bno such (?:column|table)\b", re.IGNORECASE)
DEFAULT_LOG = ".logs/deepeval_cases.jsonl"
GATE_TRACE_ENV = "RAG_EVAL_GATE_TRACE"


def gate_trace_enabled() -> bool:
    return os.environ.get(GATE_TRACE_ENV) == "1"


@contextmanager
def capture_sql_attempts() -> Iterator[list[dict[str, Any]]]:
    """Record every SQL validation and execution made while the block runs."""
    from src.retrieval import text_to_sql

    attempts: list[dict[str, Any]] = []
    validate, execute = text_to_sql.validate_sql, text_to_sql.execute_sql

    async def logged_validate(sql: str, db_url: str) -> tuple[bool, str]:
        valid, error = await validate(sql, db_url)
        attempts.append({"stage": "validate", "sql": sql, "ok": valid, "error": error or None})
        return valid, error

    async def logged_execute(sql: str, db_url: str) -> list[dict]:
        try:
            rows = await execute(sql, db_url)
        except RuntimeError as exc:
            attempts.append({"stage": "execute", "sql": sql, "ok": False, "error": str(exc)})
            raise
        attempts.append({"stage": "execute", "sql": sql, "ok": True, "error": None,
                         "row_count": len(rows), "rows": rows[:10]})
        return rows

    with (
        patch.object(text_to_sql, "validate_sql", logged_validate),
        patch.object(text_to_sql, "execute_sql", logged_execute),
    ):
        yield attempts


@contextmanager
def capture_gate_trace() -> Iterator[dict[str, Any] | None]:
    """Record every faithfulness-gate call and its raw judge reply; None when off.

    Wrappers return what they wrap and re-raise what it raises, so the gate
    decides exactly as it would unobserved. A call cut off by the handler's
    timeout is recorded with the cancellation's class.
    """
    if not gate_trace_enabled():
        yield None
        return
    from src.generation import faithfulness_gate, query_handler

    trace: dict[str, Any] = {"gate_calls": [], "judge_replies": []}
    check, chat = query_handler.check_faithfulness, faithfulness_gate.chat

    async def logged_check(*, answer: str, evidence: str, **kwargs: Any) -> Any:
        call: dict[str, Any] = {
            "draft_answer": answer,
            "evidence_sha256": hashlib.sha256(evidence.encode()).hexdigest()[:12],
            "evidence_chars": len(evidence),
        }
        trace["gate_calls"].append(call)
        started = time.monotonic()
        try:
            decision = await check(answer=answer, evidence=evidence, **kwargs)
        except BaseException as exc:
            call["exception_class"] = type(exc).__name__
            raise
        finally:
            call["latency_ms"] = round((time.monotonic() - started) * 1000, 2)
        call["decision"] = decision.as_dict()
        return decision

    async def logged_chat(*args: Any, **kwargs: Any) -> Any:
        reply: dict[str, Any] = {}
        trace["judge_replies"].append(reply)
        started = time.monotonic()
        try:
            raw = await chat(*args, **kwargs)
        except BaseException as exc:
            reply["exception_class"] = type(exc).__name__
            raise
        finally:
            reply["latency_ms"] = round((time.monotonic() - started) * 1000, 2)
        reply["raw"] = raw
        return raw

    with (
        patch.object(query_handler, "check_faithfulness", logged_check),
        patch.object(faithfulness_gate, "chat", logged_chat),
    ):
        yield trace


def time_judge_calls(metrics: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Time each DeepEval metric.measure call in place; None (and no wrapping) when off."""
    if not gate_trace_enabled():
        return None
    calls: list[dict[str, Any]] = []
    for name, metric in metrics.items():
        def timed(test_case: Any, *args: Any, _name: str = name,
                  _measure: Any = metric.measure, **kwargs: Any) -> Any:
            call: dict[str, Any] = {"metric": _name, "started_at": time.time()}
            calls.append(call)
            started = time.monotonic()
            try:
                return _measure(test_case, *args, **kwargs)
            except BaseException as exc:
                call["exception_class"] = type(exc).__name__
                raise
            finally:
                call["latency_ms"] = round((time.monotonic() - started) * 1000, 2)
        metric.measure = timed
    return calls


def _gap_summary(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """Locate the first schema-gap error and anything that ran after it."""
    gap = next((i for i, a in enumerate(attempts) if _SCHEMA_GAP.search(a["error"] or "")), None)
    after = attempts[gap + 1:] if gap is not None else []
    return {
        "schema_gap_attempt_index": gap,
        "attempts_after_schema_gap": len(after),
        # Rows returned after a schema-gap error: the path PLAN-003 closed.
        "retry_override": any(a["stage"] == "execute" and a["ok"] for a in after),
    }


def write_case_log(
    case_id: str, case: dict, result: dict, retrieval_context: list[str],
    attempts: list[dict[str, Any]], metrics: dict[str, Any], runs: dict, medians: dict,
    trace: dict[str, Any] | None = None,
) -> None:
    """Append one JSON record with the full evidence for a gate case."""
    record = {
        "case": case_id,
        "question": case["question"],
        "ground_truth": case["ground_truth"],
        "tool": result.get("tool"),
        "route": result.get("route"),
        "answer": result.get("answer"),
        "faithfulness_gate": result.get("faithfulness_gate"),
        "sql_schema_fallback": (result.get("latency_components") or {}).get("sql_schema_fallback"),
        "sql_attempts": attempts,
        **_gap_summary(attempts),
        "citations": result.get("citations", []),
        "retrieval_context": retrieval_context,
        "judge_runs": runs,
        "judge_medians": medians,
        "judge_reasons_last_run": {
            name: getattr(metric, "reason", None) for name, metric in metrics.items()
        },
    }
    if trace is not None:
        record["trace"] = trace
    path = Path(os.environ.get("RAG_EVAL_CASE_LOG", DEFAULT_LOG))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
