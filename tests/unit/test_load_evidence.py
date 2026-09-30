"""Load-test evidence contract (PLAN-003 section 4): records, sanitization, server join."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from scripts import load_evidence as ev
from scripts import load_test

RUN = "load-20260924T120000Z-abc123"
QUESTION = "How many incidents on shift SH-006?"
ANSWER = "Two incidents were logged."


def test_run_id_format_and_override_validation():
    assert ev.RUN_ID.fullmatch(ev.new_run_id())
    assert ev.new_run_id(RUN) == RUN
    for bad in ("load-2026-abc", "load-20260924T120000Z-ABC123", "x"):
        with pytest.raises(ValueError):
            ev.new_run_id(bad)


@pytest.mark.parametrize("raw,leak", [
    ("Authorization: Bearer abc.def-ghi", "abc.def"),
    ("key sk-or-v1-0123456789abcdef", "0123456789"),
    ('{"api_key": "hunter2"}', "hunter2"),
    ("password=hunter2&x=1", "hunter2"),
    ("token: s3cr3t", "s3cr3t"),
    ("GET https://user:pw@db.internal/x", "user:pw"),
    ("see https://api.example.com/v1?key=abc123", "abc123"),
    ("mail ops.lead@fairfield.example.org now", "ops.lead@"),
])
def test_sanitize_redacts_secrets(raw, leak):
    assert leak not in ev.sanitize(raw)


def test_sanitize_strips_controls_replaces_bad_utf8_and_truncates():
    text = ev.sanitize(b"a\x00b\x1b[31m\xffc" + b"z" * 1000)
    assert "\x00" not in text and "\x1b" not in text and "�" in text
    assert len(text) == ev.EXCERPT_CHARS


def test_body_fields_kinds_and_error_codes():
    assert ev.body_fields(None, None, True)["body_kind"] == "none"
    assert ev.body_fields(b"", "text/plain", True)["body_kind"] == "empty"
    raw = b"<html>Internal Server Error</html>"
    non_json = ev.body_fields(raw, "text/html", True)
    assert non_json["body_kind"] == "non_json" and non_json["body_bytes"] == len(raw)
    assert non_json["body_excerpt"] == raw.decode() and len(non_json["body_sha256"]) == 64
    top = ev.body_fields(b'{"error": "query_failed", "request_id": "r"}', None, True)
    nested = ev.body_fields(b'{"detail": {"error": "query_failed"}}', None, True)
    assert top["error_code"] == nested["error_code"] == "query_failed"
    assert ev.body_fields(b'{"answer": "x"}', None, False)["body_excerpt"] is None


async def _send(handler) -> dict:
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        return await load_test.send_one(
            client, asyncio.Semaphore(1), RUN, 7, (22, QUESTION, "complex")
        )


def _echo(status: int, payload: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        rid = request.headers["X-Request-ID"]
        return httpx.Response(status, json=payload, headers={"X-Request-ID": rid})
    return handler


@pytest.mark.asyncio
async def test_success_record_has_identity_timings_and_no_query_or_answer_text():
    rec = await _send(_echo(200, {"answer": ANSWER, "stage_timings": {"router_ms": 1.0}}))
    assert rec["request_id"] == rec["server_request_id"] == f"{RUN}-r0007"
    assert (rec["ordinal"], rec["case_number"], rec["query_class"]) == (7, 22, "complex")
    assert rec["outcome"] == "ok" and rec["http_status"] == 200 and rec["schema_version"] == 1
    assert rec["stage_timings"] == {"router_ms": 1.0} and rec["body_excerpt"] is None
    assert rec["started_at"].endswith("Z")
    assert QUESTION not in json.dumps(rec) and ANSWER not in json.dumps(rec)


@pytest.mark.asyncio
async def test_http_500_record_keeps_error_code_and_sanitized_excerpt():
    rec = await _send(_echo(500, {"error": "query_failed", "request_id": "x", "k": "sk-abcdefgh"}))
    assert (rec["outcome"], rec["http_status"], rec["error_code"]) == ("http_error", 500,
                                                                       "query_failed")
    assert rec["body_kind"] == "json" and "sk-abcdefgh" not in rec["body_excerpt"]
    assert rec["stage_timings"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("exc,outcome", [
    (httpx.ReadTimeout("t"), "timeout"),
    (httpx.ConnectError("c"), "connection_error"),
    (httpx.RemoteProtocolError("p"), "protocol_error"),
])
async def test_client_failures_have_no_status_or_body(exc, outcome):
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc
    rec = await _send(handler)
    assert (rec["outcome"], rec["http_status"], rec["body_kind"]) == (outcome, None, "none")
    assert rec["client_exception_class"] == type(exc).__name__ and rec["elapsed_ms"] >= 0


def _write(run_dir, requests, server_lines):
    (run_dir / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in requests))
    (run_dir / "server.log").write_text("INFO: uvicorn started\n" + "".join(
        json.dumps({"event": "query_request", **line}) + "\n" for line in server_lines))


def _req(n: int, case: int, outcome: str, status: int | None, code: str | None = None) -> dict:
    return {"request_id": f"{RUN}-r{n:04d}", "case_number": case, "outcome": outcome,
            "http_status": status, "error_code": code, "stage_timings": None}


def test_summary_joins_server_log_and_counts_unmatched(tmp_path):
    _write(tmp_path, [
        _req(1, 22, "timeout", None), _req(2, 23, "http_error", 500, "query_failed"),
        _req(3, 1, "ok", 200),
    ], [
        {"request_id": f"{RUN}-r0001", "status": 200, "stage_timings": {"sql_generation_ms": 9.0}},
        {"request_id": f"{RUN}-r0002", "status": 500, "exception_class": "OperationalError"},
    ])
    summary = ev.summarize(tmp_path)
    assert summary["unmatched_records"] == 1 and not summary["evidence_complete"]
    assert summary["outcomes"]["http_error|500|query_failed"] == 1
    assert summary["http_500_exception_classes"] == {"OperationalError": 1}
    # A client timeout whose server line completed still contributes stage timings.
    assert summary["case_stage_timings"]["22"]["sql_generation_ms"]["p95"] == 9.0
    assert "router_ms" in summary["case_stage_timings"]["23"]["missing"]
    assert json.loads((tmp_path / "summary.json").read_text()) == summary


@pytest.mark.asyncio
async def test_main_writes_evidence_and_fails_without_server_log(tmp_path, monkeypatch):
    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(_echo(200, {"answer": ANSWER, "stage_timings": {}}))
    monkeypatch.setattr(ev, "LOG_ROOT", tmp_path)
    monkeypatch.setattr(load_test, "TOTAL_REQUESTS", 3)
    monkeypatch.setattr(load_test.httpx, "AsyncClient",
                        lambda **kw: real_client(transport=transport, **kw))
    assert await load_test.main(["--run-id", RUN]) == 1  # no server.log: evidence incomplete
    lines = (tmp_path / RUN / "requests.jsonl").read_text().splitlines()
    assert [json.loads(line)["ordinal"] for line in lines] == [1, 2, 3]
    assert json.loads((tmp_path / RUN / "summary.json").read_text())["unmatched_records"] == 3
