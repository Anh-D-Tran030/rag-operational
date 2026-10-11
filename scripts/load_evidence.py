"""Load-test evidence (PLAN-003 section 4): sanitized per-request records + server-log join.

Every request writes one line to .logs/loadtest/<load_run_id>/requests.jsonl. TEST
redirects the API's stderr to server.log in the same directory; `summarize` joins the
two by request_id so every HTTP 500 and timeout is diagnosable from files alone.
Re-run after the server drains:  .venv/bin/python scripts/load_evidence.py <run_dir>
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 1
LOG_ROOT = Path(__file__).resolve().parent.parent / ".logs" / "loadtest"
RUN_ID = re.compile(r"^load-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{6}$")
EXCERPT_CHARS = 512
PROFILED_CASES = (22, 23)
EXPECTED_STAGES = (
    "router_ms", "retrieval_ms", "reranking_ms", "schema_linking_ms", "sql_generation_ms",
    "sql_validation_ms", "sql_execution_ms", "final_generation_ms", "faithfulness_ms",
    "unattributed_ms",
)
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]+")
_REDACTIONS = (  # order matters: URL userinfo before the e-mail rule
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"), "Bearer [REDACTED]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{6,}"), "[REDACTED]"),
    (re.compile(r"(?i)(\b(?:authorization|api[_-]?key|token|secret|password)[\"']?\s*[:=]\s*)"
                r"(\"[^\"]*\"|'[^']*'|[^\s,;&}]+)"), r"\1[REDACTED]"),
    (re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s@]+@"), r"\1[REDACTED]@"),
    (re.compile(r"(?i)(\b[a-z][a-z0-9+.-]*://[^\s?#\"']*)\?[^\s#\"']*"), r"\1?[REDACTED]"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[EMAIL]"),
)


def new_run_id(override: str | None = None) -> str:
    if override is not None:
        if not RUN_ID.fullmatch(override):
            raise ValueError(f"--run-id must match {RUN_ID.pattern}")
        return override
    return f"load-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{secrets.token_hex(3)}"


def utc_ms(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sanitize(raw: bytes | str) -> str:
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    text = _CONTROL.sub(" ", text)
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text[:EXCERPT_CHARS]


def body_fields(content: bytes | None, content_type: str | None, excerpt: bool) -> dict:
    """Describe a response body. Excerpts are for error bodies only: 2xx carries answers."""
    if content is None:
        kind, parsed = "none", None
    elif not content:
        kind, parsed = "empty", None
    else:
        try:
            kind, parsed = "json", json.loads(content)
        except ValueError:
            kind, parsed = "non_json", None
    detail = parsed.get("detail") if isinstance(parsed, dict) else None
    code = parsed.get("error") if isinstance(parsed, dict) else None
    if code is None and isinstance(detail, dict):
        code = detail.get("error")
    return {
        "content_type": content_type, "body_kind": kind,
        "body_bytes": None if content is None else len(content),
        "body_sha256": None if content is None else hashlib.sha256(content).hexdigest(),
        "body_excerpt": sanitize(content) if excerpt and content else None,
        "error_code": code if isinstance(code, str) else None,
        "parsed": parsed,
    }


def pct(values: list[float], p: int) -> float:
    """Nearest-rank percentile; NaN for an empty sample so it can never pass an SLO."""
    if not values:
        return math.nan
    s = sorted(values)
    return s[max(0, math.ceil(len(s) * p / 100) - 1)]


def _read_jsonl(path: Path, event: str | None = None) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines() if path.exists() else []:
        try:
            row = json.loads(line)
        except ValueError:
            continue  # uvicorn access/startup lines share the stderr stream
        if isinstance(row, dict) and (event is None or row.get("event") == event):
            rows.append(row)
    return rows


def summarize(run_dir: Path) -> dict:
    """Join requests.jsonl to server.log by request_id and write summary.json."""
    requests = _read_jsonl(run_dir / "requests.jsonl")
    server: dict[str, list[dict]] = {}
    for line in _read_jsonl(run_dir / "server.log", "query_request"):
        server.setdefault(str(line.get("request_id")), []).append(line)
    unmatched = [r["request_id"] for r in requests if len(server.get(r["request_id"], [])) != 1]
    outcomes = Counter(f"{r['outcome']}|{r['http_status']}|{r['error_code']}" for r in requests)
    classes = Counter(
        str(line.get("exception_class")) for lines in server.values() for line in lines
        if line.get("status") == 500
    )
    stages: dict[str, dict] = {}
    for case in PROFILED_CASES:
        rows = [
            (server.get(r["request_id"]) or [{}])[0].get("stage_timings") or r["stage_timings"]
            or {} for r in requests if r["case_number"] == case
        ]
        stages[str(case)] = {
            "requests": len(rows),
            "missing": sorted({k for row in rows or [{}] for k in EXPECTED_STAGES if k not in row}),
            **{k: {"p50": pct(v, 50), "p95": pct(v, 95), "n": len(v)} for k in EXPECTED_STAGES
               if (v := [float(row[k]) for row in rows if k in row])},
        }
    summary = {
        "schema_version": SCHEMA_VERSION, "run_dir": str(run_dir), "requests": len(requests),
        "outcomes": dict(outcomes), "http_500_exception_classes": dict(classes),
        "unmatched_records": len(unmatched), "unmatched_request_ids": unmatched[:50],
        "evidence_complete": bool(requests) and not unmatched, "case_stage_timings": stages,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    result = summarize(Path(sys.argv[1]))
    print(json.dumps({k: result[k] for k in ("outcomes", "unmatched_records")}))
    sys.exit(0 if result["evidence_complete"] else 1)
