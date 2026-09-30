"""Load test: 50 concurrent users, 200 total requests against POST /query/.

Measured p95 targets (40 req, 10 concurrency, OpenRouter backend, 2026-08-15):
  simple  p95 = 21 477 ms
  complex p95 = 42 049 ms
These replace the aspirational 1 500 ms / 2 000 ms placeholders from M4.
Questions come from eval_data/golden_dataset.json: simple_lookup cases are
"simple", multi_hop (hybrid) cases are "complex". Any timeout, transport error,
or non-2xx response is an error; exit 1 on any error, SLO miss, or incomplete
evidence. Schedule: golden order, round-robin, no seed. Evidence goes to
.logs/loadtest/<load_run_id>/ (load_evidence.py); run the API with stderr in server.log.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from scripts import load_evidence as ev

TARGET_URL = os.environ.get("LOAD_TEST_URL", "http://localhost:8000/query/")
GOLDEN_PATH = Path(__file__).resolve().parent.parent / "eval_data" / "golden_dataset.json"
CONCURRENCY = 50
TOTAL_REQUESTS = 200
# Measured 2026-08-15: 40 req, 10 concurrency, OpenRouter backend.
P95_SIMPLE_THRESHOLD_MS = 25_000   # simple p95 was 21 477 ms
P95_COMPLEX_THRESHOLD_MS = 50_000  # complex p95 was 42 049 ms
# Must exceed the complex SLO so slow-but-successful requests are measured, not cut off.
HTTP_TIMEOUT_S = float(os.environ.get("LOAD_TEST_TIMEOUT_S", "90"))
TAGS_BY_CATEGORY = {"simple_lookup": "simple", "multi_hop": "complex"}


def load_queries() -> list[tuple[int, str, str]]:
    """Return (1-based case number, question, tag) for simple and complex golden cases."""
    cases = json.loads(GOLDEN_PATH.read_text())
    return [
        (i, c["question"], TAGS_BY_CATEGORY[c["category"]])
        for i, c in enumerate(cases, start=1) if c["category"] in TAGS_BY_CATEGORY
    ]


async def send_one(
    client: httpx.AsyncClient, sem: asyncio.Semaphore, run_id: str, ordinal: int,
    query: tuple[int, str, str],
) -> dict:
    """Send one request and return its evidence record (never the query or answer text)."""
    case, text, tag = query
    request_id = f"{run_id}-r{ordinal:04d}"
    resp, outcome, exc_class = None, "ok", None
    async with sem:
        started, t0 = datetime.now(UTC), time.monotonic()
        try:
            resp = await client.post(
                TARGET_URL, json={"query": text, "filters": None},
                headers={"X-Request-ID": request_id},
            )
        except httpx.TimeoutException as exc:
            outcome, exc_class = "timeout", type(exc).__name__
        except (httpx.NetworkError, httpx.ProxyError) as exc:
            outcome, exc_class = "connection_error", type(exc).__name__
        except httpx.HTTPError as exc:
            outcome, exc_class = "protocol_error", type(exc).__name__
        elapsed_ms = (time.monotonic() - t0) * 1000
    failed = resp is not None and not resp.is_success
    body = ev.body_fields(resp.content if resp is not None else None,
                          resp.headers.get("content-type") if resp is not None else None, failed)
    parsed = body.pop("parsed")
    if failed:
        outcome = "http_error"
    elif resp is not None and not isinstance(parsed, dict):
        outcome, exc_class = "protocol_error", None
    return {
        "schema_version": ev.SCHEMA_VERSION, "load_run_id": run_id, "request_id": request_id,
        "ordinal": ordinal, "case_number": case, "query_class": tag,
        "started_at": ev.utc_ms(started), "elapsed_ms": round(elapsed_ms, 2), "outcome": outcome,
        "http_status": resp.status_code if resp is not None else None,
        "server_request_id": resp.headers.get("x-request-id") if resp is not None else None,
        **body, "client_exception_class": exc_class,
        "stage_timings": parsed.get("stage_timings") if outcome == "ok" else None,
    }


def report(summary: dict, run_dir: Path) -> None:
    print(f"\nevidence: {run_dir}\n  outcome|http_status|error_code: {summary['outcomes']}")
    print(f"  HTTP 500 by exception_class: {summary['http_500_exception_classes'] or 'none'}")
    print(f"  records without exactly one server line: {summary['unmatched_records']}")
    for case, stages in summary["case_stage_timings"].items():
        print(f"\nCase {case} stage timings over {stages['requests']} requests (p50 / p95 ms)")
        for key, v in stages.items():
            if isinstance(v, dict):
                print(f"  {key:<22} {v['p50']:>10.1f} {v['p95']:>10.1f}")
        print(f"  missing timing fields: {', '.join(stages['missing']) or 'none'}")


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=ev.new_run_id, help="load-<UTC>Z-<6 hex>")
    run_id = parser.parse_args(argv).run_id or ev.new_run_id()
    run_dir = ev.LOG_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    queries = load_queries()
    sem = asyncio.Semaphore(CONCURRENCY)
    wall_t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
        # Ordinals are fixed in submission order before any request is dispatched.
        results: list[dict] = await asyncio.gather(*[
            send_one(client, sem, run_id, n, queries[(n - 1) % len(queries)])
            for n in range(1, TOTAL_REQUESTS + 1)
        ])
    wall_s = time.monotonic() - wall_t0
    (run_dir / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    ok = [r for r in results if r["outcome"] == "ok"]
    errors = len(results) - len(ok)
    cols = {tag: [r["elapsed_ms"] for r in ok if r["query_class"] == tag]
            for tag in ("simple", "complex")}
    cols["all"] = [r["elapsed_ms"] for r in ok]
    print(f"\n{'Metric':<22} {'Simple':>10} {'Complex':>10} {'All':>10}")
    print("-" * 54)
    for p in (50, 95, 99):
        cells = " ".join(f"{ev.pct(v, p):>10.1f}" for v in cols.values())
        print(f"{f'p{p} latency (ms)':<22} {cells}")
    print(f"{'throughput (req/s)':<22} {'—':>10} {'—':>10} {TOTAL_REQUESTS / wall_s:>10.2f}")
    print(f"{'errors':<22} {'—':>10} {'—':>10} {errors:>10}")
    summary = ev.summarize(run_dir)
    report(summary, run_dir)
    p95_s, p95_c = ev.pct(cols["simple"], 95), ev.pct(cols["complex"], 95)
    failures = [f"{errors} request errors"] if errors else []
    if not p95_s < P95_SIMPLE_THRESHOLD_MS:
        failures.append(f"p95 simple {p95_s:.0f}ms >= {P95_SIMPLE_THRESHOLD_MS}ms")
    if not p95_c < P95_COMPLEX_THRESHOLD_MS:
        failures.append(f"p95 complex {p95_c:.0f}ms >= {P95_COMPLEX_THRESHOLD_MS}ms")
    if not summary["evidence_complete"]:
        failures.append(f"evidence incomplete: {summary['unmatched_records']} unmatched")
    if failures:
        print("\nFAIL — " + "; ".join(failures))
        return 1
    print(f"\nPASS — p95 simple {p95_s:.0f}ms < {P95_SIMPLE_THRESHOLD_MS}ms, "
          f"p95 complex {p95_c:.0f}ms < {P95_COMPLEX_THRESHOLD_MS}ms, 0 errors")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
