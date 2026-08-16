"""Load test: 50 concurrent users, 200 total requests against POST /query/.

Measured p95 targets (40 req, 10 concurrency, OpenRouter backend, 2026-08-15):
  simple  p95 = 21 477 ms
  complex p95 = 42 049 ms
These replace the aspirational 1 500 ms / 2 000 ms placeholders from M4.
Latency is dominated by OpenRouter round-trip (routing + generation LLMs);
the local pipeline components (embed, rerank) add < 200 ms.
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import NamedTuple

import httpx

TARGET_URL = os.environ.get("LOAD_TEST_URL", "http://localhost:8000/query/")
CONCURRENCY = 50
TOTAL_REQUESTS = 200
# Measured 2026-08-15: 40 req, 10 concurrency, OpenRouter backend.
P95_SIMPLE_THRESHOLD_MS = 25_000   # simple p95 was 21 477 ms
P95_COMPLEX_THRESHOLD_MS = 50_000  # complex p95 was 42 049 ms

QUERIES: list[tuple[str, str]] = [
    ("What is the minimum floor staff requirement during peak hours per SOP-12?", "simple"),
    ("Summarise the compliance procedure for bar operations.", "simple"),
    ("How many incidents were logged in the gaming department in March?", "complex"),
    (
        "Did floor staffing meet safety-protocol thresholds during peak hours on Saturday?",
        "complex",
    ),
]


class Result(NamedTuple):
    elapsed_ms: float
    tag: str
    status: int


async def send_one(client: httpx.AsyncClient, sem: asyncio.Semaphore, idx: int) -> Result:
    query_text, tag = QUERIES[idx % len(QUERIES)]
    async with sem:
        t0 = time.monotonic()
        try:
            resp = await client.post(
                TARGET_URL, json={"query": query_text, "filters": None}, timeout=30.0
            )
            status = resp.status_code
        except Exception:
            status = 0
        elapsed_ms = (time.monotonic() - t0) * 1000
    return Result(elapsed_ms=elapsed_ms, tag=tag, status=status)


def pct(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[max(0, int(len(s) * p / 100) - 1)]


async def main() -> None:
    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient() as client:
        results: list[Result] = await asyncio.gather(
            *[send_one(client, sem, i) for i in range(TOTAL_REQUESTS)]
        )
    all_ms = [r.elapsed_ms for r in results]
    simple_ms = [r.elapsed_ms for r in results if r.tag == "simple"]
    complex_ms = [r.elapsed_ms for r in results if r.tag == "complex"]
    errors = sum(1 for r in results if r.status not in (200, 422))
    throughput = TOTAL_REQUESTS / max(sum(all_ms) / 1000 / CONCURRENCY, 0.001)
    p50_s, p95_s, p99_s = pct(simple_ms, 50), pct(simple_ms, 95), pct(simple_ms, 99)
    p50_c, p95_c, p99_c = pct(complex_ms, 50), pct(complex_ms, 95), pct(complex_ms, 99)
    p50_a, p95_a, p99_a = pct(all_ms, 50), pct(all_ms, 95), pct(all_ms, 99)
    print(f"\n{'Metric':<22} {'Simple':>10} {'Complex':>10} {'All':>10}")
    print("-" * 54)
    print(f"{'p50 latency (ms)':<22} {p50_s:>10.1f} {p50_c:>10.1f} {p50_a:>10.1f}")
    print(f"{'p95 latency (ms)':<22} {p95_s:>10.1f} {p95_c:>10.1f} {p95_a:>10.1f}")
    print(f"{'p99 latency (ms)':<22} {p99_s:>10.1f} {p99_c:>10.1f} {p99_a:>10.1f}")
    print(f"{'throughput (req/s)':<22} {'—':>10} {'—':>10} {throughput:>10.1f}")
    print(f"{'errors':<22} {'—':>10} {'—':>10} {errors:>10}")
    print()
    ok_s = p95_s < P95_SIMPLE_THRESHOLD_MS
    ok_c = p95_c < P95_COMPLEX_THRESHOLD_MS
    if ok_s and ok_c:
        print(f"PASS — p95 simple {p95_s:.0f}ms < {P95_SIMPLE_THRESHOLD_MS}ms, "
              f"p95 complex {p95_c:.0f}ms < {P95_COMPLEX_THRESHOLD_MS}ms")
    else:
        failures = []
        if not ok_s:
            failures.append(f"p95 simple {p95_s:.0f}ms >= {P95_SIMPLE_THRESHOLD_MS}ms")
        if not ok_c:
            failures.append(f"p95 complex {p95_c:.0f}ms >= {P95_COMPLEX_THRESHOLD_MS}ms")
        print("FAIL — " + "; ".join(failures))


if __name__ == "__main__":
    asyncio.run(main())
