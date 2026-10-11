"""Median-of-N judge scoring for the DeepEval gate.

The judge is nondeterministic even at temperature 0, so every metric is
measured JUDGE_RUNS times and thresholds apply to the median (PLAN-007).
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Protocol

JUDGE_RUNS = 3


class JudgeMetric(Protocol):
    score: float | None
    threshold: float
    async_mode: bool

    def measure(self, test_case: Any) -> Any: ...


def median_scores(
    named_metrics: dict[str, JudgeMetric], test_case: Any, runs: int = JUDGE_RUNS
) -> tuple[dict[str, list[float | None]], dict[str, float | None], list[str]]:
    """Measure each metric `runs` times.

    Returns the raw scores per metric name, the median per name (None when any
    run was missing or NaN), and the names with a missing or NaN run.
    """
    scores: dict[str, list[float | None]] = {}
    for name, metric in named_metrics.items():
        # deepeval's default async_mode nests loop.run_until_complete() inside
        # pytest-asyncio's loop; its nest_asyncio patch deadlocks silently on
        # Python 3.14, so every metric is forced onto the pure-sync path.
        metric.async_mode = False
        scores[name] = []
        for _ in range(runs):
            metric.measure(test_case)
            scores[name].append(metric.score)

    invalid = [k for k, v in scores.items() if any(s is None or math.isnan(s) for s in v)]
    medians = {k: None if k in invalid else statistics.median(v) for k, v in scores.items()}
    return scores, medians, invalid


def below_threshold(
    named_metrics: dict[str, JudgeMetric], medians: dict[str, float | None]
) -> dict[str, tuple[float, float]]:
    """Return {name: (median, threshold)} for every metric whose median misses its threshold."""
    return {
        name: (median, metric.threshold)
        for name, metric in named_metrics.items()
        if (median := medians[name]) is not None and median < metric.threshold
    }
