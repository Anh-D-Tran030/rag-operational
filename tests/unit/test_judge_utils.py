"""Offline tests for the DeepEval gate's median-of-N judge scoring (no live judge)."""
from __future__ import annotations

import math
from typing import Any

from eval._judge_utils import JUDGE_RUNS, below_threshold, median_scores


class StubMetric:
    """Metric double that returns preset scores, one per measure() call."""

    def __init__(self, scores: list[float | None], threshold: float = 0.5) -> None:
        self._scores = list(scores)
        self.threshold = threshold
        self.async_mode = True
        self.score: float | None = None
        self.calls = 0

    def measure(self, test_case: Any) -> None:
        self.score = self._scores[self.calls]
        self.calls += 1


def test_median_of_mixed_runs_is_one_and_passes() -> None:
    metrics = {"contextual_recall": StubMetric([0.0, 1.0, 1.0])}

    runs, medians, invalid = median_scores(metrics, test_case=None)

    assert runs == {"contextual_recall": [0.0, 1.0, 1.0]}
    assert medians == {"contextual_recall": 1.0}
    assert invalid == []
    assert below_threshold(metrics, medians) == {}


def test_none_run_is_invalid() -> None:
    metrics = {"faithfulness": StubMetric([1.0, None, 1.0])}

    _, medians, invalid = median_scores(metrics, test_case=None)

    assert invalid == ["faithfulness"]
    assert medians == {"faithfulness": None}


def test_nan_run_is_invalid() -> None:
    metrics = {
        "faithfulness": StubMetric([1.0, 1.0, math.nan]),
        "answer_relevancy": StubMetric([1.0] * 3),
    }

    _, medians, invalid = median_scores(metrics, test_case=None)

    assert invalid == ["faithfulness"]
    assert medians == {"faithfulness": None, "answer_relevancy": 1.0}


def test_each_metric_measured_exactly_judge_runs_times() -> None:
    metrics = {"faithfulness": StubMetric([1.0] * 5), "answer_relevancy": StubMetric([1.0] * 5)}

    median_scores(metrics, test_case=None)

    assert JUDGE_RUNS == 3
    assert [m.calls for m in metrics.values()] == [3, 3]


def test_below_threshold_reports_under_metric_name_regardless_of_order() -> None:
    low = StubMetric([0.2, 0.3, 0.9], threshold=0.75)
    high = StubMetric([1.0, 1.0, 1.0], threshold=0.9)
    for metrics in ({"answer_relevancy": low, "faithfulness": high},
                    {"faithfulness": high, "answer_relevancy": low}):
        low.calls = high.calls = 0
        _, medians, _ = median_scores(metrics, test_case=None)
        assert below_threshold(metrics, medians) == {"answer_relevancy": (0.3, 0.75)}


def test_async_mode_forced_off() -> None:
    metric = StubMetric([1.0] * 3)

    median_scores({"faithfulness": metric}, test_case=None)

    assert metric.async_mode is False
