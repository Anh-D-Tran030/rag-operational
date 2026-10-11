"""Offline controls for the live DeepEval gate."""
from __future__ import annotations

from types import SimpleNamespace

from ._judge_utils import below_threshold
from .test_deepeval_gate import TRACKED_ONLY


def test_low_contextual_recall_does_not_block_gate() -> None:
    metrics = {
        "faithfulness": SimpleNamespace(threshold=0.9),
        "answer_relevancy": SimpleNamespace(threshold=0.75),
        "contextual_precision": SimpleNamespace(threshold=0.5),
        "contextual_recall": SimpleNamespace(threshold=0.5),
    }
    medians = {
        "faithfulness": 1.0,
        "answer_relevancy": 1.0,
        "contextual_precision": 1.0,
        "contextual_recall": 0.1,
    }

    blocking = {name: metric for name, metric in metrics.items() if name not in TRACKED_ONLY}
    failed = below_threshold(blocking, medians)

    assert "contextual_recall" not in blocking
    assert not failed
