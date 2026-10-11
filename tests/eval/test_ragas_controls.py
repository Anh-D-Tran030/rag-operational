"""SSRF and diskcache controls for the RAGAS runner (PLAN-003 §2.3).

Both advisories have no fixed release and are accepted risks conditional on
these controls holding: the runner must never touch the multi-modal
faithfulness metric (SSRF), and the judge LLM must never be constructed with
a cache backend (unsafe-pickle diskcache deserialization).
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest

EVAL_MODULES = ("ragas", "langchain_community", "datasets")
INSTALL_HINT = "install with `pip install -e .[eval]`"


def _check_eval_extra() -> None:
    missing = [m for m in EVAL_MODULES if importlib.util.find_spec(m) is None]
    if not missing:
        return
    msg = f"missing eval packages {missing} ({INSTALL_HINT})"
    if os.environ.get("RAG_EVAL_REQUIRED") == "1":
        pytest.fail(msg)
    pytest.skip(msg)


def test_ragas_runner_never_imports_multimodal_ssrf_metric() -> None:
    """GHSA-95ww-475f-pr4f / PYSEC-2026-3046: the runner uses only text metrics,
    so the vulnerable multi-modal faithfulness module must never load."""
    _check_eval_extra()
    from ragas.metrics import (  # noqa: F401
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )
    from ragas.metrics._answer_relevance import AnswerRelevancy  # noqa: F401

    import src.evaluation.ragas_runner  # noqa: F401  (module import only; metrics import lazily)

    assert "ragas.metrics.collections.multi_modal_faithfulness" not in sys.modules


def test_ragas_judge_has_no_cache_backend() -> None:
    """GHSA-w8v5-vhqr-4h9v / PYSEC-2026-2447: the judge LLM built by
    scripts.run_eval._build_judge must never be given a cache backend — no
    diskcache instance is created, and evaluate() is never handed one."""
    _check_eval_extra()
    sys.path.insert(0, ".")
    from scripts.run_eval import _build_judge
    from src.config import Settings

    settings = Settings(openrouter_api_key="test-key", embedding_model="all-MiniLM-L6-v2")
    judge_llm, _judge_embeddings = _build_judge(settings)
    assert judge_llm.cache is None
    assert "diskcache" not in sys.modules
