from __future__ import annotations

from scripts.case1_classifier import classify

GOOD_A = {
    "chunk_found": True,
    "chunk_has_section_text": True,
    "truncated": False,
    "in_rerank_input": True,
    "excluded": False,
    "bottom_half": False,
    "cited": True,
}
GOOD_B = {"excluded": False}


def _runs(**overrides_a):
    a = [{**GOOD_A, **overrides_a} for _ in range(5)]
    b = [dict(GOOD_B) for _ in range(5)]
    return a, b


def test_wrong_run_count_is_inconclusive() -> None:
    cls, _ = classify([dict(GOOD_A)], [dict(GOOD_B)] * 5)
    assert cls == "inconclusive"


def test_missing_chunk_is_corpus_or_chunking() -> None:
    a, b = _runs(chunk_found=False)
    cls, evidence = classify(a, b)
    assert cls == "corpus_or_chunking"
    assert "missing" in evidence or "lacks" in evidence or "truncated" in evidence


def test_no_section_text_is_corpus_or_chunking() -> None:
    a, b = _runs(chunk_has_section_text=False)
    assert classify(a, b)[0] == "corpus_or_chunking"


def test_truncated_passage_is_corpus_or_chunking() -> None:
    a, b = _runs(truncated=True)
    assert classify(a, b)[0] == "corpus_or_chunking"


def test_absent_from_rerank_input_is_retrieval() -> None:
    a, b = _runs(in_rerank_input=False)
    cls, evidence = classify(a, b)
    assert cls == "retrieval"
    assert "rerank input" in evidence


def test_consistent_exclusion_is_reranking() -> None:
    a = [{**GOOD_A, "excluded": True, "bottom_half": True, "cited": False} for _ in range(5)]
    b = [{"excluded": True} for _ in range(5)]
    cls, evidence = classify(a, b)
    assert cls == "reranking"
    assert "threshold/top_n" in evidence


def test_split_citation_with_stable_mode_b_is_expansion_variance() -> None:
    a = [
        {**GOOD_A, "cited": True} if i % 2 == 0 else {**GOOD_A, "cited": False}
        for i in range(5)
    ]
    b = [dict(GOOD_B) for _ in range(5)]
    cls, evidence = classify(a, b)
    assert cls == "expansion_variance"
    assert "stable" in evidence


def test_all_invariants_hold_is_inconclusive() -> None:
    a, b = _runs()
    cls, evidence = classify(a, b)
    assert cls == "inconclusive"
    assert "no classification invariant matched" in evidence
