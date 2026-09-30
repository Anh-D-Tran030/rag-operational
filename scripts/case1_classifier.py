"""Pure classification rule for the case-1 diagnostic (PLAN-003 §5).

Takes the per-run records `diagnose_case1.py` assembles from 5 mode-A
(production, with expansion) and 5 mode-B (original question, no expansion)
runs, and assigns exactly one class. The first rule that holds wins.
"""
from __future__ import annotations

from typing import TypedDict

CLASSES = (
    "corpus_or_chunking",
    "retrieval",
    "reranking",
    "expansion_variance",
    "inconclusive",
)


class ModeARun(TypedDict):
    chunk_found: bool
    chunk_has_section_text: bool
    truncated: bool
    in_rerank_input: bool
    excluded: bool
    bottom_half: bool
    cited: bool


class ModeBRun(TypedDict):
    excluded: bool


def classify(
    mode_a_runs: list[ModeARun], mode_b_runs: list[ModeBRun]
) -> tuple[str, str]:
    """Return (class_name, evidence) — evidence names the rule and the facts."""
    if len(mode_a_runs) != 5 or len(mode_b_runs) != 5:
        return "inconclusive", (
            f"expected 5 mode-A and 5 mode-B runs, got {len(mode_a_runs)}/{len(mode_b_runs)}"
        )

    bad_corpus = [
        i
        for i, r in enumerate(mode_a_runs)
        if not r["chunk_found"] or not r["chunk_has_section_text"] or r["truncated"]
    ]
    if bad_corpus:
        return "corpus_or_chunking", (
            f"run(s) {bad_corpus}: gold anchor missing from a chunk, chunk lacks the "
            "section 2.2-2.4 text, or the passage is truncated at max_length"
        )

    missing_from_rerank = [i for i, r in enumerate(mode_a_runs) if not r["in_rerank_input"]]
    if missing_from_rerank:
        return "retrieval", (
            f"gold absent from the rerank input in run(s) {missing_from_rerank} of 5 mode-A runs"
        )

    excluded_count = sum(1 for r in mode_a_runs if r["excluded"])
    bottom_half_count = sum(1 for r in mode_a_runs if r["bottom_half"])
    mode_b_excluded_count = sum(1 for r in mode_b_runs if r["excluded"])
    if excluded_count >= 4 and bottom_half_count >= 4 and mode_b_excluded_count == 5:
        return "reranking", (
            f"gold present in all 5 mode-A rerank inputs but excluded by threshold/top_n in "
            f"{excluded_count}/5, bottom-half FlashRank rank in {bottom_half_count}/5, and mode B "
            "reproduces the exclusion 5/5 times"
        )

    cited_flags = {r["cited"] for r in mode_a_runs}
    mode_b_stable = len({r["excluded"] for r in mode_b_runs}) == 1
    if len(cited_flags) > 1 and mode_b_stable:
        return "expansion_variance", (
            "gold is cited in some mode-A runs and not others while mode B is stable"
        )

    return "inconclusive", (
        "no classification invariant matched: corpus/chunking is sound, gold reaches every "
        "mode-A rerank input, exclusion is not consistent enough across mode A/B for "
        "'reranking', and citation is not split against a stable mode B for 'expansion_variance'"
    )
