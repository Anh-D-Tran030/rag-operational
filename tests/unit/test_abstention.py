"""Offline tests for the shared abstention recognizer."""
from __future__ import annotations

import pytest

from src.abstention import is_abstention_answer


@pytest.mark.parametrize("answer", [
    # Golden case 23: the wording that the literal-marker list missed.
    "The provided context does not include the price of a schooner of beer at Fairfield RSL.",
    "The documents do not mention the opening hours.",
    "The given context doesn't specify a dress code.",
    "insufficient context",
])
def test_context_refusal_template_is_an_abstention(answer: str) -> None:
    assert is_abstention_answer(answer)


@pytest.mark.parametrize("answer", [
    "The procedure does not include a 60-minute window; it requires 30 minutes.",
    "Fairfield RSL does not include gaming in its bistro.",
    "SH-006 had 20 staff.",
])
def test_non_context_subject_is_not_an_abstention(answer: str) -> None:
    assert not is_abstention_answer(answer)
