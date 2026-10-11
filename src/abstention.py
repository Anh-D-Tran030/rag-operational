"""Shared recognition of safe no-answer responses at system boundaries."""
from __future__ import annotations

import re

ABSTENTION_MARKERS = (
    "insufficient context",
    "not available in the provided corpus",
    "could not be verified",
    "cannot answer",
    "don't have enough",
    "do not have enough",
    # Wordings observed in the live unanswerable evaluation.
    "not provided in the given context",
    "does not contain information about",
    "does not contain any information about",
    "does not include information about",
    "does not include any information about",
    "no information provided in the given context",
    "no information provided about",
    "cannot find any information about",
    "there is no information about",
)

_UNABLE_TO_PROVIDE_MISSING_CONTEXT = re.compile(
    r"\bi[\u2019']m unable to provide\b.*\binformation is not (?:included|present)\b",
    re.IGNORECASE | re.DOTALL,
)

# Keep these patterns specific to a missing-context claim.  A bare phrase such
# as "no information about X" is not enough: a supported answer may use it
# while contrasting two sources.
_NO_INFORMATION_GIVEN_CONTEXT = re.compile(
    r"\bno information about\b.*\bis provided in the given context\b",
    re.IGNORECASE | re.DOTALL,
)
_CANNOT_FIND_PROVIDED_CONTEXT = re.compile(
    r"\bi cannot find information about\b.*\bin the provided context\b",
    re.IGNORECASE | re.DOTALL,
)
_CANNOT_PROVIDE_NOT_MENTIONED_CONTEXT = re.compile(
    r"\b(?:i cannot|i[\u2019']m unable to)\s+provide\b.*\bnot mentioned in the provided context\b",
    re.IGNORECASE | re.DOTALL,
)
_NOT_FOUND_PROVIDED_CONTEXT = re.compile(
    r"\bno information about\b.*\bwas found in the provided context\b",
    re.IGNORECASE | re.DOTALL,
)
# Structural refusal template: "<the context> does not include/contain <anything>".
# The object varies with every question ("information about", "the price of", ...),
# so anchoring on the context-noun subject and the negated coverage verb catches
# each new wording without another literal. A non-context subject ("The procedure
# does not include ...") is a supported fact, not a refusal, and does not match.
_CONTEXT_DOES_NOT_COVER = re.compile(
    r"\b(?:the\s+)?(?:(?:provided|given|retrieved|available|supplied)\s+)?"
    r"(?:context|documents?|sources?|records?|evidence|information|excerpts?|passages?|data)\s+"
    r"(?:(?:does|do)\s+not|doesn[’']t|don[’']t)\s+"
    r"(?:include|contain|mention|specify|state|provide|list|cover|say|give)\b",
    re.IGNORECASE,
)


def is_abstention_answer(answer: str) -> bool:
    """Return whether an answer explicitly declines for lack of context."""
    lowered = answer.casefold().strip()
    return any(marker in lowered for marker in ABSTENTION_MARKERS) or bool(
        _UNABLE_TO_PROVIDE_MISSING_CONTEXT.search(answer)
        or _NO_INFORMATION_GIVEN_CONTEXT.search(answer)
        or _CANNOT_FIND_PROVIDED_CONTEXT.search(answer)
        or _CANNOT_PROVIDE_NOT_MENTIONED_CONTEXT.search(answer)
        or _NOT_FOUND_PROVIDED_CONTEXT.search(answer)
        or _CONTEXT_DOES_NOT_COVER.search(answer)
    )
