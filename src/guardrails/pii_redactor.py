"""PII redaction using Presidio NER + regex for staff IDs."""
from __future__ import annotations

import os
import re
import unicodedata
from collections.abc import Sequence
from typing import Any

from src.observability.metrics import rag_pii_redactions

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine

    _PRESIDIO_AVAILABLE = True
except ImportError:
    _PRESIDIO_AVAILABLE = False

_analyzer: Any = None
_anonymizer: Any = None


def _get_engines() -> tuple[Any, Any] | tuple[None, None]:
    """Lazy-initialise Presidio engines (singleton).

    Returns (None, None) if Presidio is not installed.
    """
    global _analyzer, _anonymizer
    if not _PRESIDIO_AVAILABLE:
        return None, None
    if _analyzer is None:
        _analyzer = AnalyzerEngine()
    if _anonymizer is None:
        _anonymizer = AnonymizerEngine()
    return _analyzer, _anonymizer


_STAFF_ID_RE = re.compile(r"\bSTF-\d{4,6}\b", re.IGNORECASE)

# Restrict redaction to entities that are actually personal. Presidio's default is
# every recognizer it has, which in operational text means venue names, suburbs and
# shift dates all disappear — the answer stops being useful without protecting anyone.
_ENTITIES = tuple(
    e.strip()
    for e in os.environ.get(
        "PII_ENTITIES",
        "PERSON,EMAIL_ADDRESS,PHONE_NUMBER,CREDIT_CARD,IBAN_CODE,AU_TFN,AU_MEDICARE",
    ).split(",")
    if e.strip()
)

# Domain nouns the NER model mislabels as people. "Fairfield RSL" was being scored
# as PERSON and redacted out of every answer that named the venue.
_ALLOW_LIST = tuple(
    t.strip()
    # Both spellings: prose says "Fairfield RSL", chunk metadata says "Fairfield_RSL".
    for t in os.environ.get(
        "PII_ALLOW_LIST", "Fairfield RSL,Fairfield_RSL,Fairfield,RSL"
    ).split(",")
    if t.strip()
)


def redact_pii(text: str, label: str = "input", evidence: Sequence[str] = ()) -> str:
    """Redact PII from text using Presidio NER + regex for staff IDs.

    Args:
        text: Input string to redact.
        label: "input" or "output" — used as the Prometheus label.
        evidence: Queried SQL row values and retrieved chunk text. A detected
            entity whose exact span occurs verbatim in any of them was copied from
            the evidence, not generated, so it is answer content and is kept. NER
            is not consistent enough to decide this by itself: it masked one
            supervisor name and kept another in the same sentence. Staff IDs are
            always redacted.

    Returns:
        String with detected PII replaced by <REDACTED_ENTITY_TYPE> tokens.
        Increments rag_pii_redactions{label=label} by the number of
        entities found. Returns text unchanged if no entities found.
        If Presidio is not installed, only staff-ID regex redaction runs.
    """
    matches = _STAFF_ID_RE.findall(text)
    if matches:
        text = _STAFF_ID_RE.sub("<REDACTED_STAFF_ID>", text)
        rag_pii_redactions.labels(label=label).inc(len(matches))

    analyzer, anonymizer = _get_engines()
    if analyzer is None:
        return text
    results = analyzer.analyze(
        text=text,
        language="en",
        entities=list(_ENTITIES),
        allow_list=list(_ALLOW_LIST),
    )
    if evidence:
        normalized = [_typographic(item) for item in evidence]
        results = [r for r in results if not _in_evidence(text[r.start:r.end], normalized)]
    if results:
        anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
        rag_pii_redactions.labels(label=label).inc(len(results))
        return anonymized.text
    return text


def _typographic(text: str) -> str:
    """Fold rendering-only differences: generators write "S. Lee" for "S. Lee"."""
    return "".join(
        " " if unicodedata.category(ch) == "Zs" else "-" if unicodedata.category(ch) == "Pd"
        else ch
        for ch in unicodedata.normalize("NFKC", text)
    )


def _in_evidence(span: str, normalized_evidence: Sequence[str]) -> bool:
    """Return whether a detected span was copied from the (normalized) evidence."""
    span = _typographic(span)
    return bool(span) and any(span in item for item in normalized_evidence)
