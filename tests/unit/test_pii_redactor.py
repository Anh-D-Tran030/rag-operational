"""Unit tests for PII redactor — no real Presidio models loaded."""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def _make_analyzer_result(entity_type: str = "PERSON"):
    result = MagicMock()
    result.entity_type = entity_type
    return result


def _make_anonymizer_result(text: str):
    result = MagicMock()
    result.text = text
    return result


def test_staff_id_redacted():
    with (
        patch("src.guardrails.pii_redactor._get_engines") as mock_engines,
        patch("src.guardrails.pii_redactor.rag_pii_redactions") as _counter,
    ):
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = []
        mock_anonymizer = MagicMock()
        mock_engines.return_value = (mock_analyzer, mock_anonymizer)

        from src.guardrails.pii_redactor import redact_pii

        result = redact_pii("Contact STF-1234 for details")
        assert "<REDACTED_STAFF_ID>" in result


def test_no_pii_unchanged():
    with (
        patch("src.guardrails.pii_redactor._get_engines") as mock_engines,
        patch("src.guardrails.pii_redactor.rag_pii_redactions") as _counter,
    ):
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = []
        mock_anonymizer = MagicMock()
        mock_engines.return_value = (mock_analyzer, mock_anonymizer)

        from src.guardrails.pii_redactor import redact_pii

        result = redact_pii("The venue opened at 9pm")
        assert result == "The venue opened at 9pm"


def test_label_forwarded():
    with (
        patch("src.guardrails.pii_redactor._get_engines") as mock_engines,
        patch("src.guardrails.pii_redactor.rag_pii_redactions") as mock_counter,
    ):
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = [_make_analyzer_result("PERSON")]
        mock_anonymizer = MagicMock()
        mock_anonymizer.anonymize.return_value = _make_anonymizer_result("Hello <PERSON>")
        mock_engines.return_value = (mock_analyzer, mock_anonymizer)

        from src.guardrails.pii_redactor import redact_pii

        redact_pii("Hello Alice", label="output")
        mock_counter.labels.assert_called_with(label="output")


def test_person_entity_redacted():
    with (
        patch("src.guardrails.pii_redactor._get_engines") as mock_engines,
        patch("src.guardrails.pii_redactor.rag_pii_redactions") as _counter,
    ):
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = [_make_analyzer_result("PERSON")]
        mock_anonymizer = MagicMock()
        mock_anonymizer.anonymize.return_value = _make_anonymizer_result("Hello <PERSON>")
        mock_engines.return_value = (mock_analyzer, mock_anonymizer)

        from src.guardrails.pii_redactor import redact_pii

        result = redact_pii("Hello Alice")
        assert result == "Hello <PERSON>"
