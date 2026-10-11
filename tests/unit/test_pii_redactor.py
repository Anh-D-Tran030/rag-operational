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


def _span_result(text: str, span: str, entity_type: str = "PERSON"):
    result = _make_analyzer_result(entity_type)
    result.start = text.index(span)
    result.end = result.start + len(span)
    return result


_CASE_19 = "No. It was supervised by S. Lee, while SH-001 was supervised by J. Thompson."
_CASE_19_ROW = "{'incident_supervisor': 'S. Lee', 'comparison_supervisor': 'J. Thompson'}"


def _redact_with_names(text: str, names: list[str], evidence: list[str]):
    """Run redact_pii with a fake analyzer flagging `names`; return (output, redacted)."""
    with (
        patch("src.guardrails.pii_redactor._get_engines") as mock_engines,
        patch("src.guardrails.pii_redactor.rag_pii_redactions"),
    ):
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = [_span_result(text, n) for n in names]
        mock_anonymizer = MagicMock()

        def anonymize(text, analyzer_results):
            for r in sorted(analyzer_results, key=lambda r: r.start, reverse=True):
                text = text[:r.start] + "<PERSON>" + text[r.end:]
            return _make_anonymizer_result(text)

        mock_anonymizer.anonymize.side_effect = anonymize
        mock_engines.return_value = (mock_analyzer, mock_anonymizer)

        from src.guardrails.pii_redactor import redact_pii

        output = redact_pii(text, label="output", evidence=evidence)
        calls = mock_anonymizer.anonymize.call_args_list
        redacted = [text[r.start:r.end] for c in calls for r in c.kwargs["analyzer_results"]]
        return output, redacted


def test_names_copied_from_evidence_rows_are_kept():
    output, redacted = _redact_with_names(_CASE_19, ["S. Lee", "J. Thompson"], [_CASE_19_ROW])
    assert output == _CASE_19
    assert redacted == []


def test_name_absent_from_evidence_is_still_redacted():
    text = "S. Lee was on duty with A. Nguyen."
    output, redacted = _redact_with_names(text, ["S. Lee", "A. Nguyen"], [_CASE_19_ROW])
    assert output == "S. Lee was on duty with <PERSON>."
    assert redacted == ["A. Nguyen"]


def test_staff_id_redacted_even_when_in_evidence():
    output, _ = _redact_with_names("Rostered: STF-1234.", [], ["{'staff_id': 'STF-1234'}"])
    assert output == "Rostered: <REDACTED_STAFF_ID>."


def test_no_evidence_redacts_every_detected_name():
    output, redacted = _redact_with_names(_CASE_19, ["S. Lee", "J. Thompson"], [])
    assert "S. Lee" not in output and "J. Thompson" not in output
    assert sorted(redacted) == ["J. Thompson", "S. Lee"]


def test_evidence_match_ignores_typographic_spacing():
    # Golden 19 live output: the generator wrote a narrow no-break space (U+202F).
    text = "Supervised by S. Lee and A. Nguyen."
    output, redacted = _redact_with_names(text, ["S. Lee", "A. Nguyen"], [_CASE_19_ROW])
    assert output == "Supervised by S. Lee and <PERSON>."
    assert redacted == ["A. Nguyen"]
