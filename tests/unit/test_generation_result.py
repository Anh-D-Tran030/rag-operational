"""Unit tests for src/generation/generation_result.py."""
from __future__ import annotations

import json

from src.generation.generation_result import GenerationResult, parse_generation_result


def _wire(**fields) -> str:
    return json.dumps(fields)


def test_valid_false_with_answer_parses():
    result = parse_generation_result(_wire(is_abstention=False, answer="Staff count is 5."))
    assert result == GenerationResult(is_abstention=False, answer="Staff count is 5.")


def test_valid_true_parses_even_with_empty_answer():
    result = parse_generation_result(_wire(is_abstention=True, answer=""))
    assert result == GenerationResult(is_abstention=True, answer="")


def test_true_ignores_nonempty_answer_field_content_but_still_parses():
    result = parse_generation_result(_wire(is_abstention=True, answer="some prose"))
    assert result is not None
    assert result.is_abstention is True


def test_tolerates_surrounding_prose_around_the_json_object():
    raw = 'Sure, here it is:\n{"is_abstention": false, "answer": "5 staff."}\nThanks.'
    result = parse_generation_result(raw)
    assert result == GenerationResult(is_abstention=False, answer="5 staff.")


def test_invalid_json_returns_none():
    assert parse_generation_result("{is_abstention: false, answer: 5}") is None


def test_missing_is_abstention_key_returns_none():
    assert parse_generation_result(_wire(answer="5 staff.")) is None


def test_non_bool_is_abstention_returns_none():
    assert parse_generation_result(_wire(is_abstention="false", answer="5 staff.")) is None


def test_empty_answer_with_false_returns_none():
    assert parse_generation_result(_wire(is_abstention=False, answer="")) is None


def test_whitespace_only_answer_with_false_returns_none():
    assert parse_generation_result(_wire(is_abstention=False, answer="   ")) is None


def test_non_string_answer_returns_none():
    assert parse_generation_result(_wire(is_abstention=False, answer=5)) is None


def test_no_braces_at_all_returns_none():
    assert parse_generation_result("no json here") is None


def test_empty_string_returns_none():
    assert parse_generation_result("") is None
