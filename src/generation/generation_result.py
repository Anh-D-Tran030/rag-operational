"""Typed generation output: an explicit is_abstention signal, not prose matching.

Both local and cloud answer-generation calls are asked for exactly one JSON
object: {"is_abstention": true|false, "answer": "<string>"}. This is the
single shared adapter that parses that wire contract; callers get a
GenerationResult or None (malformed), never a raw string to pattern-match.
Free-text abstention detection (src.abstention.is_abstention_answer) stays
for legacy callers only and is never used to classify this signal.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

# Appended to the vector/hybrid answer prompts. The model must set is_abstention
# whenever the context cannot answer the question -- including a near-miss
# (a related topic named, but not the specific fact asked for).
GENERATION_JSON_INSTRUCTION = (
    'Respond with exactly one JSON object and nothing else: '
    '{"is_abstention": true|false, "answer": "<string>"}\n'
    "Set is_abstention to true whenever the provided context cannot answer the "
    "question -- including when it names a related topic but not the specific "
    "fact asked for. When is_abstention is true, the answer field is ignored and "
    "may be empty. Set is_abstention to false only when giving an actual "
    "substantive answer grounded in the context; when false, answer must be a "
    "complete, non-empty sentence."
)


@dataclass(frozen=True)
class GenerationResult:
    is_abstention: bool
    answer: str


def parse_generation_result(raw: str) -> GenerationResult | None:
    """Parse the strict wire contract. Returns None on any malformed reply."""
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
    except ValueError:
        return None
    try:
        parsed = json.loads(raw[start:end])
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or "is_abstention" not in parsed:
        return None
    is_abstention = parsed["is_abstention"]
    if not isinstance(is_abstention, bool):
        return None
    answer = parsed.get("answer", "")
    if not isinstance(answer, str):
        return None
    if not is_abstention and not answer.strip():
        return None
    return GenerationResult(is_abstention=is_abstention, answer=answer)
