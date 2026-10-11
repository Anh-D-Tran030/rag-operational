"""Bounded provider diagnostics, timing/diagnostic records and exhaustion status.

Records never hold credentials, raw exception messages or unbounded SQL. Alias-only
exhaustion is reported as ``candidate_alias_rejected``, never as proven missing schema.
"""
from __future__ import annotations

import hashlib
import re

import openai

from src.llm_client import ChatCompletionResult
from src.retrieval.sql_candidate_aliases import _SCHEMA_CAPABILITY_REJECTION

_DIAGNOSTIC_RAW_RESPONSE_LIMIT = 512
_DIAGNOSTIC_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]+"),
    re.compile(r"(?i)\b(?:api[_-]?key|authorization)\s*[:=]\s*\S+"),
)
CANDIDATE_ALIAS_REJECTED = "candidate_alias_rejected"


def _safe_diagnostic_response(raw: str) -> dict[str, str | int]:
    """Provide bounded, credential-redacted model output for opt-in diagnostics."""
    redacted = raw
    for pattern in _DIAGNOSTIC_SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    excerpt = redacted[:_DIAGNOSTIC_RAW_RESPONSE_LIMIT]
    if len(redacted) > _DIAGNOSTIC_RAW_RESPONSE_LIMIT:
        excerpt += "…[truncated]"
    return {
        "sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "char_count": len(raw),
        "excerpt": excerpt,
    }


def _provider_error_details(exc: Exception) -> tuple[str, dict[str, str | int | None]]:
    """Classify provider errors without retaining potentially sensitive messages."""
    if isinstance(exc, openai.APITimeoutError):
        outcome = "timeout"
        error_class = "api_timeout"
    elif isinstance(exc, openai.APIConnectionError):
        outcome = "provider_error"
        error_class = "api_connection_error"
    elif isinstance(exc, openai.RateLimitError):
        outcome = "provider_error"
        error_class = "rate_limit"
    elif isinstance(exc, openai.APIStatusError):
        outcome = "provider_error"
        error_class = "api_status_error"
    else:
        outcome = "provider_error"
        error_class = type(exc).__name__.lower()
    status = getattr(exc, "status_code", None)
    return outcome, {"class": error_class, "status": status if isinstance(status, int) else None}


def empty_completion_evidence(model: str, max_tokens: int) -> dict[str, object]:
    """Evidence for a call without provider metadata (no reply or a mocked string)."""
    return {
        "response_id": None,
        "finish_reason": None,
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        "configured_output_token_limit": max_tokens,
        "model": model,
        "provider": None,
    }


def _completion_text_and_evidence(
    completion: ChatCompletionResult | str,
    model: str,
    max_tokens: int,
) -> tuple[str, dict[str, object]]:
    """Normalize real and mocked completions into bounded diagnostic evidence."""
    if isinstance(completion, ChatCompletionResult):
        return completion.text, {
            "response_id": completion.response_id,
            "finish_reason": completion.finish_reason,
            "usage": completion.usage,
            "configured_output_token_limit": completion.configured_output_token_limit,
            "model": completion.model or model,
            "provider": completion.provider,
        }
    return str(completion or ""), empty_completion_evidence(model, max_tokens)


def timing_record(
    *, latency_ms: float, timed_out: bool, timeout_s: float, output_token_limit: int,
    n_candidates: int, phase: str, model: str,
) -> dict:
    """One candidate-call timing entry, before its outcome is known."""
    return {
        "latency_ms": latency_ms,
        "timed_out": timed_out,
        "timeout_s": timeout_s,
        "configured_output_token_limit": output_token_limit,
        "requested_candidates": n_candidates,
        "phase": phase,
        "model": model,
    }


def diagnostic_record(*, raw: str, **fields: object) -> dict:
    """One opt-in candidate-call diagnostic in its established key order."""
    keys = (
        "case_id", "query_sha256", "model_attempt", "phase", "model",
        "fallback_trigger_reason", "elapsed_ms", "timeout_s", "timed_out",
        "timeout_error_class", "completion_evidence",
    )
    tail = (
        "provider_error", "structured_output_error", "parse_source",
        "candidate_diagnostics", "outcome", "will_retry",
    )
    return {
        **{key: fields[key] for key in keys},
        "raw_response": _safe_diagnostic_response(raw),
        **{key: fields[key] for key in tail},
    }


def exhaustion_status(rejections: list[list[str]]) -> dict[str, str | None]:
    """Generation status after every candidate was rejected.

    Only when every rejected candidate carried the alias rejection is it alias-only
    exhaustion. That is a candidate-filter outcome, not proof of missing schema.
    """
    if rejections and all(_SCHEMA_CAPABILITY_REJECTION in r for r in rejections):
        return {
            "status": "no_valid_candidates", "reason": CANDIDATE_ALIAS_REJECTED,
            "failure_kind": CANDIDATE_ALIAS_REJECTED,
            "candidate_rejection_reason": _SCHEMA_CAPABILITY_REJECTION,
        }
    return {"status": "no_valid_candidates", "reason": None}
