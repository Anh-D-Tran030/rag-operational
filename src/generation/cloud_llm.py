"""Cloud LLM generation via an OpenAI-compatible endpoint (OpenRouter)."""
from __future__ import annotations

import logging

from src.llm_client import chat

logger = logging.getLogger(__name__)

# A format fallback needs positive evidence, not a marker in some message: a
# trusted bad-request status (400/422) plus an explicit unsupported-parameter
# diagnostic bound to `response_format` / its requested `json_object` type.
_FALLBACK_STATUSES = frozenset({400, 422})
_PARAM_MARKERS = ("response_format", "json_object")
_REJECTION_PHRASES = (
    "unsupported", "not supported", "does not support", "doesn't support",
    "unrecognized", "unknown parameter",
)
# Codes that say the failure was something else, even on a 400-class status.
_OTHER_FAILURE_CODES = ("rate_limit", "auth", "api_key", "quota", "timeout", "permission")


def _error_fields(exc: Exception) -> dict:
    """The provider's structured error object (nested or flat body), if any."""
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return {}
    inner = body.get("error")
    return inner if isinstance(inner, dict) else body


def _is_unsupported_response_format_error(exc: Exception) -> bool:
    """True only for a trusted bad-request rejecting the response_format parameter.

    An untyped exception, a marker with no rejection wording, an unrelated
    unsupported parameter, any other status (401/403/429/5xx), a timeout or a
    contradictory structured code never qualifies, however many format markers
    its text contains.
    """
    status = getattr(exc, "status_code", None)
    if type(status) is not int or status not in _FALLBACK_STATUSES:
        return False
    fields = _error_fields(exc)
    for code in (fields.get("code"), getattr(exc, "code", None)):
        if type(code) is int and code not in _FALLBACK_STATUSES:
            return False
        if isinstance(code, str) and any(m in code.lower() for m in _OTHER_FAILURE_CODES):
            return False
    parts = [getattr(exc, "param", None), getattr(exc, "code", None)]
    parts += [fields.get(k) for k in ("param", "code", "message", "type")]
    if not fields:
        parts.append(str(exc))
    text = " ".join(p.lower() for p in parts if isinstance(p, str))
    return any(m in text for m in _PARAM_MARKERS) and any(r in text for r in _REJECTION_PHRASES)


async def generate_cloud(
    prompt: str,
    model: str,
    api_key: str,
    base_url: str,
) -> str:
    """Call a cloud LLM and return the generated text.

    Routed through OpenRouter's OpenAI-compatible API; `model` is an OpenRouter
    model id (e.g. "openai/gpt-oss-120b").
    Raises RuntimeError on API failure; raises ValueError if prompt or api_key is empty.
    """
    if not prompt:
        raise ValueError("prompt must not be empty")
    if not api_key:
        raise ValueError("api_key must not be empty")

    try:
        return await chat(
            prompt=prompt,
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        # The strict JSON contract is a request, not a guarantee: some
        # OpenRouter-routed models reject response_format entirely. Fall back
        # to an unconstrained call only when that parameter is what the
        # provider rejected — any other failure (auth, rate limit, network,
        # model error) should not be silently retried without it, and is
        # instead wrapped and raised below.
        if not _is_unsupported_response_format_error(exc):
            raise RuntimeError(f"cloud_generation_failed: {exc}") from exc
        logger.warning(
            "cloud generation falling back without response_format: %s", type(exc).__name__
        )
    try:
        return await chat(
            prompt=prompt,
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=1024,
        )
    except Exception as exc:
        raise RuntimeError(f"cloud_generation_failed: {exc}") from exc
