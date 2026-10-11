"""Single-turn chat against an OpenAI-compatible endpoint.

The project routes every cloud LLM call through OpenRouter, which speaks the
OpenAI wire format. Keeping construction in one place means the base URL, key and
error handling are consistent across generation, SQL synthesis, the faithfulness
gate and ingest-time context generation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
import openai


@dataclass(frozen=True)
class LocalLLMConfig:
    """Where the small-model calls go: routing, query expansion, simple answers.

    These are the only calls that ever needed a GPU. `provider="ollama"` keeps
    them on a local model for development; `provider="openrouter"` sends them out
    over HTTP so a deployed container needs nothing but CPU.
    """

    provider: str
    model: str
    ollama_url: str = ""
    api_key: str = ""
    base_url: str = ""


@dataclass(frozen=True)
class ChatCompletionResult:
    """Text plus provider evidence from one chat completion."""

    text: str
    response_id: str | None
    finish_reason: str | None
    usage: dict[str, int | None]
    model: str | None
    provider: str | None
    configured_output_token_limit: int


async def complete(
    prompt: str,
    config: LocalLLMConfig,
    max_tokens: int = 512,
    system: str | None = None,
    temperature: float | None = None,
) -> str:
    """Single-turn completion via whichever provider `config` selects."""
    if config.provider == "ollama":
        return await _ollama_generate(prompt, config.ollama_url, config.model, system)
    return await chat(
        prompt=prompt,
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        max_tokens=max_tokens,
        system=system,
        temperature=temperature,
    )


async def _ollama_generate(
    prompt: str, ollama_url: str, model: str, system: str | None = None
) -> str:
    body: dict = {"model": model, "prompt": prompt, "stream": False}
    if system:
        body["system"] = system
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{ollama_url}/api/generate", json=body)
        resp.raise_for_status()
        return resp.json().get("response", "")


async def chat(
    prompt: str,
    model: str,
    api_key: str,
    base_url: str,
    max_tokens: int,
    system: str | None = None,
    temperature: float | None = None,
    response_format: dict[str, Any] | None = None,
    return_metadata: bool = False,
) -> str | ChatCompletionResult:
    """Send a single-turn prompt and return the assistant text (never None).

    `temperature` is only sent when set, so the provider default stands for
    open-ended generation. Pin it to 0 for classifier-style calls whose answer
    should not vary between identical requests.
    """
    client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    extra: dict[str, Any] = {}
    if temperature is not None:
        extra["temperature"] = temperature
    if response_format is not None:
        extra["response_format"] = response_format
    response = await client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        **extra,
    )
    choice = response.choices[0]
    usage = getattr(response, "usage", None)
    result = ChatCompletionResult(
        text=choice.message.content or "",
        response_id=getattr(response, "id", None),
        finish_reason=getattr(choice, "finish_reason", None),
        usage={
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        },
        model=getattr(response, "model", None),
        provider=getattr(response, "provider", None),
        configured_output_token_limit=max_tokens,
    )
    return result if return_metadata else result.text
