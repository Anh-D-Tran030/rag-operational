"""Single-turn chat against an OpenAI-compatible endpoint.

The project routes every cloud LLM call through OpenRouter, which speaks the
OpenAI wire format. Keeping construction in one place means the base URL, key and
error handling are consistent across generation, SQL synthesis, the faithfulness
gate and ingest-time context generation.
"""
from __future__ import annotations

from dataclasses import dataclass

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
) -> str:
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

    extra = {} if temperature is None else {"temperature": temperature}
    response = await client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        **extra,
    )
    return response.choices[0].message.content or ""
