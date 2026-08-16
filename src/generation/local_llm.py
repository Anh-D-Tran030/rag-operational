"""Small-model generation via the configured provider."""
from __future__ import annotations

import json
from collections.abc import AsyncGenerator

import httpx

from src.llm_client import LocalLLMConfig, complete


async def generate_local(
    prompt: str,
    config: LocalLLMConfig,
    stream: bool = False,
) -> str | AsyncGenerator[str, None]:
    """Generate a response using the configured small model.

    stream=False: returns the full response string.
    stream=True: returns an AsyncGenerator yielding tokens. Streaming is only
      implemented for the Ollama provider; no caller uses it today.
    Wraps provider errors as {"error": "generation_failed", "detail": <msg>}.
    Raises ValueError if prompt is empty.
    """
    if not prompt:
        raise ValueError("prompt must not be empty")

    if stream:
        if config.provider != "ollama":
            raise NotImplementedError("streaming is only implemented for the ollama provider")
        return _stream(prompt, config.ollama_url, config.model)

    try:
        return await complete(prompt, config, max_tokens=1024)
    except Exception as exc:
        return str({"error": "generation_failed", "detail": str(exc)})


async def _stream(prompt: str, ollama_url: str, model: str) -> AsyncGenerator[str, None]:
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{ollama_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": True},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line:
                        token = json.loads(line).get("response", "")
                        if token:
                            yield token
    except Exception as exc:
        yield str({"error": "generation_failed", "detail": str(exc)})
