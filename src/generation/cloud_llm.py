"""Cloud LLM generation via an OpenAI-compatible endpoint (OpenRouter)."""
from __future__ import annotations

from src.llm_client import chat


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
        )
    except Exception as exc:
        raise RuntimeError(f"cloud_generation_failed: {exc}") from exc
