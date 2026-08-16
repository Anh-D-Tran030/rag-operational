"""Intent router: classify a query as vector_search, text_to_sql, or hybrid."""
from __future__ import annotations

import json
import logging

from src.llm_client import LocalLLMConfig, complete

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a query router. Given a user query, classify it and return ONLY valid JSON "
    "with two keys: 'tool' (one of: vector_search, text_to_sql, hybrid) and "
    "'complexity' (one of: simple, complex). "
    "Use 'vector_search' for document/procedure/compliance questions. "
    "Use 'text_to_sql' for aggregation/count/staffing-level questions. "
    "Use 'hybrid' for questions requiring both documents and database data. "
    "Use 'complex' when multi-step reasoning or self-correction may be needed.\n\n"
    "A number in the question does not make it a database question. Licence terms,"
    " capacities and stated thresholds are written in documents; the database holds"
    " logged events and rostered counts.\n\n"
    "Examples:\n"
    'Query: What does the evacuation procedure say about assembly points?\n'
    'JSON: {"tool": "vector_search", "complexity": "simple"}\n'
    'Query: What trading hours does the licence permit on a Friday?\n'
    'JSON: {"tool": "vector_search", "complexity": "simple"}\n'
    'Query: What is the maximum number of patrons the venue is licensed for?\n'
    'JSON: {"tool": "vector_search", "complexity": "simple"}\n'
    'Query: How many incidents were logged in April?\n'
    'JSON: {"tool": "text_to_sql", "complexity": "simple"}\n'
    'Query: What was the average staff count across evening shifts?\n'
    'JSON: {"tool": "text_to_sql", "complexity": "simple"}\n'
    'Query: Which shift had the most security staff on duty?\n'
    'JSON: {"tool": "text_to_sql", "complexity": "simple"}\n'
    'Query: Did staffing on 12 March meet the minimum the procedure requires?\n'
    'JSON: {"tool": "hybrid", "complexity": "complex"}\n'
    'Query: Were RSA interventions logged within the window the policy sets out?\n'
    'JSON: {"tool": "hybrid", "complexity": "complex"}'
)

_DEFAULT = {"tool": "hybrid", "complexity": "complex"}


async def classify_intent(query: str, config: LocalLLMConfig) -> dict:
    """Return routing decision for the query.

    Parses a JSON response with 'tool' and 'complexity' keys.
    Falls back to {tool: hybrid, complexity: complex} on any provider error.
    """
    prompt = f"{_SYSTEM}\n\nQuery: {query}\n\nJSON:"
    try:
        # Routing is a classification, not a generation: identical queries must
        # take identical routes, so the sampling temperature is pinned.
        raw = await complete(prompt, config, max_tokens=128, temperature=0.0)
        # Extract first JSON object from the response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            return _DEFAULT
        parsed = json.loads(raw[start:end])
        tool = parsed.get("tool", "hybrid")
        complexity = parsed.get("complexity", "complex")
        if tool not in {"vector_search", "text_to_sql", "hybrid"}:
            tool = "hybrid"
        if complexity not in {"simple", "complex"}:
            complexity = "complex"
        return {"tool": tool, "complexity": complexity}
    except Exception as exc:
        logger.warning("classify_intent fell back to default: %s", exc)
        return _DEFAULT
