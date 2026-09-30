"""Prometheus metrics for the RAG pipeline.

All metrics are defined at module level so they are registered once at import
time and shared across the process lifetime.
"""

from prometheus_client import Counter, Histogram, Summary

rag_retrieval_confidence = Histogram(
    "rag_retrieval_confidence",
    "Distribution of top-chunk retrieval confidence scores",
)

rag_reranker_score_max = Histogram(
    "rag_reranker_score_max",
    "Maximum reranker score per query",
)

rag_chunks_per_query = Histogram(
    "rag_chunks_per_query",
    "Number of chunks returned per query after retrieval and reranking",
)

rag_route_decision = Counter(
    "rag_route_decision",
    "Query routing decisions by route type",
    labelnames=["route"],
)

rag_query_cost_usd = Summary(
    "rag_query_cost_usd",
    "Estimated cost in USD per query (LLM tokens)",
)

rag_latency_by_component = Histogram(
    "rag_latency_by_component",
    "Latency in seconds broken down by pipeline component",
    labelnames=["component"],
)

rag_cache_hit_total = Counter(
    "rag_cache_hit_total",
    "Cache lookup results (hit or miss)",
    labelnames=["result"],
)

rag_faithfulness_gate_failures = Counter(
    "rag_faithfulness_gate_failures",
    "Answers blocked by the faithfulness gate before delivery",
)

rag_faithfulness_gate_total = Counter(
    "rag_faithfulness_gate_total",
    "Faithfulness gate decisions by outcome; only 'passed' delivers an answer",
    labelnames=["outcome"],
)

rag_agent_iterations = Histogram(
    "rag_agent_iterations",
    "Number of agentic loop iterations per complex query",
)

rag_sql_execution_errors = Counter(
    "rag_sql_execution_errors",
    "SQL self-correction events (execution errors triggering retry)",
)

rag_sql_generation_attempts_total = Counter(
    "rag_sql_generation_attempts_total",
    "SQL model generation attempts by phase, model, and outcome",
    labelnames=["phase", "model", "outcome"],
)

rag_sql_fallback_invocations_total = Counter(
    "rag_sql_fallback_invocations_total",
    "SQL fallback model invocations by primary failure reason",
    labelnames=["trigger"],
)

rag_sql_generation_unavailable_total = Counter(
    "rag_sql_generation_unavailable_total",
    "SQL generation requests unavailable after model failure",
    labelnames=["reason"],
)

rag_sql_unsupported_by_schema_total = Counter(
    "rag_sql_unsupported_by_schema_total",
    "SQL requests blocked because curated views cannot answer them",
)

rag_sql_partial_schema_support_total = Counter(
    "rag_sql_partial_schema_support_total",
    "SQL requests blocked because curated views contain only partial evidence",
)

rag_sql_schema_guard_abstention_total = Counter(
    "rag_sql_schema_guard_abstention_total",
    "SQL requests safely abstained by the schema coverage guard",
)

rag_pii_redactions = Counter(
    "rag_pii_redactions",
    "PII redaction events at pipeline input and output",
    labelnames=["label"],
)
