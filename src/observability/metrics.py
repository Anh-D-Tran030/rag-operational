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

rag_agent_iterations = Histogram(
    "rag_agent_iterations",
    "Number of agentic loop iterations per complex query",
)

rag_sql_execution_errors = Counter(
    "rag_sql_execution_errors",
    "SQL self-correction events (execution errors triggering retry)",
)

rag_pii_redactions = Counter(
    "rag_pii_redactions",
    "PII redaction events at pipeline input and output",
    labelnames=["label"],
)
