"""Langfuse tracing wrapper for the RAG pipeline.

RAGTracer operates in noop mode when public_key or secret_key is empty,
allowing the pipeline to run without a Langfuse instance configured.
trace_id is always returned as a plain string; the caller is responsible
for passing it explicitly to subsequent span/score calls.
"""

from typing import Any
from uuid import uuid4


class RAGTracer:
    """Thin wrapper around Langfuse that degrades to noop when keys are absent."""

    def __init__(self, public_key: str, secret_key: str, host: str) -> None:
        """Initialise the tracer.

        Sets noop mode when either key is empty; otherwise creates a real
        Langfuse client.
        """
        self._noop = not public_key or not secret_key
        self._client: Any = None
        if not self._noop:
            from langfuse import Langfuse  # type: ignore[import]

            self._client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
            )

    def start_trace(self, name: str, user_query: str) -> tuple[str, Any]:
        """Begin a new trace for a single user query.

        Returns (trace_id, trace_obj). In noop mode trace_obj is None and
        trace_id is a random UUID string.
        """
        if self._noop:
            return (str(uuid4()), None)
        trace = self._client.trace(name=name, input=user_query)
        return (str(trace.id), trace)

    def log_span(
        self,
        trace_obj: Any,
        name: str,
        input: Any,
        output: Any,
        latency_ms: float,
        metadata: dict | None = None,
    ) -> None:
        """Record a child span on an existing trace.

        Does nothing when in noop mode or when trace_obj is None.
        """
        if self._noop or trace_obj is None:
            return
        trace_obj.span(
            name=name,
            input=input,
            output=output,
            metadata=metadata or {},
            end_time=None,  # latency tracked separately
        )

    def end_trace(self, trace_obj: Any, output: Any, usage: dict) -> None:
        """Finalise a trace with its output and token usage.

        Does nothing when in noop mode or when trace_obj is None.
        """
        if self._noop or trace_obj is None:
            return
        trace_obj.update(output=output, usage=usage)
        self._client.flush()

    def log_score(self, trace_id: str, name: str, value: float) -> None:
        """Attach a named score to an existing trace by its ID.

        Does nothing in noop mode.
        """
        if self._noop:
            return
        self._client.score(trace_id=trace_id, name=name, value=value)
