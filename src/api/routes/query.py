"""POST /query route."""
from __future__ import annotations

import asyncio
import json
import re
import sys
import time
import uuid

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

router = APIRouter(prefix="/query")

_STAGE_KEYS = (
    "router_ms", "retrieval_ms", "reranking_ms", "schema_embedding_ms", "unattributed_ms",
    "schema_search_ms", "schema_linking_ms", "final_generation_ms", "faithfulness_ms",
)
_SQL_STAGE_KEYS = ("generation_ms", "validation_ms", "execution_ms")
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,80}$")


def _request_id(raw: object) -> str:
    """Accept a caller's X-Request-ID only if short and inert; otherwise mint one."""
    return raw if isinstance(raw, str) and _REQUEST_ID.fullmatch(raw) else uuid.uuid4().hex


def _log_request(rid: str, status: int | str, t0: float, route: object = None,
                 timings: dict | None = None, exception_class: str | None = None) -> None:
    """One JSON line per request on stderr: no query, answer, SQL, or exception text."""
    record = {"event": "query_request", "request_id": rid, "status": status,
              "elapsed_ms": round((time.monotonic() - t0) * 1000, 2), "route": route,
              "stage_timings": timings,
              **({"exception_class": exception_class} if exception_class else {})}
    print(json.dumps(record), file=sys.stderr, flush=True)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _stage_timings(components: object) -> dict[str, float] | None:
    """Keep only allowlisted numeric stage durations; drop SQL text, models, and flags."""
    if not isinstance(components, dict):
        return None
    timings = {k: float(components[k]) for k in _STAGE_KEYS if _is_number(components.get(k))}
    sql = components.get("sql")
    if isinstance(sql, dict):
        timings |= {f"sql_{k}": float(sql[k]) for k in _SQL_STAGE_KEYS if _is_number(sql.get(k))}
    return timings or None


class QueryRequest(BaseModel):
    query: str
    filters: dict | None = None

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be empty")
        return v


class VectorCitation(BaseModel):
    doc_type: str | None
    venue: str | None
    date: str | None
    shift_id: str | None
    page_num: int | None
    snippet: str
    score: float


class SQLCitation(BaseModel):
    view_name: str
    sql_used: str
    row_count: int


class QueryResponse(BaseModel):
    answer: str
    vector_citations: list[VectorCitation]
    sql_citations: list[SQLCitation]
    trace_id: str
    route: str
    tool: str
    latency_ms: float
    schema_status: str | None = None
    missing_schema: list[dict[str, str]] | None = None
    rows: list[dict] | None = None
    stage_timings: dict[str, float] | None = None


@router.post("/", response_model=QueryResponse, response_model_exclude_none=True)
async def query(
    request: Request, body: QueryRequest, response: Response
) -> QueryResponse | JSONResponse:
    """Handle a RAG query and return a cited answer."""
    from src.generation.query_handler import handle_query

    t0 = time.monotonic()
    rid = _request_id(request.headers.get("x-request-id"))
    response.headers["X-Request-ID"] = rid
    try:
        result = await handle_query(
            query=body.query,
            qdrant_client=request.app.state.qdrant_client,
            settings=request.app.state.settings,
            tracer=request.app.state.tracer,
            cache=request.app.state.cache,
            filters=body.filters,
        )
    except asyncio.CancelledError:  # client disconnected mid-request
        _log_request(rid, "cancelled", t0)
        raise
    except Exception as exc:
        _log_request(rid, 500, t0, exception_class=type(exc).__name__)
        return JSONResponse(status_code=500, headers={"X-Request-ID": rid},
                            content={"error": "query_failed", "request_id": rid})

    citations: list[dict] = result.get("citations", [])
    _vector_meta = ("doc_type", "venue", "date", "shift_id", "page_num")
    vector_citations = [
        VectorCitation(**{k: c.get(k) for k in _vector_meta},
                       snippet=c.get("snippet", ""), score=c.get("score", 0.0))
        for c in citations if "snippet" in c
    ]
    sql_citations = [
        SQLCitation(view_name=c.get("view_name", ""), sql_used=c.get("sql_used", ""),
                    row_count=c.get("row_count", 0))
        for c in citations if "view_name" in c
    ]
    timings = _stage_timings(result.get("latency_components"))
    _log_request(rid, 200, t0, result.get("route"), timings)
    return QueryResponse(
        answer=result.get("answer", ""),
        vector_citations=vector_citations,
        sql_citations=sql_citations,
        trace_id=result.get("trace_id", ""),
        route=result.get("route", ""),
        tool=result.get("tool", ""),
        latency_ms=result.get("latency_ms", (time.monotonic() - t0) * 1000),
        schema_status=result.get("schema_status"),
        missing_schema=result.get("missing_schema"),
        rows=result.get("rows"),
        stage_timings=timings,
    )
