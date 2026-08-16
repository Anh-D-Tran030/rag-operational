"""POST /query route."""
from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, field_validator

router = APIRouter(prefix="/query")


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


@router.post("/", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest) -> QueryResponse:
    """Handle a RAG query and return a cited answer."""
    from src.generation.query_handler import handle_query

    t0 = time.monotonic()
    try:
        result = await handle_query(
            query=body.query,
            qdrant_client=request.app.state.qdrant_client,
            settings=request.app.state.settings,
            tracer=request.app.state.tracer,
            cache=request.app.state.cache,
            filters=body.filters,
        )
    except Exception as exc:
        detail = {"error": "query_failed", "detail": str(exc)}
        raise HTTPException(status_code=500, detail=detail) from exc

    citations: list[dict] = result.get("citations", [])
    vector_citations = [
        VectorCitation(
            doc_type=c.get("doc_type"),
            venue=c.get("venue"),
            date=c.get("date"),
            shift_id=c.get("shift_id"),
            page_num=c.get("page_num"),
            snippet=c.get("snippet", ""),
            score=c.get("score", 0.0),
        )
        for c in citations if "snippet" in c
    ]
    sql_citations = [
        SQLCitation(
            view_name=c.get("view_name", ""),
            sql_used=c.get("sql_used", ""),
            row_count=c.get("row_count", 0),
        )
        for c in citations if "view_name" in c
    ]

    return QueryResponse(
        answer=result.get("answer", ""),
        vector_citations=vector_citations,
        sql_citations=sql_citations,
        trace_id=result.get("trace_id", ""),
        route=result.get("route", ""),
        tool=result.get("tool", ""),
        latency_ms=result.get("latency_ms", (time.monotonic() - t0) * 1000),
    )
