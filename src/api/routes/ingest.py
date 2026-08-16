"""POST /ingest route."""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter(prefix="/ingest")


class IngestRequest(BaseModel):
    file_path: str
    doc_type: str
    venue: str
    date: str
    shift_id: str


class IngestResponse(BaseModel):
    doc_type: str
    chunks_ingested: int
    chunks_deduplicated: int
    status: str


async def _progress_stream(request: Request, body: IngestRequest):
    """SSE generator that runs ingestion and streams progress events."""
    from src.ingestion.embedder import Embedder
    from src.ingestion.pipeline import ingest_document

    yield "data: {\"status\": \"started\"}\n\n"
    await asyncio.sleep(0)

    settings = request.app.state.settings
    embedder = Embedder(settings.embedding_model)
    try:
        result = await ingest_document(
            file_path=body.file_path,
            doc_metadata={
                "doc_type": body.doc_type,
                "venue": body.venue,
                "date": body.date,
                "shift_id": body.shift_id,
            },
            qdrant_client=request.app.state.qdrant_client,
            embedder=embedder,
            settings=settings,
        )
        payload = IngestResponse(
            doc_type=result["doc_type"],
            chunks_ingested=result["chunks_ingested"],
            chunks_deduplicated=result["chunks_deduplicated"],
            status="ok",
        )
        yield f"data: {json.dumps(payload.model_dump())}\n\n"
    except FileNotFoundError as exc:
        yield f"data: {json.dumps({'status': 'error', 'detail': str(exc)})}\n\n"
    except Exception as exc:
        yield f"data: {json.dumps({'status': 'error', 'detail': str(exc)})}\n\n"


@router.post("/")
async def ingest(request: Request, body: IngestRequest) -> StreamingResponse:
    """Ingest a document and stream progress via SSE."""
    return StreamingResponse(
        _progress_stream(request, body),
        media_type="text/event-stream",
    )


@router.post("/sync", response_model=IngestResponse)
async def ingest_sync(request: Request, body: IngestRequest) -> IngestResponse:
    """Ingest a document and return a single JSON response (used by tests)."""
    from src.ingestion.embedder import Embedder
    from src.ingestion.pipeline import ingest_document

    settings = request.app.state.settings
    embedder = Embedder(settings.embedding_model)
    try:
        result = await ingest_document(
            file_path=body.file_path,
            doc_metadata={
                "doc_type": body.doc_type,
                "venue": body.venue,
                "date": body.date,
                "shift_id": body.shift_id,
            },
            qdrant_client=request.app.state.qdrant_client,
            embedder=embedder,
            settings=settings,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        detail = {"error": "ingest_failed", "detail": str(exc)}
        raise HTTPException(status_code=500, detail=detail) from exc

    return IngestResponse(
        doc_type=result["doc_type"],
        chunks_ingested=result["chunks_ingested"],
        chunks_deduplicated=result["chunks_deduplicated"],
        status="ok",
    )
