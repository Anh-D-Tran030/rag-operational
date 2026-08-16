"""FastAPI application entry point — wiring only, no business logic."""
from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import start_http_server
from prometheus_fastapi_instrumentator import Instrumentator
from sqlalchemy import text as sqlalchemy_text
from sqlalchemy.ext.asyncio import create_async_engine

from src.api.routes.feedback import router as feedback_router
from src.api.routes.ingest import router as ingest_router
from src.api.routes.query import router as query_router
from src.config import Settings
from src.observability.tracer import RAGTracer
from src.retrieval.cache import SemanticCache


@asynccontextmanager
async def lifespan(app: FastAPI):
    from qdrant_client import AsyncQdrantClient

    settings = Settings()
    app.state.settings = settings

    app.state.qdrant_client = AsyncQdrantClient(url=settings.qdrant_url)
    app.state.tracer = RAGTracer(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
    app.state.cache = SemanticCache(
        cache_url=settings.semantic_cache_url,
        embedding_model=settings.embedding_model,
    )
    app.state.db_engine = create_async_engine(settings.db_url, echo=False)

    start_http_server(settings.prometheus_port)

    yield

    await app.state.qdrant_client.close()
    await app.state.db_engine.dispose()


app = FastAPI(title="Operational RAG API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

app.include_router(query_router)
app.include_router(ingest_router)
app.include_router(feedback_router)


@app.get("/health")
async def health(response: Response) -> dict:
    """Report readiness of the dependencies this deployment actually has.

    Returns 503 when a required dependency is down. A probe that answers 200
    unconditionally tells an orchestrator nothing, and keeps a broken container
    in the load balancer.
    """
    settings: Settings = app.state.settings

    qdrant_ok = False
    try:
        await app.state.qdrant_client.get_collections()
        qdrant_ok = True
    except Exception:
        pass

    db_ok = False
    try:
        async with app.state.db_engine.connect() as conn:
            await conn.execute(sqlalchemy_text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    checks = {"qdrant": qdrant_ok, "db": db_ok}

    # Ollama is only a dependency when the small-model calls are pointed at it.
    # The CPU-only deployment talks to OpenRouter, so probing Ollama there would
    # report a permanent failure for a service it does not use.
    if settings.local_provider == "ollama":
        ollama_ok = False
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"{settings.ollama_url}/api/tags")
                ollama_ok = r.status_code == 200
        except Exception:
            pass
        checks["ollama"] = ollama_ok

    healthy = all(checks.values())
    response.status_code = 200 if healthy else 503
    return {"status": "ok" if healthy else "degraded", **checks}
