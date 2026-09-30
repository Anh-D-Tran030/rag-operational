# Enterprise Operational RAG System

## Overview

Venue operations teams face a fundamental information retrieval problem: answering questions
that simultaneously span unstructured compliance documents, messy shift-log narratives, and
structured tabular records — staff counts, incident logs, scheduling data — spread across
incompatible formats and storage systems. This system is a production-grade adaptive RAG
pipeline that fuses all three modalities at query time. Given a question such as "Did floor
staffing meet safety-protocol thresholds during peak hours on Saturday?", the pipeline routes
to the correct retrieval strategy, retrieves from Qdrant and/or executes SQL against curated
views, reranks with FlashRank, gates on faithfulness, and returns a cited answer with full
OTel trace coverage.

Key capabilities:
- Hybrid BM25 + dense retrieval with RRF merging (k=60), top-50 → FlashRank Nano rerank → top-7
- Adaptive intent router (qwen/qwen3-30b-a3b-instruct-2507, via OpenRouter) classifies
  vector_search / text_to_sql / hybrid with 23/23 accuracy on the golden category set
- Text-to-SQL over five curated views with schema-linking, 3-candidate generation, self-correction
- Cloud LLM (openai/gpt-oss-120b via OpenRouter) for complex multi-hop queries and SQL generation
- Faithfulness gate (qwen/qwen3-30b-a3b-instruct-2507) blocks hallucinated answers pre-delivery;
  scored 5/5 on grounded/ungrounded probe vs 0/5 for gpt-oss-120b and 1/5 for mistral-nemo
- PII redaction (Presidio NER + staff-ID regex) on every query in and answer out
- Prometheus metrics + Langfuse OTel traces on every pipeline component
- CPU-only Docker image — no GPU, no Ollama; all LLM calls via OpenRouter HTTP

---

## Architecture

```
 FastAPI (/query, /ingest, /feedback, /health, /ready)
              │
     ┌────────▼────────┐
     │  Intent Router  │  qwen3-30b (OpenRouter)
     └──┬──────────┬───┘
        │          │
┌───────▼──────┐  ┌▼─────────────────┐
│ vector_search│  │   text_to_sql    │
│  (hybrid RRF)│  │ schema-link+SQL  │
└───────┬──────┘  └────────┬─────────┘
        │                  │
┌───────▼──────────────────▼─────────┐
│            FlashRank Reranker       │
└────────────────────┬───────────────┘
                     │
          ┌──────────▼──────────┐
          │      Generator      │
          │  mistral-nemo local │   (simple queries)
          │  gpt-oss-120b cloud │   (complex / hybrid)
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │  Faithfulness Gate  │  qwen3-30b (OpenRouter)
          └──────────┬──────────┘
                     │
               Cited Response

Dependencies:
  Managed Qdrant (vector index)   │ SQLite demo DB (curated views)
  GPTCache (semantic cache, opt)  │ Langfuse (OTel traces)
  Prometheus + Grafana (metrics)  │
```

### Supported v1 topology

```text
Client
  ↓
Fly.io FastAPI
  ├── SQLite demo relational DB (baked into the application image)
  ├── Managed Qdrant (versioned document and schema collections)
  └── OpenRouter (routing, generation, contextualization, and faithfulness calls)
```

This is the only supported v1 deployment topology. SQLite is intentionally the relational
database for the small portfolio demo. PostgreSQL portability, a self-hosted production Qdrant
deployment, and full-stack Docker Compose reproducibility are not claimed for v1.

---

## Tech Stack

| Layer | Tool | Justification |
|:------|:-----|:--------------|
| Backend | FastAPI | Async, Pydantic v2 validation, Prometheus middleware, OpenAPI docs out of the box |
| Vector DB | Managed Qdrant | Durable versioned vector collections; runtime selection makes corpus rollback explicit |
| Embedding | all-MiniLM-L6-v2 | 384-dim, 80MB, CPU batch ingestion; contextual retrieval + FlashRank compensate for first-pass quality |
| Reranker | FlashRank Nano | 4MB CPU-only cross-encoder; sub-50ms; no Torch dependency; ~2pt nDCG loss vs ColBERT is acceptable |
| Router LLM | qwen/qwen3-30b-a3b-instruct-2507 (OpenRouter) | Chosen by measurement: 23/23 on golden categories vs 21/23 for mistral-nemo |
| Local LLM | mistralai/mistral-nemo (OpenRouter) | Query expansion and simple-query generation; degrades gracefully |
| Cloud LLM | openai/gpt-oss-120b (OpenRouter) | Complex multi-hop synthesis and SQL generation |
| Gate LLM | qwen/qwen3-30b-a3b-instruct-2507 (OpenRouter) | Faithfulness gate; 5/5 on grounded/ungrounded probe (gpt-oss-120b: 0/5, mistral-nemo: 1/5) |
| Semantic Cache | GPTCache | Disabled by default (`semantic_cache_url=""`); 40-80% LLM cost reduction in production on repeated queries |
| PII Guardrails | Presidio NER + regex | Redacts PERSON/EMAIL/PHONE/CREDIT_CARD/TFN/MEDICARE on every query in and answer out |
| Tracing | Langfuse v3 + OTel GenAI semconv | Full trace trees; OTLP endpoint; no SDK lock-in; cost-per-query visible in dashboard |
| Monitoring | Prometheus + Grafana | 10 custom RAG metrics; 7-panel Grafana dashboard; per-component latency histograms |
| Eval | DeepEval + RAGAS | DeepEval pytest CI gate blocks regressions; RAGAS computes retrieval + generation metrics separately |

---

## Milestones

| # | Name | Status | Scope |
|:--|:-----|:-------|:------|
| M1 | Foundation | Complete | Repo scaffold, config, Docker infra, ingestion pipeline (parser → chunker → dedup → contextual prefix → embed → Qdrant upsert), 33 unit tests |
| M2 | Core RAG Pipeline | Complete | Hybrid BM25+dense+RRF retrieval, FlashRank reranker, text-to-SQL with schema-linking, intent router, query handler, Prometheus + Langfuse observability |
| M3 | API + UI | Complete | FastAPI routes (/query /ingest /feedback /health), Grafana dashboard, 6 integration tests |
| M4 | Eval + CI/CD | Complete | 30-question golden dataset, DeepEval CI gate, RAGAS runner, GitHub Actions (lint → test → eval → deploy) |
| M5 | Polish + README | Complete | README, load test script, design decisions document |
| M6 | Production hardening | Complete | OpenRouter CPU-only deploy, PII guardrails, faithfulness gate, router model selection by measurement |

---

## Quick Start

**Prerequisites:** Python 3.12+, an [OpenRouter](https://openrouter.ai) API key, and a
managed Qdrant cluster. Docker is optional for running Qdrant locally during development.

No GPU required. No Ollama. All LLM calls go out over HTTP to OpenRouter.

```bash
# 1. Optional local-development Qdrant (use your managed URL for deployment)
docker run -p 6333:6333 qdrant/qdrant

# 2. Install Python dependencies
pip install -e '.[dev]'

# 3. Set required environment variables (see below)
cp .env.example .env   # then fill in OPENROUTER_API_KEY

# 4. Seed SQLite and bootstrap the versioned corpus
python3 scripts/ingest_corpus.py --manifest corpus/v1.yaml

# 5. Start the API
uvicorn src.api.main:app --reload --port 8000
```

Required deployment environment variables:

```bash
OPENROUTER_API_KEY=<secret>                 # all OpenRouter calls
QDRANT_URL=https://<cluster>.<provider>     # managed Qdrant endpoint
QDRANT_API_KEY=<secret>                     # managed Qdrant credential
QDRANT_COLLECTION=rag_operational_v1        # active document corpus
QDRANT_SCHEMA_COLLECTION=rag_schema_meta_v1 # active schema metadata
DB_URL=sqlite+aiosqlite:///./operational.db # supported v1 relational DB
```

Store credentials with the platform secret manager (for example, Fly secrets), not in Git.
`QDRANT_API_KEY` may be empty only for an unsecured local-development Qdrant instance.

Optional variables (all have sensible defaults in `src/config.py`):

```bash
LOCAL_PROVIDER=openrouter          # or "ollama" for local dev with Ollama
LOCAL_MODEL=mistralai/mistral-nemo
ROUTER_MODEL=qwen/qwen3-30b-a3b-instruct-2507
CLOUD_MODEL=openai/gpt-oss-120b
GATE_MODEL=qwen/qwen3-30b-a3b-instruct-2507
LANGFUSE_PUBLIC_KEY=...            # optional: OTel traces in Langfuse
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://cloud.langfuse.com
PII_ENTITIES=PERSON,EMAIL_ADDRESS,PHONE_NUMBER,CREDIT_CARD,IBAN_CODE,AU_TFN,AU_MEDICARE
PII_ALLOW_LIST=Fairfield RSL,Fairfield_RSL,Fairfield,RSL
```

## Corpus bootstrap and version lifecycle

The corpus manifest is the human-readable deployment contract. `corpus/v1.yaml` pins the
collection names, embedding model/dimension, contextualization model, chunking settings,
document metadata, exact schema metadata, and SHA-256 checksums of every source file. Runtime
settings still own endpoints, credentials, the SQLite URL, and which already-bootstrapped
collections are active. This separation prevents a manifest from becoming a secret-bearing
deployment configuration file.

From a clean environment:

```bash
# Install the same project dependencies used by the API image.
pip install -e '.[dev]'

# Export the required variables listed above, then run:
python3 scripts/ingest_corpus.py --manifest corpus/v1.yaml

# Start the API with collection settings matching the manifest.
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

The command validates the manifest and source checksums before writes, seeds the three SQLite
tables and five views, creates missing Qdrant collections, indexes exactly the declared schema
views and documents, and validates both non-empty collections, all expected document source IDs,
and database connectivity. Its summary reports the corpus version, collection names, source
document count, chunk/point counts, schema state, and relational rows processed. It never deletes
an older collection. Bootstrap also requires the OpenRouter credential and fails before a document
upsert if any declared contextual prefix cannot be generated; it does not certify a partial corpus.

Document point IDs are UUIDv5 values derived from corpus version, source ID, page/chunk position,
and normalized content hash. Schema IDs use corpus identity plus view name. Therefore retrying the
same unchanged manifest upserts the same points instead of appending duplicates. Each point also
stores the manifest digest; bootstrap refuses a non-empty target collection with a different
version or digest. Contextual prefixes are LLM-generated and therefore are not claimed to be
byte-for-byte reproducible, but retries remain structurally idempotent and cannot mix manifests.

### Deploy a new corpus version

1. Copy `corpus/v1.yaml` to `corpus/v2.yaml`.
2. Set `corpus_version: v2`, use new names such as `rag_operational_v2` and
   `rag_schema_meta_v2`, update the source list/checksums, and review ingestion settings.
3. Run `python3 scripts/ingest_corpus.py --manifest corpus/v2.yaml`.
4. Check the successful bootstrap summary before changing application configuration.
5. Set `QDRANT_COLLECTION=rag_operational_v2` and
   `QDRANT_SCHEMA_COLLECTION=rag_schema_meta_v2`, then restart/redeploy the API.

Collection switching is deliberately configuration-based for v1. Do not reuse a v1 collection
name for changed sources, and do not delete v1 during the v2 bootstrap.

### Rollback

Set `QDRANT_COLLECTION=rag_operational_v1` and
`QDRANT_SCHEMA_COLLECTION=rag_schema_meta_v1`, then restart/redeploy the API. Because bootstrap
does not delete prior collections, rollback requires no re-ingestion. The SQLite demo dataset is
baked into the application image; roll back the image too if the relational fixture set changed.

### Intentionally deferred

- Full-stack Docker Compose reproducibility
- CI deploy-artifact and post-deploy smoke tests
- API authentication and ingestion authorization
- Retry/circuit-breaker policy
- Production relational database migration
- Monitoring and alerting improvements

## Runtime readiness

`/ready` validates the active runtime configuration without changing it. It checks SQLite
access and confirms that the collections named by `QDRANT_COLLECTION` and
`QDRANT_SCHEMA_COLLECTION` both exist and contain at least one point. It does not call
OpenRouter or any LLM, and it never creates, populates, or deletes Qdrant collections.

Bootstrap provisions a manifest-defined corpus; `/ready` only verifies that the already
provisioned corpus selected by the deployed environment is safe to serve.

```bash
# Ready deployment
curl -i http://localhost:8000/ready
# HTTP/1.1 200 OK
# {"status":"ready","checks":{"database":{"status":"ok"},"document_collection":{"status":"ok","collection":"rag_operational_v1","points":42},"schema_collection":{"status":"ok","collection":"rag_schema_meta_v1","points":5}}}

# A configured collection is absent, empty, or inaccessible
curl -i http://localhost:8000/ready
# HTTP/1.1 503 Service Unavailable
# {"status":"not_ready","checks":{"database":{"status":"ok"},"document_collection":{"status":"collection_missing","collection":"rag_operational_v2"},"schema_collection":{"status":"ok","collection":"rag_schema_meta_v1","points":5}}}
```

`/health` is unchanged and remains the existing broad dependency probe.

---

## Design Decisions

Full rationale with context, trade-offs, and benchmarks in `docs/design_decisions.md`.

- **Qdrant over pgvector/FAISS** — payload filtering, HNSW tuning, production durability justify operational overhead
- **FlashRank Nano over ColBERT/cross-encoder transformers** — 4MB CPU-only, sub-50ms, ~2pt nDCG cost acceptable
- **BM25 + dense hybrid retrieval (RRF k=60)** — operational text has exact identifiers (shift IDs, codes) where BM25 wins
- **text_to_sql safety design** — schema-linker, 3-candidate generation, self-correction (≤3 retries), DML + join-depth validator
- **No LangChain / LlamaIndex** — direct SDK (qdrant-client, openai, httpx); full observability per component
- **qwen3-30b router** — measurement-driven: 23/23 golden categories vs 21/23 for mistral-nemo; misroute yields confident wrong answer
- **qwen3-30b faithfulness gate** — measurement-driven: 5/5 on grounded/ungrounded probe; gpt-oss-120b scored 0/5 (over-rejects)
- **OpenRouter for all LLMs** — CPU-only container, no Ollama; LOCAL_PROVIDER=ollama still works for local dev
- **Semantic cache disabled by default** — `semantic_cache_url=""` is the off switch; avoids stale answers during evaluation

---

## Evaluation Results

Measured 2026-08-21 against the live Qdrant + OpenRouter stack. Judge model:
`openai/gpt-oss-120b`. Contexts truncated to 600 chars/citation to stay within
the model's output budget. See `docs/incident_log.md` for the grounded/ungrounded
probe that drove gate model selection.

| Metric | Threshold | Measured (30q full) |
|:-------|:----------|:--------------------|
| faithfulness | ≥ 0.9 | 0.65 ⚠ below threshold |
| answer_relevancy | ≥ 0.8 | 0.41 ⚠ below threshold |
| context_precision | ≥ 0.8 | 0.00 ⚠ below threshold |
| context_recall | ≥ 0.8 | 0.22 ⚠ below threshold |
| MRR | tracked | 0.28 |
| nDCG@10 | tracked | 0.27 |

> Live eval run: 2026-08-21, 30 golden QA pairs (23 with evidence ids for the
> retrieval metrics; the remaining 7 are unanswerable-by-design), Qdrant +
> `mistralai/mistral-nemo` + `openai/gpt-oss-120b`.

The full 30-question run is materially harder than the earlier 5-question smoke
(2026-08-15): answer_relevancy, context_recall, MRR, and nDCG@10 all drop
sharply, and `context_precision` remains 0.00 — indicating retrieved chunks
are not aligning with the ground-truth evidence ids in the golden dataset for
the majority of questions. This is the honest floor; see M6 known limitations
for the retrieval-quality workstream (contextual-prefix regeneration and BM25
tokenizer review) needed before publishing improvements.

---

## Load Test Results

Measured 2026-08-17: 200 requests, 50 concurrency, OpenRouter backend (qwen3-30b-a3b for routing, gpt-oss-120b for generation).
Latency is dominated by OpenRouter round-trip; local embed + rerank add < 200 ms.
p95/p99 values at exactly 30 s reflect the client-side timeout; 33/200 requests timed out at that ceiling.

| Metric | Simple | Complex | All |
|:-------|:-------|:--------|:----|
| p50 latency (ms) | 17 753 | 22 129 | 20 290 |
| p95 latency (ms) | 30 011 | 30 029 | 30 020 |
| p99 latency (ms) | 30 030 | 30 031 | 30 030 |
| throughput (req/s) | n/a | n/a | 2.5 |

> Load test run: 2026-08-17, 50 concurrent users, 200 total requests.
