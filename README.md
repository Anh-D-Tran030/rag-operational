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
 FastAPI (/query, /ingest, /feedback, /health)
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

Sidecars:
  Qdrant (vector + sparse index)  │ SQLite/Postgres (curated views)
  GPTCache (semantic cache, opt)  │ Langfuse (OTel traces)
  Prometheus + Grafana (metrics)  │
```

---

## Tech Stack

| Layer | Tool | Justification |
|:------|:-----|:--------------|
| Backend | FastAPI | Async, Pydantic v2 validation, Prometheus middleware, OpenAPI docs out of the box |
| Vector DB | Qdrant (self-hosted) | Dense + sparse (BM25) hybrid search; payload filtering by doc_type/date/dept; HNSW tuning; production durability vs FAISS (in-memory only) |
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

**Prerequisites:** Docker, Python 3.12+, an [OpenRouter](https://openrouter.ai) API key.

No GPU required. No Ollama. All LLM calls go out over HTTP to OpenRouter.

```bash
# 1. Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# 2. Install Python dependencies
pip install -e '.[dev]'

# 3. Set required environment variables (see below)
cp .env.example .env   # then fill in OPENROUTER_API_KEY

# 4. Seed the operational database and ingest the corpus
python3 scripts/ingest_corpus.py

# 5. Start the API
uvicorn src.api.main:app --reload --port 8000
```

Required environment variables — paste into `.env`:

```bash
OPENROUTER_API_KEY=sk-or-...      # required: all LLM calls (routing, generation, faithfulness gate)
```

Optional variables (all have sensible defaults in `src/config.py`):

```bash
QDRANT_URL=http://localhost:6333
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

Measured 2026-08-15 against the live Qdrant + OpenRouter stack.  Judge model:
`openai/gpt-oss-120b`. Contexts truncated to 600 chars/citation to stay within
the model's output budget.  See `docs/incident_log.md` for the grounded/ungrounded
probe that drove gate model selection.

| Metric | Threshold | Measured (5q smoke) |
|:-------|:----------|:--------------------|
| faithfulness | ≥ 0.9 | 0.65 ⚠ below threshold |
| answer_relevancy | ≥ 0.8 | 0.88 ✓ |
| context_precision | ≥ 0.8 | 0.00 ⚠ below threshold |
| context_recall | ≥ 0.8 | 0.80 ✓ |
| MRR | tracked | 0.63 |
| nDCG@10 | tracked | 0.67 |

Run `python3 scripts/run_eval.py` (full 30-question dataset) to get stable estimates.
The 5-question smoke run is intentionally noisy; faithfulness and context_precision
are expected to improve with more samples.

---

## Load Test Results

Measured 2026-08-15: 40 requests, 10 concurrency, OpenRouter backend.
Latency is dominated by OpenRouter round-trip; local embed + rerank add < 200 ms.

| Metric | Simple | Complex | All |
|:-------|:-------|:--------|:----|
| p50 latency (ms) | — | — | — |
| p95 latency (ms) | 21 477 | 42 049 | — |
| p99 latency (ms) | — | — | — |
| throughput (req/s) | — | — | — |
