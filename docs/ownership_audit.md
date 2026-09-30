# Ownership Audit — rag-operational

**Audit basis (2026-09-16):** source, tests, configuration, containers, workflow and checked-in documentation were inspected; protected environment/secret files and database/vector-store data were not read. **FACT** means directly verified in the repository. **INFERENCE** is a reasoned conclusion from those facts. No code was changed.

# 1. Executive Summary

**FACT:** This is a FastAPI application that answers venue-operations questions over three small fixture documents in Qdrant and three CSV-derived relational tables. It routes each question to document retrieval, generated SQL over curated SQLite views, or both; then generates, redacts, and optionally faithfulness-checks a cited answer. The implementation centres on `src/generation/query_handler.py:handle_query`.

**FACT:** The architecture is direct-SDK Python: FastAPI → Pydantic settings/app state → Qdrant/SQLite/OpenRouter → Streamlit UI. Ingestion is parser → word-window chunker → MD5 deduplication → optional LLM contextual prefix → MiniLM embeddings → Qdrant. Querying is PII redaction → optional in-memory semantic cache → LLM router → retrieval and/or text-to-SQL → LLM answer → faithfulness gate → output PII redaction.

**INFERENCE:** It is portfolio-demo ready only conditionally, not production ready. Core paths and a deployment workflow exist, but measured README evaluation quality is poor (full 30-question run: faithfulness .65, answer relevance .41, context precision .00, recall .22) and the deployed state depends on an externally managed, pre-populated Qdrant collection. The API has no authentication and the ingestion endpoint accepts arbitrary server paths.

Five strengths:

1. **FACT:** Clear end-to-end orchestration is concentrated in `handle_query`, and direct SDK use makes each integration inspectable.
2. **FACT:** Retrieval combines dense search, lexical BM25, RRF, FlashRank reranking, metadata filters, and citations (`src/retrieval/vector_search.py`, `reranker.py`).
3. **FACT:** SQL is constrained to five named views and rejects leading DML and join depth over three (`src/retrieval/text_to_sql.py`).
4. **FACT:** Input/output PII redaction, Prometheus metrics, optional Langfuse tracing, `/health`, and a Docker health check are implemented.
5. **FACT:** There is a meaningful test suite across units, route wiring, Qdrant-backed ingestion, and live-dependency evaluation code.

Five biggest risks:

1. **FACT:** `POST /ingest` takes `file_path` rather than an uploaded file and passes it to parsers; no path allow-list, authentication, authorization, or tenancy exists (`src/api/routes/ingest.py`).
2. **FACT:** Ingestion creates random Qdrant point IDs on every run, so deduplication is only within one document invocation; repeated/concurrent ingestion accumulates duplicates (`qdrant_store.py:upsert_chunks`).
3. **FACT:** “BM25” scrolls only `top_k * 4` points from Qdrant without pagination and runs BM25 in-process; it is not a Qdrant sparse index nor corpus-wide lexical retrieval (`vector_search.py:hybrid_search`).
4. **FACT:** SQL validation permits non-`SELECT` statements not beginning with its DML list (for example `PRAGMA`), interpolates LLM SQL into `EXPLAIN`, and relies on views by prompt/validator rather than database privileges (`text_to_sql.py`).
5. **FACT:** Qdrant, OpenRouter, Presidio, model loading, and LLM gate failures often fail open or are caught broadly; no request timeout/circuit breaker/retry policy applies consistently. The gate deliberately fails open on provider error (`faithfulness_gate.py`).

Personally understand first: (1) the concrete retrieval math and its incomplete BM25 implementation; (2) async FastAPI lifecycle plus CPU-bound model work; (3) SQL safety and SQLite/Postgres differences; (4) ingestion idempotency/data lifecycle; (5) evaluation interpretation and why current metrics do not substantiate quality.

# 2. System Architecture

**FACT — components and responsibility**

```text
Streamlit UI (app/streamlit_app.py) or HTTP client
                 |
FastAPI: /query, /ingest, /feedback, /health (src/api/main.py)
                 |
       app.state: Settings, Qdrant client, DB engine, tracer, cache
        |                         |                         |
        |                         |                         +--> SQLite by default / SQLAlchemy
        |                         +--> Qdrant: document + schema collections
        +--> OpenRouter or Ollama via src/llm_client.py

Upload-by-server-path --> parser --> chunker --> prefix LLM --> MiniLM --> Qdrant
Question --> PII --> cache --> router --> [hybrid retrieval and/or SQL] --> generator
         --> faithfulness LLM --> PII --> cited JSON

Prometheus /metrics and port 8001 ----> Prometheus ----> Grafana
Optional Langfuse tracer ----> Langfuse (Postgres + ClickHouse sidecars)
```

The API is the composition root. Its lifespan in `src/api/main.py:lifespan` makes `AsyncQdrantClient`, `RAGTracer`, `SemanticCache`, and an async SQLAlchemy engine available through `app.state`. Routes import implementation inside handlers, avoiding the eager package-import cycles warned about in repository policy. The only persistent application data explicitly seeded by code are Qdrant points and the relational database. OpenRouter is the default LLM provider; Ollama is an alternate development provider.

External dependencies: Qdrant; OpenRouter-compatible chat API (or Ollama); Hugging Face/SentenceTransformers and FlashRank local model files; SQLite by default (or a DB URL that SQLAlchemy supports); optional Langfuse, Prometheus/Grafana, DeepEval/RAGAS/W&B. **INFERENCE:** the current deployment architecture is split: `fly.toml` deploys only the API and expects `QDRANT_URL` externally, while `infra/docker-compose.aws.yml` describes observability/Qdrant sidecars but no API service.

# 3. End-to-End Data Flow

## A. Document upload

**FACT:** This is not a binary upload. `POST /ingest/` SSE and `POST /ingest/sync` accept `IngestRequest(file_path, doc_type, venue, date, shift_id)` in `src/api/routes/ingest.py`. `_progress_stream` or `ingest_sync` constructs `Embedder(settings.embedding_model)` per request and calls `ingest_document`.

`src/ingestion/pipeline.py:ingest_document` chooses `parse_text` only for `doc_type == "shift_log"`; every other type is parsed as PDF. `parser.py:parse_pdf` calls `pymupdf4llm.to_markdown(..., page_chunks=True)` and flags markdown-table syntax. `parse_text` splits a UTF-8 file on blank lines. `chunker.py:chunk_pages` splits on whitespace into 512-word windows by default with 64-word overlap; it never crosses parser page/block boundaries and keeps table-flagged content whole. This is word count, not tokenizer count.

`anchors.py:annotate_chunks` adds evaluation-specific section anchors. `dedup.py:deduplicate` normalizes case/edge Unicode then uses MD5, only over chunks in this invocation. For each surviving chunk the pipeline awaits `context_generator.py:generate_contextual_prefix`; missing key or API failure returns an empty prefix. `bm25_builder.py:add_bm25_text` prepares text. `embedder.py:embed_chunks` batches contextual-prefix-plus-content using CPU SentenceTransformers. `qdrant_store.py:ensure_collection` creates named dense vector `dense` at 384 dimensions/cosine/HNSW m=16, ef_construct=100; `upsert_chunks` writes payloads and random UUID point IDs in batches of 100. The response returns only document type, count upserted, and count removed in the in-call dedupe.

No document bytes, file metadata record, job status, source checksum registry, or upload audit row is persisted. **INFERENCE:** retries or two workers can create duplicate semantic records; a partial failure after earlier batches leaves partial ingestion.

## B. Query / retrieval

**FACT:** `POST /query/` validates only non-blank `query`; accepts unconstrained `filters: dict | None` (`src/api/routes/query.py`). `handle_query` redacts query PII before cache lookup. It calls `SemanticCache.get`; when enabled this is a process-local list, embeds again in a thread executor, uses cosine >= .95, and has no TTL, max size, external backend, or invalidation.

On a miss, `router.py:classify_intent` asks the configured router model for JSON `{tool: vector_search|text_to_sql|hybrid, complexity: simple|complex}` at temperature zero; parse/provider failure defaults to costly `hybrid/complex`.

For vector/hybrid paths, `retrieval/pipeline.py:vector_search_pipeline` calls `expand_query` (original plus up to three LLM expansions), embeds every expansion, invokes `hybrid_search`, deduplicates by payload content hash, then `rerank` runs FlashRank in a two-worker `ThreadPoolExecutor`. Defaults: each search asks top 50, obtains up to 200 dense and 200 scrolled records, RRF merges at k=60, reranker returns ≤7 only if the maximum score reaches .3. Complex requests grade each returned chunk with an LLM, accepting if at least half say “yes”, reformulating up to five iterations / a nominal 4,000-token budget.

`vector_search.py` creates `Filter(**filters)` directly. It runs dense Qdrant `query_points(using="dense")`; it then scrolls a bounded subset and runs `BM25Okapi` locally, RRF-merges ranks, and emits payload fields. Thus filtering applies to both branches, but lexical ranking does not span the whole collection.

For SQL/hybrid paths, it embeds the query, retrieves up to three schema metadata points from `rag_schema_meta`, calls `text_to_sql_tool`, and supplies rows plus document chunks to `build_hybrid_prompt`; otherwise it uses `build_vector_rag_prompt`. Complex/hybrid requests go through `generate_cloud`; simple cases use `generate_local`. `check_faithfulness` receives only vector chunks (not SQL rows), so SQL-only answers skip grounding because `context_chunks` is empty. Citations are response metadata made before generation, not parsed from model citations.

## C. Structured extraction

**FACT:** No generic extraction endpoint, Pydantic structured LLM output, or entity-extraction pipeline exists. The nearest structured path is natural-language-to-SQL: `generate_sql` requests three fenced SELECT candidates, regex-extracts code, `validate_sql` blocks a leading DML keyword / >3 joins and runs `EXPLAIN`, then `execute_sql` returns dictionaries. The validation is syntactic and execution-based; it does not validate result semantics, SQL view membership, limits, read-only credentials, or schema types. PII validation is separate redaction in `guardrails/pii_redactor.py`.

## D. Startup / dependency initialization

**FACT:** `Settings()` loads `.env` through Pydantic Settings (contents not inspected). The FastAPI lifespan initializes clients but does not verify connections, create collections, seed DB, index schemas, warm MiniLM/FlashRank/Presidio, or run migrations. `scripts/ingest_corpus.py` is the required separate initialization: it seeds CSVs, creates views, indexes five hard-coded schema descriptions, and ingests three fixture documents. Docker builds SQLite from fixtures, but does not populate Qdrant.

# 4. Repository Map

| Area | Responsibility / dependents | Assessment |
|---|---|---|
| `src/api/` | HTTP schema/routes and lifespan; all clients enter here | Essential; route modules are thin and understandable. |
| `src/ingestion/` | File parsing, chunking, contextualization, embeddings, Qdrant, SQLite seed | Essential; document pipeline is modular, but no durable job/source model. |
| `src/retrieval/` | query embeddings, cache, hybrid retrieval, rerank, schema retrieval, SQL | Essential; logic is readable but “hybrid” is misleadingly local/bounded BM25. |
| `src/generation/` | router, prompts, providers, end-to-end handler, gate | Essential; `query_handler` is the real service layer and is tightly coupled to orchestration. |
| `src/guardrails/` | Presidio/staff-ID redaction | Essential for stated PII claim; no per-tenant policy. |
| `src/observability/` | Prometheus metrics and optional Langfuse wrapper | Useful; tracer API is simple but not demonstrably OTel despite documentation. |
| `src/evaluation/` + `eval_data/` | RAGAS calculations and 30 golden cases | Useful; live judge dependency makes reproducibility weak. |
| `tests/unit` | mostly isolated function/module behavior | Valuable, but many mocks verify calls rather than real model/service behavior. |
| `tests/integration` | routes/handler and actual Qdrant ingestion | Essential, though many “integration” tests mock dependencies. |
| `tests/eval` | parametrized live DeepEval gate | High-value intent, but it is not run by CI. |
| `scripts/` | corpus bootstrap, eval/load tests, failure demonstrations | Useful operational evidence; demos duplicate pipeline control flow and write ignored logs. |
| `app/streamlit_app.py` | manual UI to query, feedback, server-path ingest | Optional demo surface; `streamlit` and `requests` are not declared in `pyproject.toml`. |
| `infra/` | local/basic Qdrant+Ollama compose; AWS observability compose | Useful but split and inconsistent with Fly/OpenRouter deployment. |
| `Dockerfile`, `fly.toml` | API image and Fly configuration | Essential deployment assets; neither provisions Qdrant data. |
| `.github/workflows/ci.yml` | lint, unit tests, image push, Fly deploy | Essential but incomplete quality gate. |
| `README.md`, `PRD.md`, `PROJECT_STATUS.md`, `docs/` | product/design/history | Important but materially stale in places: PRD describes 150–200 docs/DOCX and different views/models; status claims differ from checked-in code. |

# 5. ML / RAG Audit

| Choice | Current implementation (FACT) | Trade-off / alternatives / judgement (INFERENCE) |
|---|---|---|
| Embeddings | `all-MiniLM-L6-v2`, 384-D; ingest CPU wrapper, query singleton auto-selects CUDA unless overridden. `embedding_fallback` is configured but unused. | Small and inexpensive. Separate wrappers/device policy risk behavioral drift. A domain/e5/BGE model or managed embeddings may improve quality; current is reasonable for tiny demo corpus, unsupported by current low retrieval scores for quality claims. |
| Chunking | 512 whitespace words, overlap 64; parser pages/blocks preserve boundaries; tables remain one unsized chunk. | Cheap and transparent but not token/semantic/section-aware. Alternatives: tokenizer-aware recursive/heading chunks, parent-child chunks, table serialization. Reasonable baseline, not robust for long tables or model contexts. |
| Contextualization | Sequential one LLM call per deduped chunk, max source 1,000 chars; failure returns empty string. | Can add document context but increases latency/cost and produces mixed-quality indexed corpus on failures. Batch/concurrency, stored version/provenance, or deterministic metadata prefix would be safer. |
| Dense + lexical | Dense Qdrant + in-memory `rank_bm25` over bounded `scroll(limit=top_k*4)`; RRF k=60. | Exact term support is sensible. This does not scale/correctly rank a full collection; native Qdrant sparse vectors, Elasticsearch/OpenSearch, or corpus-wide maintained BM25 would be stronger. |
| Reranking | FlashRank `ms-marco-MiniLM-L-12-v2`, max length 512, global ranker, two thread workers, threshold .3. | Good CPU demo choice. Threshold is uncalibrated in code; reranker sees content only, losing metadata/prefix. |
| Prompt/generation | prompts directly concatenate up to seven chunks/all SQL rows. Simple uses configured local provider; complex/hybrid OpenRouter cloud model. | Explicit and inspectable; missing prompt length budgets, escaping, prompt-injection isolation, JSON structured output, and SQL-row limits. |
| Grounding | Prompt asks answer only from context; binary LLM gate checks ≤5 vector chunk contents. Gate skips with no key/chunks and fails open on error. | Better than no check, but not a proof and does not check SQL facts. Fail-closed for sensitive routes, claim/citation validation, and answer abstention calibration are alternatives. |
| Evaluation | RAGAS runner computes four judge metrics plus MRR/nDCG; DeepEval test runs live stack; README records poor live result. | Golden set is good practice. Live nondeterministic proprietary judge/model, no pinned corpus snapshot, and CI omission mean quality is not release-enforced. |
| Resilience/cache | query expansion catches provider errors; gate fails open; local generation returns an error-looking string; cache disabled by default and local only. | Availability is favored over correctness. Add typed errors/timeouts/retries/backoff/rate limits and external TTL cache for production. |

**FACT:** `generate_sql` is free-form text parsing, not structured output. `generate_contextual_prefix` uses an LLM but no extraction schema. No extraction-specific evaluation exists.

# 6. Backend / Software Engineering Audit

**FACT:** Layering is intuitive (API → handler/pipeline → adapters) without a repository/service abstraction. FastAPI dependency injection is not used; dependencies are global app state and are passed manually only below `handle_query`. That makes route setup simple but couples tests to mutating global `app.state`.

**FACT:** Pydantic validates request shape but `filters` is arbitrary and ingest metadata dates/path/doc type are barely validated. Broad `except Exception` in all routes returns details to callers (`query.py:query`, `ingest.py:ingest_sync`, `feedback.py:feedback`), leaking internal provider/filesystem information. CORS permits all origins in `main.py`.

**FACT:** Networking is mostly async. CPU SentenceTransformers at query time runs synchronously inside `async def embed_query`, blocking the event loop; ingestion `Embedder.embed_batch` also blocks it. FlashRank and cache embedding are moved to executor threads. Creating an `Embedder` and loading a model for every ingest request is expensive. **INFERENCE:** Fly’s 150 soft / 200 hard request concurrency conflicts with local CPU model use and multiple multi-call LLM workflows.

**FACT:** `src/llm_client.py` is a positive consolidation of OpenRouter/Ollama calls, but it instantiates a new `AsyncOpenAI` client for each request and has no explicit OpenRouter timeout/retry policy. `local_llm.py` converts errors to a string that may be sent to the user and then potentially pass/fail a gate; cloud errors throw. This is inconsistent failure semantics.

**FACT:** Config uses Pydantic and no literal secrets are present in inspected files. However unpinned dependencies include `uvicorn`, `pydantic`, Qdrant, SQLAlchemy, model libs, OpenAI, and all eval packages; Docker/compose images use `latest` in several places. **INFERENCE:** reproducibility is weaker than the README’s production framing.

# 7. Database and Data Engineering Audit

**FACT:** Default `db_url` is `sqlite+aiosqlite:///./operational.db`. `structured_loader.py:seed_database` creates `shift_records`, `incident_logs`, and `staffing_history`, each with a text primary key. It reads three fixture CSVs and executes SQLite-specific `INSERT OR IGNORE`. `create_views` makes five views: `v_shift_summary`, `v_incident_by_venue`, `v_staffing_gap`, `v_incident_severity`, `v_peak_staffing`.

There are no ORM models, migrations, foreign keys, secondary indexes, constraints beyond primary keys, timestamps, ingestion tables, transaction across all CSVs/views, or backup/restore configuration. Each table seed runs its own transaction; view creation is another. **INFERENCE:** a PostgreSQL DB URL will not necessarily work because `INSERT OR IGNORE` is SQLite syntax and views/typing have not been tested cross-database.

Rows are cited using IDs in `anchors.py:row_anchor`; aggregate views often lack source IDs, so citations can have no evaluation anchor. There is idempotence only for CSV primary keys. No worker lock or source versioning exists. Concurrent seeders can race around DDL/insert; concurrent document workers are worse because Qdrant IDs are random.

Own SQL/data topics: SQLAlchemy async engine lifecycle; SQLite locking and `INSERT OR IGNORE`; view semantics; read-only database users; query planning/indexes; migrations; data lineage from fixture row through view/citation; and transaction/idempotency design.

# 8. Vector Database Audit

**FACT:** `rag_operational` is created lazily with one named vector, `dense`, 384 dimensions, cosine distance, HNSW m=16/ef_construct=100. Payload includes content, contextual prefix, BM25 text, page/doc metadata, MD5 content hash, and evaluation anchors. Qdrant payload indexes are not created. Inserts are random UUID upserts; no delete/update-by-source exists. `rag_schema_meta` is a separate unnamed-vector cosine collection populated with deterministic UUID5 IDs for five hard-coded view descriptions.

Queries use `query_points` for dense and `scroll` for local BM25; optional raw Qdrant `Filter(**filters)` is forwarded. Persistence is Qdrant’s volume in compose or an unverified external `QDRANT_URL` in Fly. Failure bubbles to routes except schema linking returns `[]`; no collection readiness at startup.

Why Qdrant here: it provides an async client, payload filtering and durable vector storage independently of the relational SQLite database. pgvector could replace the document collection if the project standardizes on Postgres: one database, ACID source/embedding lifecycle, SQL metadata filters/joins and simpler backup/deployment. Qdrant is better if the corpus grows into high-throughput ANN/vector-specific operations and Qdrant-native sparse/hybrid indexes are actually used. **INFERENCE:** with this repository’s tiny corpus, local client-side BM25, SQLite default, and no Qdrant indexes beyond HNSW, Postgres + pgvector would likely be simpler operationally; current Qdrant is defensible as a portfolio demonstration, not yet justified by implementation scale.

# 9. Testing Audit

**FACT:** Categories are unit tests (`tests/unit`, roughly parser/chunker/dedup/embedder/Qdrant/RRF/rerank/router/LLM/gate/PII/SQL), integration tests (routes, handler, actual Qdrant ingestion), smoke (`tests/test_smoke.py`), and optional/live evaluation (`tests/eval/test_deepeval_gate.py`). `tests/conftest.py` uses a configured live Qdrant and creates/deletes `test_*` collections; it deletes `test_operational.db` at session end.

Most handler/route tests patch LLM/retrieval functions or use `AsyncMock`; they validate wiring and response transforms, not deployed answer quality. Qdrant ingestion tests load actual MiniLM and require Qdrant. Unit vector-search tests mock Qdrant; there is no real corpus-wide BM25 test. No Docker image test, compose smoke, Fly deploy smoke, auth/security test, request cancellation/timeout test, concurrent ingest test, SQL injection/read-only-role test, or end-to-end seeded-corpus API test.

**Verification result:** the repository sensor command, `bash .agent-context/scripts/run_tests.sh`, failed during pytest configuration in this sandbox because `pytest-rerunfailures` attempted a TCP socket and got `PermissionError`; no test collection or execution result was obtained. This neither proves a code failure nor validates the claimed passing counts.

Ten tests to own: `test_ingest_pdf_happy_path`, `test_ingest_missing_file`, `test_rrf_merges_results`, `test_dedup_by_content_hash`, `test_hard_reject_below_threshold`, `test_self_correction_tries_next_candidate`, `test_vector_query_happy_path`, `test_tabular_query_path`, `test_gate_open_on_api_error`, and parametrized `test_deepeval_gate`.

# 10. Docker / Infrastructure Audit

**FACT:** Dockerfile is multi-stage Python 3.12 slim. It installs CPU Torch, the package, bakes MiniLM/FlashRank/spaCy `en_core_web_lg`, copies source/fixtures/scripts/eval data, seeds SQLite at build, becomes non-root, exposes 8000 and healthchecks `/health`. It does not ingest documents into Qdrant. `infra/docker-compose.yml` runs Qdrant latest and Ollama with named volumes; it contains no API/Streamlit/Prometheus service. `docker-compose.aws.yml` has Qdrant, Langfuse, Postgres, ClickHouse, Prometheus, Grafana but no health checks, API, env file contract, or explicit network.

Likely failures: API health is permanently degraded if external Qdrant is missing/unreachable; Docker-created SQLite exists but Qdrant has no collection/data; Prometheus targets `host.docker.internal:8001`, which is platform-sensitive and mismatches a containerized API; Langfuse needs undeclared compose variables; Compose has startup ordering only for Langfuse, not readiness; Qdrant latest risks API drift; Fly’s 2 GB VM must fit Python, spaCy large model, MiniLM and FlashRank; auto-stop can make model cold starts visible. Model files are baked, which avoids runtime model download, but increases build time/image size and relies on network during build.

# 11. CI/CD Audit

**FACT:** `.github/workflows/ci.yml` triggers push/PR to main. `lint` installs/runs Ruff. `test` waits for lint, runs a Qdrant service, installs `.[dev]`, and runs only `pytest tests/unit/ -v`; OpenRouter key is made available but unit tests are mocked. On main, `build` pushes image to GHCR, then `deploy` uses `FLY_API_TOKEN` to `flyctl deploy` a SHA image.

No workflow step runs integration tests, live DeepEval/RAGAS, Docker container smoke tests, dependency vulnerability scanning, type checking, migration/seed/Qdrant initialization, or post-deploy verification/rollback. The README/PRD claim eval-in-CI; the checked-in workflow does not do it. To explain it in an interview, distinguish CI’s real gate (lint + unit test) from desired quality/release gates and explain GHCR SHA immutability vs the absent runtime data deployment.

# 12. Deployment / Production Readiness

| Area | Status | Evidence |
|---|---|---|
| Demo API/image | GREEN | FastAPI, non-root image, docs and Fly manifest exist. |
| Corpus bootstrap/reproducibility | RED | Image seeds SQLite only; Qdrant requires an out-of-band ingest. |
| Secrets | YELLOW | env-based configuration/GitHub secrets; no secret values inspected, but no documented Fly secret provisioning or rotation. |
| Health/readiness | YELLOW | `/health` checks Qdrant/DB; it does not check OpenRouter, collection/data, or model readiness. |
| Authentication/authorization | RED | No auth and permissive CORS. |
| Scaling/concurrency | RED | 150/200 Fly limits with synchronous CPU embedding and external multi-call LLMs; no load shedding. |
| Timeouts/retries/rate limits | RED | inconsistent 60-second calls, no standard retry/backoff/circuit breaker/rate limit. |
| Persistence/backups | RED | volumes exist locally; no backup, Qdrant lifecycle, migration, or restore plan. |
| Evaluation quality | RED | checked-in README reports low quality; CI does not enforce it. |
| Observability | YELLOW | metrics/traces/health exist but alerting, trace completeness and production scrape topology do not. |
| Cost controls | YELLOW | nominal retrieval token budget and cost estimate exist; no provider usage accounting, quotas, or per-request LLM caps. |

# 13. Observability and Monitoring

**FACT:** Prometheus instrumentator exposes framework metrics. `metrics.py` adds retrieval confidence, reranker score metric (defined but no increment found), chunks/query, route counter, estimated cost summary, component latency, cache hit/miss, gate failures, agent iterations, SQL errors, and PII redactions. `query_handler` observes route/retrieval/SQL/generation latency; `pipeline.py` observes iterations/chunks/confidence. `RAGTracer` is a noop unless Langfuse keys exist and emits start/end/spans/scores. `/health` exists.

Missing or unverified: structured JSON logs (no logging setup), request/correlation ID at HTTP boundary, input validation/error metrics by reason, LLM token/provider/actual cost metrics, cache size, Qdrant collection counts, alert rules, SLOs, dashboards tested against actual metric names, trace propagation into Qdrant/OpenRouter, and safe logging policy. **FACT:** there is no alerting configuration in `infra/prometheus/prometheus.yml`.

# 14. Security Audit

**FACT:** No inspected source contains a literal credential; Settings reads env values. PII redaction detects selected Presidio entities plus `STF-` IDs and runs pre-cache/pre-trace and post-generation. Risks: unauthenticated API, wildcard CORS, user-controlled server file path, unbounded file size/parser work, error details returned to callers, raw arbitrary `filters`, prompt text directly concatenated into router/SQL/generation prompts, SQL validation without read-only DB role, and SQL text returned in API citations/UI.

Prompt injection can originate in retrieved document content because it is treated as prompt context without delimiters/instruction hierarchy hardening. PII can remain in Qdrant payload/SQLite and be sent to OpenRouter during contextualization/generation; redaction only surrounds user query and final answer. For a student portfolio, prioritize clearly documenting synthetic data, locking `/ingest`, restricting origins, and using read-only database credentials before claiming enterprise safeguards.

# 15. Dead Code / Duplication / Overengineering

| Item | Finding | Action classification |
|---|---|---|
| `embedding_fallback`, `anthropic_api_key`, `openai_api_key` | configured but no production source usage found | Actively confusing; simplify eventually. |
| `gptcache` dependency | cache is custom in-memory list; no import found | Actively confusing/risky dependency claim. |
| Streamlit/requests | app imports both, neither declared in project dependencies | Actively confusing build/runtime gap. |
| `RAGTracer.log_span` | `latency_ms` accepted but not passed to Langfuse span API | Should eventually simplify/fix; observability claim is incomplete. |
| `rag_reranker_score_max` | defined/exported; no increment found | Actively confusing metric. |
| `db_url` portability | SQLite-specific loader despite “SQLite/Postgres” documentation | Actively confusing production claim. |
| `schema_indexer` `db_url` arg | unused | Safe to leave but unnecessary. |
| demo scripts | reimplement private retrieval logic / write ignored logs | Safe experiments, but duplicate behavior can drift. |
| README/PRD/status | contradictory models, corpus sizes, CI/eval claims | Actively confusing ownership risk. |
| Compose variants | one stale Ollama dev topology and one observability topology; neither is full app stack | Should eventually simplify. |

# 16. Ownership Gap Map

## GREEN — Must already be able to explain

FastAPI route/request-response Pydantic models; the three pipeline routes; MiniLM embeddings and cosine similarity; basic chunking/overlap; Qdrant payloads; RRF and reranking sequence; SQL views; Docker image versus Compose; pytest mocks/fixtures; and the query-handler data flow.

## YELLOW — Need deeper understanding

Python import/lifecycle behavior and `app.state`; async I/O versus CPU-bound inference and thread executors; SentenceTransformers model/device caching; HNSW settings; BM25 tokenization/recall boundaries; prompt construction/token budgets; SQLAlchemy async engines; SQLite locking versus PostgreSQL; Pydantic settings/secrets; Prometheus metric types; Langfuse no-op behavior; Docker layers/model baking; and CI artifact/deploy promotion.

## RED — High-priority learning gaps

RAG evaluation design and judging bias; retrieval relevance debugging and corpus/evidence versioning; real native hybrid sparse retrieval; SQL injection/least privilege and semantic text-to-SQL validation; idempotent distributed ingestion/concurrency; rate limiting/timeouts/retry/circuit breakers; API auth/authorization/file-upload threat modeling; production database migrations/backups; capacity planning for LLM fan-out; observability SLOs/alerting; and explaining the discrepancy between documented architecture and actual implementation without overclaiming.

# 17. Interview Questions Based on This Repo

## Architecture questions

1. Walk through `POST /query/` to its final cited response.
2. Why is `handle_query` the orchestration boundary rather than a FastAPI dependency?
3. Why are Qdrant/DB/tracer/cache held on `app.state`?
4. What happens at process startup before the first query?
5. Why do routes import pipeline modules inside functions?
6. Why is Qdrant separate from the relational database here?
7. What differs between Docker, Compose, and Fly topology?
8. How would a Qdrant outage affect health and query behavior?
9. What exact persistence is built into the container image?
10. Which checked-in docs would you correct before presenting this architecture?

## ML/RAG questions

1. Why all-MiniLM-L6-v2 and why is the vector dimension 384?
2. Is `chunk_size=512` tokens or words, and why does it matter?
3. Explain the RRF calculation in `_rrf_merge` and k=60.
4. Is the BM25 implementation actually corpus-wide hybrid search?
5. What role does contextual prefix generation play and what happens on failure?
6. Why rerank after expansion and RRF, and what does .3 reject?
7. How does the complex retrieval grader loop terminate?
8. How are evaluation section anchors produced and why?
9. What does the faithfulness gate verify, and what does it miss for SQL answers?
10. How do the README’s .00 context precision and low recall change your roadmap?

## Production/MLOps questions

1. What is and is not validated by `/health`?
2. How are OpenRouter credentials provisioned locally, CI, and Fly?
3. Why could 200 Fly request concurrency be unsafe?
4. What happens on a cold start with baked versus missing model files?
5. Which CI checks run on PRs and which claimed checks do not?
6. How would you deploy a new corpus safely and reversibly?
7. Which Prometheus metrics are actually updated versus merely declared?
8. Why is a process-local semantic cache unsuitable across replicas?
9. How would you back up and restore both SQL and Qdrant state coherently?
10. How would you measure actual per-query LLM cost instead of an estimate?

## Debugging/failure scenario questions

1. Why might repeated ingestion return duplicate citations?
2. What happens if OpenRouter is down during routing, generation, and gate checks?
3. Why may `POST /ingest` expose host files?
4. What happens if Qdrant starts empty after a Fly deploy?
5. What breaks if `db_url` changes to Postgres?
6. Why might BM25 miss an exact term in a large collection?
7. What happens when a long table exceeds the generation context?
8. How could a valid SQL query still answer the wrong question?
9. Why did the sandbox test suite not execute, and how would you distinguish that from a code failure?
10. What would you inspect first after a 30-second load-test timeout?

# 18. Hands-On Ownership Exercises

1. **Objective:** Trace one vector query on paper. **Files:** query route, handler, retrieval pipeline. **Difficulty:** easy. **Outcome:** explain every input/output object.
2. **Objective:** Reimplement `_rrf_merge` with three fake hits. **Files:** `vector_search.py`. **Difficulty:** easy. **Outcome:** rank fusion intuition.
3. **Objective:** Compare 512-word chunks with tokenizer counts for fixtures. **Files:** chunker/fixtures. **Difficulty:** easy. **Outcome:** chunk-unit trade-off.
4. **Objective:** Add a throwaway test locally for one filter shape (do not retain it). **Files:** vector search tests. **Difficulty:** medium. **Outcome:** Qdrant filter schema understanding.
5. **Objective:** Seed a fresh temporary SQLite DB and explain each curated view. **Files:** structured loader/CSVs. **Difficulty:** medium. **Outcome:** SQL lineage.
6. **Objective:** Write an oracle SQL query for a golden question and compare it to generated SQL. **Files:** demo_bad_sql_audit. **Difficulty:** medium. **Outcome:** semantic SQL validation.
7. **Objective:** Manually calculate MRR/nDCG for five citations. **Files:** ragas runner/golden dataset. **Difficulty:** medium. **Outcome:** retrieval eval literacy.
8. **Objective:** Simulate empty retrieval with the existing handler test pattern. **Files:** integration query-handler test. **Difficulty:** easy. **Outcome:** abstention path ownership.
9. **Objective:** Simulate gate API failure and explain fail-open implications. **Files:** faithfulness gate tests. **Difficulty:** easy. **Outcome:** safety/availability trade-off.
10. **Objective:** Draw the exact container/data dependency graph without README. **Files:** Dockerfile/Compose/Fly. **Difficulty:** medium. **Outcome:** deployment accuracy.
11. **Objective:** Run a small load test against a local live stack and inspect component latency metrics. **Files:** load_test/metrics. **Difficulty:** hard. **Outcome:** bottleneck diagnosis.
12. **Objective:** Trace a PII string through query, cache, trace, model prompt, and answer. **Files:** PII redactor/handler. **Difficulty:** medium. **Outcome:** data-leak reasoning.
13. **Objective:** Reproduce duplicate ingestion in an isolated Qdrant collection. **Files:** pipeline/qdrant store. **Difficulty:** hard. **Outcome:** idempotency ownership.
14. **Objective:** Rebuild the API Dockerfile from memory, then compare layers. **Files:** Dockerfile. **Difficulty:** medium. **Outcome:** image/runtime fluency.
15. **Objective:** Write a release checklist separating code image, SQLite seed, Qdrant corpus, and secrets. **Files:** CI/Fly/ingest script. **Difficulty:** hard. **Outcome:** operational ownership.

# 19. Top 10 Learning Priorities

1. **Retrieval evaluation.** README metrics are weak; internship-level means explaining MRR/nDCG/precision/recall and designing a reproducible offline set.
2. **Hybrid retrieval correctness.** `vector_search.py` only BM25-scores a scroll subset; explain/repair conceptually using native sparse search or a lexical index.
3. **Async/concurrency.** CPU embedding occurs in async calls and Fly allows high concurrency; explain blocking and bounded executors.
4. **Text-to-SQL security.** `text_to_sql.py` needs read-only privilege and allow-list reasoning; internship level is identifying why prompt constraints are insufficient.
5. **Ingestion idempotency.** Random IDs/source-less Qdrant points make re-ingestion unsafe; explain deterministic IDs and source versioning.
6. **RAG grounding.** Gate ignores SQL rows/fails open; explain claim support, abstention, and error policy.
7. **Data engineering.** Loader/views are SQLite-specific; explain views, transactions, migrations and PostgreSQL portability.
8. **Deployment topology.** Fly API and external Qdrant split; explain stateful services, bootstrap, backups and config.
9. **Testing strategy.** CI runs units only; explain mock versus integration/e2e and deterministic model test seams.
10. **Security/observability.** File-path ingest/auth/CORS and metrics/traces need a practical threat and monitoring story.

# 20. Definition of Done for v1.0

Minimum portfolio-ready v1.0 should demonstrate one reproducible seeded corpus (versioned manifest and deterministic Qdrant lifecycle), documented `POST /query` answers across vector/SQL/hybrid paths, locked-down or removed demo ingestion, and a README that accurately names corpus size/models/topology/current measured metrics. It should have a reproducible Docker/Compose command that starts API plus all required state or documents managed endpoints precisely; health/readiness that verifies initialized collections/data; CI that runs lint, unit tests, required integration/container smoke tests, and a deterministic quality regression check; a small documented evaluation with honest thresholds/results; basic production-safe configuration/secrets guidance; and a demo script/UI flow. Keep scope to these foundations—do not add agent frameworks, additional models, or features before quality, safety, and reproducibility are credible.
