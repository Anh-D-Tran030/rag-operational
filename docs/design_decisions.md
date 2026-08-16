# Design Decisions

Eight architectural decisions made during the build of the Enterprise Operational RAG System,
with context, the decision taken, and the trade-offs accepted.

---

### 1. Qdrant over pgvector / FAISS

**Context:**
Three viable vector-store options were evaluated: Qdrant (self-hosted on EC2), pgvector
(Postgres extension), and FAISS (in-process library). The workload requires payload filtering
by `doc_type`, `department`, and `date_range` on every query — not just vector similarity.

**Decision:**
Qdrant on EC2 (`localhost:6333` in dev, EC2 in prod).

**Trade-offs:**
- Qdrant provides native payload filtering combined with HNSW vector search in a single query;
  pgvector requires a compound SQL predicate that can degrade to a full table scan without careful
  index design, and FAISS has no native metadata filtering at all.
- HNSW parameters (`m`, `ef_construct`) are tunable per collection; Qdrant exposes these via
  the client API without rebuilding the index from scratch.
- Production durability: Qdrant persists to disk with WAL; FAISS is in-memory only and requires
  a custom serialisation strategy for restarts.
- Accepted overhead: an additional Docker service and EC2 network hop (kept <2ms on same VPC).

---

### 2. FlashRank Nano over ColBERT / full cross-encoder transformers

**Context:**
After first-pass hybrid retrieval returns 50 candidates, a reranker improves precision before
the final top-7 enters the generation prompt. ColBERT (multi-vector late interaction) and full
cross-encoder models (e.g., `ms-marco-MiniLM-L-12-v2`) were considered.

**Decision:**
FlashRank Nano (4MB in-process, CPU-only).

**Trade-offs:**
- FlashRank Nano runs entirely in-process with no Torch dependency; cold-start is negligible
  and p95 rerank latency is under 50ms on the 50-candidate window.
- ColBERT requires 1-2 orders of magnitude more storage (per-token late-interaction vectors)
  and introduces a microservice or Torch runtime dependency — unjustifiable for a 100-200 doc
  corpus on a solo build.
- The accepted cost is approximately 2 nDCG points versus a full cross-encoder. Contextual
  prefix generation at ingest time closes most of that gap by enriching chunk representations
  before they reach the reranker.

---

### 3. BM25 + dense hybrid retrieval (RRF k=60) over dense-only

**Context:**
Pure dense retrieval (bi-encoder cosine similarity) works well for paraphrase and semantic
queries but misses exact-match operational jargon that embeddings compress away.

**Decision:**
Qdrant sparse (BM25) + dense vectors merged via Reciprocal Rank Fusion with k=60.

**Trade-offs:**
- Operational documents contain exact identifiers — shift IDs, regulation codes, venue
  department codes, equipment part numbers — that appear verbatim in queries. BM25 scores
  these exact matches with high precision; dense embeddings treat them as out-of-vocabulary
  or hash them into overlapping regions.
- RRF with k=60 is parameter-free in the sense that k is a dampening constant, not a
  learned weight; no training data is needed to tune the merge.
- Accepted cost: two Qdrant searches per query (dense + sparse) instead of one. On a 16GB
  EC2 instance this adds ~5-10ms; well within the p95 SLA.

---

### 4. text_to_sql design — curated views, 3-candidate generation, safety validator

**Context:**
Pointing an LLM at raw database tables on a BIRD-class benchmark yields ~75-82% execution
accuracy; on enterprise schemas (Spider 2.0) this collapses to ~10-21%. The system needs
reliable SQL over operational staffing and incident data.

**Decision:**
SQL tool operates over five pre-modelled curated views only (`v_shift_summary`, `v_incidents`,
`v_compliance_checks`, `v_staff_roster`, `v_protocol_thresholds`). Schema-linker retrieves
relevant view metadata from Qdrant before generation. Three SQL candidates are generated and
the best is selected. A safety validator blocks DML statements and joins deeper than 3 tables.
Self-correction retries on execution error up to 3 times (SQL-of-Thought pattern).

**Trade-offs:**
- Curated views with clean column names and ≤3 join depth push execution accuracy toward
  ~98-100% on covered queries (dbt 2026 benchmark) — a substantial improvement over raw tables.
- The accepted constraint is coverage: queries outside the five views cannot be answered via
  SQL; the agent falls back to vector search and explains the limitation in the response.
- Wrong-but-runnable SQL (executes, returns a plausible but incorrect number) is the primary
  silent failure mode. Mitigations: result-schema validation, immutable audit log of every
  executed query, and faithfulness gating on the synthesised answer.

---

### 5. No LangChain / LlamaIndex — direct SDK only

**Context:**
LangChain and LlamaIndex offer pre-built retrieval chains, agent executors, and LLM
abstraction layers that could accelerate prototyping.

**Decision:**
Direct SDK calls throughout: `qdrant-client`, `ollama` (Python), `anthropic`, `openai`,
`sqlite3` / `aiosqlite`. No orchestration framework.

**Trade-offs:**
- Frameworks introduce hidden retry logic, version-drift between framework and provider SDKs,
  and prompt obfuscation that makes per-component Langfuse instrumentation at the required
  granularity harder to maintain.
- The custom routing logic (simple_doc / simple_sql / complex / unanswerable) and the
  hybrid local/cloud/SQL agent do not map cleanly to any standard chain abstraction; forcing
  them in would require more workarounds than writing direct calls.
- The 150-line file cap (enforced project-wide) stays tractable when each file does one
  thing via direct calls; it becomes unworkable when a framework's abstractions are wired
  in alongside custom observability hooks.
- Accepted cost: more boilerplate per integration point. This is intentional — each
  integration is explicit, testable, and observable.

---

### 6. 3B local model only (llama3.2:3b) for routing and expansion

**Context:**
The dev machine has an RTX 3070 with 8GB VRAM. Llama 3.2 ships 1B/3B (text-only) and
11B/90B (vision). Llama 3.1 8B at aggressive quantisation is an alternative.

**Decision:**
llama3.2:3b handles intent routing and multi-query expansion (3 reformulations per query).
All complex multi-hop synthesis routes to Claude 3.5 Sonnet or GPT-4o via `cloud_llm.py`.

**Trade-offs:**
- The 3B model fits comfortably within 8GB VRAM alongside the FastAPI process; 8B at
  aggressive quantisation adds VRAM pressure with minimal benefit for classification-only tasks.
- Routing and query reformulation do not require long-context reasoning; 3B is sufficient
  for intent classification over a 4-class schema and 3-reformulation expansion.
- All latency-sensitive, quality-sensitive generation (faithfulness-gated answers for
  compliance queries) routes to cloud LLMs with significantly higher capability.
- Accepted cost: cloud LLM invocations on the complex path add API latency and cost per query.
  Per-query cost is tracked in Langfuse and bounded by a hard token budget.

---

### 7. Contextual prefix generation (Claude haiku at ingest)

**Context:**
Raw chunk embeddings from all-MiniLM-L6-v2 lose document context — a chunk extracted from
page 4 of a compliance PDF has no embedding signal that it came from a compliance doc about
floor staffing, dated June 2025. This hurts multi-hop retrieval.

**Decision:**
At ingest time, Claude 3 Haiku generates a one-sentence context string per chunk:
"This chunk is from the [doc_type] dated [date], [dept] dept, discussing [topic summary]."
This prefix is prepended before embedding and stored with the chunk payload.

**Trade-offs:**
- Anthropic benchmarks report a ~67% reduction in top-20-chunk retrieval failure rate
  (5.7% → 1.9%) with contextual embeddings + contextual BM25 — the highest single-change
  ROI for retrieval quality.
- Cost: ~$1/M document tokens with prompt caching. A 200-document corpus averaging
  5k tokens each totals ~1M tokens = approximately $1 for the full initial index.
  This is a one-time ingest cost, not a per-query cost.
- Accepted constraint: contextual prefix generation adds ingest-time latency (~2-5s per
  chunk via API) and requires `ANTHROPIC_API_KEY`. The ingestion pipeline is a no-op
  (returns raw chunk text) when the key is absent.

---

### 8. Semantic cache (GPTCache) disabled by default

**Context:**
GPTCache provides a semantic similarity cache layer that can return a cached answer when
an incoming query is semantically similar to a prior one, bypassing LLM generation.
Published benchmarks show 40-80% LLM cost reduction; ~31% of LLM queries are semantically
similar to prior ones.

**Decision:**
`semantic_cache_url=""` is the off switch (configured in `src/config.py`). The cache is
instantiated only when this setting is non-empty. Production deployments turn it on via
the `SEMANTIC_CACHE_URL` environment variable.

**Trade-offs:**
- During evaluation runs, a warm cache would return stale answers that match prior golden
  questions, making eval metrics artificially high and masking regressions. Disabled by
  default prevents this class of eval contamination.
- Every request — hit or miss — increments `rag_cache_hit_total{result="hit"}` or
  `rag_cache_hit_total{result="miss"}` in Prometheus. The miss path is always fully traced
  in Langfuse regardless of cache state.
- Accepted cost: no latency or cost reduction in development/CI. The benefit is intentionally
  deferred to production where stale answers are managed via TTL and invalidation policies.
