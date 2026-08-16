# Incident Log

Failures encountered during development, how they were found, how they were fixed,
and what each one changed about the design. Listed in rough chronological order.

---

## INC-001 — FlashRank model vanishes on container restart, causes mid-request 500

**Discovered:** M3 integration testing. Container health check passed but the first
rerank call after a restart raised `FileNotFoundError`.

**Root cause:** FlashRank's default model cache directory is `/tmp`. Linux container
runtimes and CI runners sweep `/tmp` at restart (via `tmpfiles.d` or equivalent).
The model was downloaded on first request, disappeared on the next container restart,
and the next rerank call crashed mid-request before the error was caught.

**Fix:** Set `FLASHRANK_CACHE_DIR=/opt/models/flashrank` (a persistent path) in the
Dockerfile runtime stage and bake the model into the image during build with an
explicit `Ranker()` call. The `_get_ranker()` function now reads `FLASHRANK_CACHE_DIR`
rather than defaulting to `/tmp`. The model is now present before the first request.

**Design change:** The model download moved from first-request to build time. Cold-start
latency on the first query dropped from ~80s (network fetch) to negligible.

**Files:** `Dockerfile` (runtime `ENV FLASHRANK_CACHE_DIR`), `src/retrieval/reranker.py`
(`_CACHE_DIR` and `_get_ranker()`).

---

## INC-002 — Spacy NLP engine pip-installs at runtime, crashes non-root container

**Discovered:** M6 PII guardrail integration. First query after container start returned
HTTP 500 instead of a redacted answer. Log showed `pip install en_core_web_lg` followed
by a permission error.

**Root cause:** Presidio's default NLP engine (`SpacyNlpEngine`) checks for the spacy
model at runtime and, if absent, tries to `pip install` it inline. A non-root container
(user `appuser`, UID 10001) cannot write to the system Python path, so the install fails
and the exception propagated all the way to the API response.

**Fix:** Added `RUN python -m spacy download en_core_web_lg` to the Dockerfile runtime
stage, before the `USER appuser` directive. The model is now present at startup and
Presidio's runtime check is a no-op.

**Design change:** Any model or asset that the runtime needs must be baked in during
build. A container that reaches for the network or a package manager mid-request is
also a container that breaks when the network says no.

**Files:** `Dockerfile` (runtime stage, `spacy download` step).

---

## INC-003 — Venue names redacted as PERSON entities, answers become useless

**Discovered:** M6 PII guardrail manual testing. The answer "Fairfield RSL had 6
security staff on Friday" was being returned as "<REDACTED_PERSON> had 6 security
staff on Friday".

**Root cause:** Presidio's default recogniser set includes every entity type it knows.
The spacy NER model mislabels "Fairfield RSL" (a proper noun with a person-like
structure: `Firstname Lastname`) as a `PERSON` entity. With the default entity list,
every answer that named the venue was redacted.

**Fix:** Two changes to `src/guardrails/pii_redactor.py`:

1. Restricted the entity set to genuinely personal data via `PII_ENTITIES` env var,
   defaulting to `PERSON,EMAIL_ADDRESS,PHONE_NUMBER,CREDIT_CARD,IBAN_CODE,AU_TFN,
   AU_MEDICARE`. Location names, suburb names, and shift dates are no longer in scope.

2. Added a `PII_ALLOW_LIST` env var (default: `Fairfield RSL,Fairfield_RSL,Fairfield,RSL`)
   passed to `analyzer.analyze(allow_list=...)`. Presidio skips tokens that appear
   verbatim in the allow list before scoring them.

**Design change:** PII redaction scope is now configurable per-deployment. The allow
list handles domain nouns that the NER model consistently mislabels.

**Files:** `src/guardrails/pii_redactor.py` (`_ENTITIES`, `_ALLOW_LIST`).

---

## INC-004 — Router misclassifies licence/capacity questions as text_to_sql

**Discovered:** M6 router accuracy measurement. Running 23 golden queries through
`classify_intent` showed 2 misroutes: "What is the maximum patron capacity?" and
"What trading hours does the licence permit?" were both classified as `text_to_sql`
instead of `vector_search`. SQL execution found no rows and the answer degraded to
"insufficient context".

**Root cause:** The router was trained on the intuition that numbers imply a database.
Patron capacity and trading hours are numbers that appear in the licence document, not
in any database table. With only 2 few-shot examples in the system prompt, the model
had no signal to distinguish "a number written in a document" from "a count stored in a
database".

**Fix:** Added four new few-shot examples to `_SYSTEM` in `src/generation/router.py`
explicitly covering licence terms, capacities, and stated thresholds:

```
Query: What is the maximum number of patrons the venue is licensed for?
JSON: {"tool": "vector_search", "complexity": "simple"}

Query: What trading hours does the licence permit on a Friday?
JSON: {"tool": "vector_search", "complexity": "simple"}
```

Also added an instruction line: "A number in the question does not make it a database
question. Licence terms, capacities and stated thresholds are written in documents; the
database holds logged events and rostered counts."

**Result:** Misroute rate 2/23 → 0/23 on the golden set after the fix.

**Model note:** After fixing the prompt, measured accuracy on the full golden set:
- `mistralai/mistral-nemo`: 21/23
- `qwen/qwen3-30b-a3b-instruct-2507`: 23/23

Router model updated to `qwen3-30b` in `src/config.py`.

**Files:** `src/generation/router.py` (`_SYSTEM`), `src/config.py` (`router_model`).

---

## INC-005 — Faithfulness gate over-rejects correct answers (gpt-oss-120b), under-rejects hallucinations (mistral-nemo)

**Discovered:** M6 faithfulness gate calibration. After wiring up `check_faithfulness`,
ran a grounded/ungrounded probe: 5 answers with clear supporting context and 5 answers
that fabricated facts not in the retrieved chunks.

**Root cause (over-rejection):** `openai/gpt-oss-120b` as the gate model rejected 4 of
5 grounded answers (false-positive rate 80%). A gate that rejects correct answers
degrades the user experience worse than no gate — the system returns the canned
"cannot be verified" message instead of a correct, supported answer.

**Root cause (under-rejection):** `mistralai/mistral-nemo` failed to catch 4 of 5
hallucinated answers (false-negative rate 80%). This is the opposite failure mode —
hallucinated answers go through unchecked.

**Probe results:**

| Model | Grounded pass (should be 5/5) | Ungrounded block (should be 5/5) |
|:------|:------------------------------|:---------------------------------|
| `openai/gpt-oss-120b` | 1/5 | 5/5 |
| `mistralai/mistral-nemo` | 5/5 | 1/5 |
| `qwen/qwen3-30b-a3b-instruct-2507` | 5/5 | 5/5 |

**Fix:** Gate model updated to `qwen/qwen3-30b-a3b-instruct-2507` in `src/config.py`.

**Design change:** Model selection for the faithfulness gate is measurement-driven, not
price-driven. The gate is the last line of defence against hallucinations; a gate that
fires on correct answers is worse than no gate.

**Files:** `src/config.py` (`gate_model`), `src/generation/faithfulness_gate.py`.

---

## INC-006 — Docker build pulls 2GB CUDA torch despite CPU-only target

**Discovered:** M6 Docker build. `docker build` on a CI runner without a GPU
downloaded a CUDA-linked PyTorch wheel (~2.3GB) pulled in as a transitive dependency
of `sentence-transformers`.

**Root cause:** The default PyPI `torch` wheel includes CUDA libraries. When
`sentence-transformers` is listed in `pyproject.toml` without a pinned torch,
pip resolves the latest CUDA wheel. The container has no GPU and can never use these
libraries, but they inflate the image by ~2GB and slow CI by 3-4 minutes.

**Fix:** In the Dockerfile builder stage, install CPU-only torch first from PyTorch's
CPU index before installing the project:

```dockerfile
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
```

Because `torch` is already present when `pip install .` runs, pip does not reinstall
it or upgrade to a CUDA variant.

**Design change:** CPU-only torch is an explicit first step in the builder stage.
Comment in the Dockerfile documents why — without it, `sentence-transformers` silently
pulls the GPU build as a transitive dependency.

**Files:** `Dockerfile` (builder stage, torch install step).
