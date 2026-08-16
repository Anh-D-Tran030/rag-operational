# CPU-only image. Every model call that used to need a local GPU now goes out over
# HTTP to OpenRouter (LOCAL_PROVIDER=openrouter), so this runs on a small shared
# instance. The only models in the container are the embedder and the reranker,
# both CPU-sized and baked in below.
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build into a venv rather than --prefix. With --prefix, the torch install below
# lands in the builder's own site-packages, so the later project install sees torch
# as already satisfied and leaves it (and its dependencies) out of the copied tree.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch, from PyTorch's CPU index. The default PyPI wheel drags in ~2GB of
# CUDA libraries that can never be used here; installing it first stops
# sentence-transformers pulling the GPU build in as a transitive dependency.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .


FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/models/hf \
    # Not /tmp: FlashRank's default cache dir is swept by container restarts and
    # tmpfiles jobs, and the model vanishing shows up as a mid-request crash.
    FLASHRANK_CACHE_DIR=/opt/models/flashrank

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

# Bake both local models into the image. Downloading them on first request is what
# made a cold query take ~80s; a container that reaches for the network mid-request
# is also a container that breaks when the network says no.
# This stays above the source COPY so editing code does not re-download ~100MB.
RUN mkdir -p "$FLASHRANK_CACHE_DIR" \
    && python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2')" \
    && python -c "\
import os; from flashrank import Ranker; \
Ranker(model_name='ms-marco-MiniLM-L-12-v2', cache_dir=os.environ['FLASHRANK_CACHE_DIR'])"

# Presidio's default NLP engine wants spaCy's en_core_web_lg and, when it is
# absent, pip-installs it in the middle of the first request — which a non-root
# container cannot do, so PII redaction failed with a 500 instead of degrading.
RUN python -m spacy download en_core_web_lg

# Only after the models are cached — set earlier, this would block the downloads above.
ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

COPY src ./src
COPY tests/fixtures ./tests/fixtures
COPY scripts ./scripts
COPY eval_data ./eval_data

# Seed the read-only operational database at build time so the runtime filesystem
# can stay immutable and no volume is required.
RUN python -c "\
import asyncio; \
from src.ingestion.structured_loader import seed_database; \
asyncio.run(seed_database('sqlite+aiosqlite:///./operational.db', 'tests/fixtures'))"

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app /opt/models
USER appuser

EXPOSE 8000

# The platform probes this; it is also what a readiness gate should block on.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "\
import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
