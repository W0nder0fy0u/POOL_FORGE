# PoolForge Coordinator Dockerfile
#
# The coordinator is a lightweight FastAPI process — it does NOT run model training.
# All heavy ML work (PyTorch, tensor ops) is handled by the agent processes that run
# on participant laptops.  This image therefore skips torch/numpy entirely, keeping
# the image small (~200 MB vs ~4 GB with torch).
#
# Build:
#   docker build -f deploy/Dockerfile -t poolforge-coordinator .
# Run:
#   docker run -p 8000:8000 -v poolforge_data:/app/data poolforge-coordinator

# ---------------------------------------------------------------------------
# Stage 1: Dependency layer (separate so pip cache survives code changes)
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS deps

WORKDIR /app

# Only install coordinator-specific dependencies (no torch/heavy ML)
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    websockets \
    pydantic \
    aiofiles \
    psutil \
    cryptography

# ---------------------------------------------------------------------------
# Stage 2: Runtime image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from deps stage
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy only the coordinator and shared code — do NOT copy agent/ or .venv/
COPY coordinator/ coordinator/
COPY common/ common/
COPY dashboard/dist/ dashboard/dist/

# Persistent storage directory for SQLite database
RUN mkdir -p /app/data

# ── Environment ──────────────────────────────────────────────────────────────
# DATABASE_URL can be overridden at runtime (e.g. for a cloud Postgres future)
ENV DATABASE_URL=sqlite:///./data/poolforge.db
ENV HOST=0.0.0.0
ENV PORT=8000

# Expose coordinator WebSocket + REST port
EXPOSE 8000

# ── Healthcheck ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["uvicorn", "coordinator.server:app", "--host", "0.0.0.0", "--port", "8000"]
