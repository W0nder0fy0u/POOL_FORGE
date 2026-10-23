# PoolForge coordinator image (REST + WebSockets + relay + dashboard).
#
# The coordinator never runs PyTorch (agents do the training), so the image
# installs only FastAPI/uvicorn and stays small (~150 MB instead of ~2 GB).
#
# Build and run (from the repo root) - ONE command each:
#   docker build -t poolforge-coordinator .
#   docker run -p 8000:8000 -v poolforge-data:/app/data poolforge-coordinator
# Then open http://localhost:8000

# ---- stage 1: build the React dashboard ----------------------------------
FROM node:22-alpine AS dashboard
WORKDIR /build
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY dashboard/ ./
RUN npm run build

# ---- stage 2: Python runtime ----------------------------------------------
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    POOLFORGE_DB_PATH=/app/data/poolforge.db POOLFORGE_HOST=0.0.0.0 PORT=8000
WORKDIR /app

COPY requirements-coordinator.txt .
RUN pip install --no-cache-dir -r requirements-coordinator.txt

# Only what the coordinator needs (no agent/, no torch).
COPY common/ common/
COPY coordinator/ coordinator/
COPY --from=dashboard /build/dist dashboard/dist

# Run as an unprivileged user; /app/data holds the SQLite database (mount a volume there).
RUN useradd --create-home --uid 1000 poolforge && mkdir -p /app/data && chown -R poolforge /app/data
USER poolforge
VOLUME ["/app/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health')"

CMD ["python", "-m", "coordinator.server"]
