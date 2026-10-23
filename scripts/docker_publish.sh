#!/usr/bin/env bash
# Build the coordinator image and push it to Docker Hub, ready for Render
# ("New -> Web Service -> Existing Image").
#
#   bash scripts/docker_publish.sh <dockerhub-username> [tag]
#
# Example:  bash scripts/docker_publish.sh subh123        -> docker.io/subh123/poolforge:latest
# Needs a free Docker Hub account; you are asked to log in the first time.
# Works with docker or podman (whichever is installed).
set -euo pipefail
cd "$(dirname "$0")/.."

USER_NAME="${1:-}"
TAG="${2:-latest}"
if [[ -z "$USER_NAME" ]]; then
  echo "usage: bash scripts/docker_publish.sh <dockerhub-username> [tag]"
  exit 1
fi

if command -v docker >/dev/null 2>&1; then ENGINE=docker
elif command -v podman >/dev/null 2>&1; then ENGINE=podman
else echo "Install Docker (or podman) first."; exit 1; fi

IMAGE="docker.io/${USER_NAME}/poolforge:${TAG}"
echo "== Building $IMAGE with $ENGINE (linux/amd64, what Render runs)"
# --format docker keeps the HEALTHCHECK when building with podman.
EXTRA=""
[[ "$ENGINE" == "podman" ]] && EXTRA="--format docker"
$ENGINE build $EXTRA --platform linux/amd64 -f deploy/Dockerfile -t "$IMAGE" .

echo "== Quick self-test: start the image and call /health"
CID=$($ENGINE run -d -p 127.0.0.1:18000:8000 "$IMAGE")
trap '$ENGINE rm -f "$CID" >/dev/null 2>&1 || true' EXIT
for _ in $(seq 1 30); do
  if curl -sf http://127.0.0.1:18000/health >/dev/null; then echo "   health OK"; break; fi
  sleep 1
done
curl -sf http://127.0.0.1:18000/health >/dev/null || { echo "Image did not start; logs:"; $ENGINE logs "$CID"; exit 1; }

echo "== Pushing (log in to Docker Hub if asked)"
$ENGINE login docker.io
$ENGINE push "$IMAGE"

echo
echo "Done. On Render: New -> Web Service -> Existing Image -> $IMAGE"
echo "Health check path: /health   (Render sets PORT itself; the app reads it)"
