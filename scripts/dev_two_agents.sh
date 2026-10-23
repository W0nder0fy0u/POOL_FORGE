#!/usr/bin/env bash
# Run a whole pool on ONE machine: coordinator + 2 agents with different RAM caps.
#
#   bash scripts/dev_two_agents.sh            # just start everything
#   bash scripts/dev_two_agents.sh --train    # ...and start a training job via one API call
#
# Agent A pretends to have 1 GB free RAM, agent B 2 GB (--ram-cap-gb), so the
# scheduler gives B about twice as many layers. Press Ctrl+C to stop everything.
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8000}"
URL="http://127.0.0.1:${PORT}"
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"
mkdir -p run

pids=()
cleanup() { echo; echo "Stopping..."; kill "${pids[@]}" 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "== Starting coordinator on $URL (log: run/coordinator.log)"
POOLFORGE_PORT="$PORT" "$PY" -m coordinator.server > run/coordinator.log 2>&1 &
pids+=($!)
for _ in $(seq 1 60); do curl -sf "$URL/health" >/dev/null && break; sleep 0.5; done
curl -sf "$URL/health" >/dev/null || { echo "Coordinator did not start; see run/coordinator.log"; exit 1; }

ROOM_JSON=$(curl -sf -X POST "$URL/api/rooms")
CODE=$("$PY" -c 'import json,sys; print(json.load(sys.stdin)["room_code"])' <<< "$ROOM_JSON")
TOKEN=$("$PY" -c 'import json,sys; print(json.load(sys.stdin)["admin_token"])' <<< "$ROOM_JSON")

echo "== Starting agent laptop-a (1 GB cap) and laptop-b (2 GB cap) (logs: run/agent-*.log)"
"$PY" -m agent.cli join "$CODE" --server "$URL" --node-id laptop-a --ram-cap-gb 1 --workdir run/a > run/agent-a.log 2>&1 &
pids+=($!)
"$PY" -m agent.cli join "$CODE" --server "$URL" --node-id laptop-b --ram-cap-gb 2 --workdir run/b > run/agent-b.log 2>&1 &
pids+=($!)
sleep 4

echo
echo "  Room code   : $CODE"
echo "  Admin token : $TOKEN"
echo "  Dashboard   : $URL/room/$CODE#admin=$TOKEN"
echo "  Join link   : $URL/join/$CODE"
echo

if [[ "${1:-}" == "--train" ]]; then
  echo "== Starting training with ONE API call (mnist_cnn, auto split)"
  curl -sf -X POST "$URL/api/rooms/$CODE/jobs" -H "X-Admin-Token: $TOKEN" -H "Content-Type: application/json" \
    -d '{"model":"mnist_cnn","batch_size":64,"epochs":1,"steps_per_epoch":100}' | "$PY" -m json.tool
fi

echo "Open the dashboard link above. Press Ctrl+C to stop."
wait
