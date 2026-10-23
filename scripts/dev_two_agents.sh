#!/usr/bin/env bash
# PoolForge Local Multi-Agent Simulation
# Launches 1 Coordinator + 2 Agents with different RAM caps on a single machine

set -e

PORT=8000
SERVER_URL="http://127.0.0.1:${PORT}"
VENV_PY="./.venv/bin/python"

if [ ! -f "$VENV_PY" ]; then
    VENV_PY="python3"
fi

echo "=========================================================="
echo " Starting PoolForge Coordinator on ${SERVER_URL}..."
echo "=========================================================="

# Start coordinator in background
$VENV_PY -m coordinator.server &
COORD_PID=$!

# Trap cleanup to terminate child processes on exit
cleanup() {
    echo ""
    echo "Terminating Coordinator and Agents..."
    kill $COORD_PID 2>/dev/null || true
    kill $AGENT1_PID 2>/dev/null || true
    kill $AGENT2_PID 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Wait for coordinator to start listening
echo "Waiting for Coordinator..."
for i in {1..20}; do
    if curl -s "${SERVER_URL}/docs" > /dev/null 2>&1; then
        echo "Coordinator is ready!"
        break
    fi
    sleep 0.5
done

# Create a test room
ROOM_RESP=$(curl -s -X POST "${SERVER_URL}/api/rooms")
ROOM_CODE=$(echo $ROOM_RESP | $VENV_PY -c "import sys, json; print(json.load(sys.stdin)['room_code'])")
ADMIN_TOKEN=$(echo $ROOM_RESP | $VENV_PY -c "import sys, json; print(json.load(sys.stdin)['admin_token'])")

echo "Created Pool Room: ${ROOM_CODE}"
echo "Admin Token:      ${ADMIN_TOKEN}"
echo "Join URL:         ${SERVER_URL}/join/${ROOM_CODE}"
echo ""

# Launch Agent 1 (Capped at 1.0 GB RAM)
echo "Starting Agent 1 (Capped at 1.0 GB RAM)..."
$VENV_PY agent/cli.py join "$ROOM_CODE" --server "$SERVER_URL" --node-id "laptop-alpha" --ram-cap-gb 1.0 &
AGENT1_PID=$!

# Launch Agent 2 (Capped at 2.0 GB RAM)
echo "Starting Agent 2 (Capped at 2.0 GB RAM)..."
$VENV_PY agent/cli.py join "$ROOM_CODE" --server "$SERVER_URL" --node-id "laptop-beta" --ram-cap-gb 2.0 &
AGENT2_PID=$!

sleep 2
echo ""
echo "Both agents joined! Inspect room state:"
curl -s "${SERVER_URL}/api/rooms/${ROOM_CODE}" | $VENV_PY -m json.tool

echo ""
echo "To trigger distributed training, execute:"
echo "curl -X POST \"${SERVER_URL}/api/rooms/${ROOM_CODE}/start\" -H \"X-Admin-Token: ${ADMIN_TOKEN}\" -H \"Content-Type: application/json\" -d '{\"model_name\":\"tiny_test\",\"epochs\":1,\"steps_per_epoch\":5}'"
echo ""
echo "Press Ctrl+C to stop all processes."

wait $COORD_PID
