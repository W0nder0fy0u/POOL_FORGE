"""Phase 4 Tests: Dashboard static serving, SPA routing, and live WebSocket data streaming.

Acceptance criteria:
- Coordinator serves built dashboard HTML at / and /join/:code.
- Dashboard WebSocket connects to /ws/dashboard/:code and receives state snapshot.
- Live telemetry and step metrics broadcast to dashboard WebSocket.
"""

import asyncio
import json
import httpx
import pytest
import uvicorn
import websockets
from coordinator.server import app, broadcast_to_dashboard


@pytest.mark.asyncio
async def test_dashboard_static_and_spa_routing():
    """Verify coordinator serves dashboard HTML for root and client routes."""
    config = uvicorn.Config(app, host="127.0.0.1", port=8002, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    await asyncio.sleep(0.4)

    try:
        async with httpx.AsyncClient(base_url="http://127.0.0.1:8002") as client:
            # 1. Root route
            res_root = await client.get("/")
            assert res_root.status_code == 200
            assert "PoolForge" in res_root.text
            assert '<div id="root">' in res_root.text

            # 2. SPA client route (/join/ABC123)
            res_join = await client.get("/join/ABC123")
            assert res_join.status_code == 200
            assert '<div id="root">' in res_join.text
    finally:
        server.should_exit = True
        await server_task


@pytest.mark.asyncio
async def test_dashboard_websocket_streaming():
    """Verify live state snapshot and event broadcasting over /ws/dashboard/:code."""
    config = uvicorn.Config(app, host="127.0.0.1", port=8003, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    await asyncio.sleep(0.4)

    room_code = "DASH99"
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8003") as client:
        # Create room in DB
        res = await client.post("/api/rooms")
        room_code = res.json()["room_code"]

    ws_url = f"ws://127.0.0.1:8003/ws/dashboard/{room_code}"

    try:
        async with websockets.connect(ws_url) as ws:
            # 1. Should receive initial snapshot
            raw_snapshot = await ws.recv()
            snapshot = json.loads(raw_snapshot)
            assert snapshot["type"] == "snapshot"
            assert "nodes" in snapshot

            # 2. Test server broadcast to dashboard
            test_event = {"type": "step_metric", "payload": {"step": 1, "loss": 0.42}}
            await broadcast_to_dashboard(room_code, test_event)

            received_event = await ws.recv()
            event_data = json.loads(received_event)
            assert event_data["type"] == "step_metric"
            assert event_data["payload"]["loss"] == 0.42
    finally:
        server.should_exit = True
        await server_task
