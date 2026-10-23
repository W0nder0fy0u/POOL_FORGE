"""Phase 5 Tests: Robustness, Rate Limiting, Pause/Resume, Node-Drop Handling, and Checkpoints.

Acceptance criteria:
- Rate limiting rejects excessive room/join attempts with 429.
- Pause/resume endpoints pause and continue distributed training loop.
- Node drop halts training gracefully and resumes when the node reconnects.
- Automatic stage checkpoint generation.
"""

import asyncio
from pathlib import Path
import httpx
import pytest
import uvicorn
from coordinator.server import app, check_rate_limit
from agent.worker import AgentWorker


def test_rate_limiter_logic():
    """Verify rate limiter throws 429 when threshold exceeded."""
    ip = "192.168.1.100"
    # Allow 5 requests
    for _ in range(5):
        check_rate_limit(ip, max_requests=5, window_seconds=10.0)

    # 6th request should fail
    with pytest.raises(Exception) as exc_info:
        check_rate_limit(ip, max_requests=5, window_seconds=10.0)
    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_pause_resume_training_api():
    """Verify Pause and Resume control signals during distributed training."""
    config = uvicorn.Config(app, host="127.0.0.1", port=8004, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    await asyncio.sleep(0.4)
    base_url = "http://127.0.0.1:8004"

    async with httpx.AsyncClient(base_url=base_url) as client:
        # Create room
        res = await client.post("/api/rooms")
        room_data = res.json()
        room_code = room_data["room_code"]
        admin_token = room_data["admin_token"]

        # Connect 2 agents
        a1 = AgentWorker(room_code=room_code, server_url=base_url, node_id="rob-a1", ram_cap_gb=1.0)
        a2 = AgentWorker(room_code=room_code, server_url=base_url, node_id="rob-a2", ram_cap_gb=1.0)
        t1 = asyncio.create_task(a1.start())
        t2 = asyncio.create_task(a2.start())

        # Wait for registration
        for _ in range(25):
            await asyncio.sleep(0.2)
            if (await client.get(f"/api/rooms/{room_code}")).json()["node_count"] == 2:
                break

        # Start 6-step training
        start_res = await client.post(
            f"/api/rooms/{room_code}/start",
            headers={"X-Admin-Token": admin_token},
            json={
                "model_name": "tiny_test",
                "batch_size": 8,
                "epochs": 1,
                "steps_per_epoch": 6,
                "checkpoint_interval": 2,
            },
        )
        assert start_res.status_code == 200

        # Wait for step 1
        await asyncio.sleep(0.8)

        # PAUSE training
        pause_res = await client.post(
            f"/api/rooms/{room_code}/pause",
            headers={"X-Admin-Token": admin_token},
        )
        assert pause_res.status_code == 200
        assert pause_res.json()["status"] == "paused"
        assert a1.status == "paused" or not a1._pause_event.is_set()

        # RESUME training
        resume_res = await client.post(
            f"/api/rooms/{room_code}/resume",
            headers={"X-Admin-Token": admin_token},
        )
        assert resume_res.status_code == 200
        assert resume_res.json()["status"] == "resumed"

        # Wait for all steps to finish
        for _ in range(30):
            await asyncio.sleep(0.3)
            metrics_res = (await client.get(f"/api/rooms/{room_code}/metrics")).json()
            if len(metrics_res.get("metrics", [])) >= 6:
                break

        metrics = (await client.get(f"/api/rooms/{room_code}/metrics")).json().get("metrics", [])
        assert len(metrics) >= 6

        # Checkpoints should exist in checkpoints directory
        ckpts = list(Path("./checkpoints").glob("*.pt"))
        assert len(ckpts) > 0, "Expected checkpoint files to be created"

        # Clean shutdown
        await a1.stop()
        await a2.stop()
        t1.cancel()
        t2.cancel()
        server.should_exit = True
        await server_task
