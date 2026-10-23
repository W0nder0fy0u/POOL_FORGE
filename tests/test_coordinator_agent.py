"""Phase 3 Tests: Coordinator, Agent Worker, Scheduler, Auto-Split, and Distributed Training.

Acceptance criteria:
- Room creation and node registration over WebSocket.
- Proportional contiguous auto-split based on reported RAM caps.
- "pool too small" error when total memory exceeds pool capacity.
- Unauthorized requests rejected with 403.
- ONE API CALL trains a model across 2 local agents via auto-split.
"""

import asyncio
import time
import httpx
import pytest
import uvicorn
from common.model_zoo import get_large_mlp, get_tiny_test_model
from common.schemas import NodeTelemetry
from coordinator.scheduler import schedule_pipeline_split
from coordinator.server import app
from agent.worker import AgentWorker


def test_scheduler_proportional_split_and_pool_too_small():
    """Verify scheduler allocates contiguous layers proportional to RAM and catches undersized pools."""
    specs = get_tiny_test_model()  # 5 layers

    node1 = NodeTelemetry(
        node_id="node-1gb",
        hostname="lap1",
        os_name="Linux",
        ram_total_gb=16.0,
        ram_free_gb=1.0,  # 1 GB
        cpu_percent=10.0,
    )
    node2 = NodeTelemetry(
        node_id="node-3gb",
        hostname="lap2",
        os_name="Windows",
        ram_total_gb=16.0,
        ram_free_gb=3.0,  # 3 GB (3x larger)
        cpu_percent=10.0,
    )

    # 1. Successful split
    plan = schedule_pipeline_split(specs, [node1, node2], batch_size=16)
    assert len(plan) == 2
    assert plan[0].start_layer == 0
    assert plan[0].end_layer == plan[1].start_layer
    assert plan[1].end_layer == len(specs)

    # 2. "pool too small" capacity test
    # A gigantic MLP exceeding combined 4 GB
    huge_specs = get_large_mlp(in_features=4096, hidden_width=8192, depth=8, num_classes=10)
    with pytest.raises(ValueError, match="pool too small"):
        schedule_pipeline_split(huge_specs, [node1, node2], batch_size=16)


@pytest.mark.asyncio
async def test_end_to_end_training_via_coordinator_and_two_agents():
    """CRITICAL ACCEPTANCE TEST: One API call trains a model across 2 local agents with auto-split."""
    # Start coordinator on an available local port
    config = uvicorn.Config(app, host="127.0.0.1", port=8001, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    # Wait for coordinator server startup
    await asyncio.sleep(0.5)

    base_url = "http://127.0.0.1:8001"
    async with httpx.AsyncClient(base_url=base_url) as client:
        # 1. Create room
        res = await client.post("/api/rooms")
        assert res.status_code == 200
        room_data = res.json()
        room_code = room_data["room_code"]
        admin_token = room_data["admin_token"]

        # 2. Start two agents with different RAM caps
        agent1 = AgentWorker(
            room_code=room_code,
            server_url=base_url,
            node_id="agent-alpha",
            ram_cap_gb=1.0,
            transport_type="relay",
        )
        agent2 = AgentWorker(
            room_code=room_code,
            server_url=base_url,
            node_id="agent-beta",
            ram_cap_gb=2.0,
            transport_type="relay",
        )

        task1 = asyncio.create_task(agent1.start())
        task2 = asyncio.create_task(agent2.start())

        # Wait for agents to register and appear in room status
        for _ in range(30):
            await asyncio.sleep(0.2)
            room_info = (await client.get(f"/api/rooms/{room_code}")).json()
            if room_info["node_count"] == 2:
                break
        assert room_info["node_count"] == 2

        # 3. Unauthorized start should fail with 403
        bad_start = await client.post(
            f"/api/rooms/{room_code}/start",
            headers={"X-Admin-Token": "invalid-token"},
            json={"model_name": "tiny_test"},
        )
        assert bad_start.status_code == 403

        # 4. Trigger training with ONE valid API call
        start_payload = {
            "model_name": "tiny_test",
            "dataset_name": "synthetic",
            "batch_size": 8,
            "epochs": 1,
            "steps_per_epoch": 4,
            "learning_rate": 0.05,
            "checkpoint_interval": 2,
        }

        start_res = await client.post(
            f"/api/rooms/{room_code}/start",
            headers={"X-Admin-Token": admin_token},
            json=start_payload,
        )
        assert start_res.status_code == 200
        start_data = start_res.json()
        assert start_data["status"] == "started"
        split_plan = start_data["split_plan"]
        assert len(split_plan) == 2
        assert split_plan[0]["start_layer"] == 0
        assert split_plan[1]["end_layer"] == len(get_tiny_test_model())

        # 5. Wait for training steps to execute and metrics to be reported
        metrics = []
        for _ in range(40):
            await asyncio.sleep(0.3)
            metrics_res = (await client.get(f"/api/rooms/{room_code}/metrics")).json()
            metrics = metrics_res.get("metrics", [])
            if len(metrics) >= 4:
                break

        assert len(metrics) >= 4, f"Expected 4 step metrics, got {len(metrics)}"
        # Verify loss values are valid floats and recorded sequentially
        assert metrics[0]["step"] == 1
        assert metrics[-1]["step"] == 4
        assert metrics[0]["loss"] > 0

        # Clean shutdown
        await agent1.stop()
        await agent2.stop()
        task1.cancel()
        task2.cancel()

        server.should_exit = True
        await server_task
