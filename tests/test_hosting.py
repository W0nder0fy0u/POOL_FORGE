"""Phase 6 Tests: Distributed Edge Application Hosting.

Acceptance criteria:
- App catalog enforces allow-list security (no arbitrary commands).
- AppRunner launches sample apps (static site, microservice).
- Graceful fallback when cloudflared or docker is missing.
- Coordinator auto-picks node with the MOST free RAM.
- Stop and list endpoints function correctly.
"""

import asyncio
import httpx
import pytest
import uvicorn
from common.apps import get_allow_listed_app, ALLOW_LISTED_APPS
from agent.hosting import AppRunner
from agent.worker import AgentWorker
from coordinator.server import app


def test_allow_list_security():
    """Verify security constraint: arbitrary network commands are strictly blocked."""
    # Pre-approved apps should resolve
    static_cfg = get_allow_listed_app("static_site")
    assert static_cfg["id"] == "static_site"

    # Arbitrary / malicious names must be rejected
    with pytest.raises(PermissionError, match="Unauthorized app"):
        get_allow_listed_app("malicious_curl; rm -rf /")


@pytest.mark.asyncio
async def test_app_runner_lifecycle():
    """Verify AppRunner starts sample static site, serves HTTP traffic, and stops cleanly."""
    runner = AppRunner(node_id="test-host-node")
    assert not runner.list_apps()

    # Launch static site
    res = await runner.start_app("static_site")
    assert res["status"] == "running"
    port = res["port"]
    url = res["url"]
    assert port > 0
    assert "http" in url

    # Query the running local app
    async with httpx.AsyncClient() as client:
        http_res = await client.get(f"http://127.0.0.1:{port}", timeout=3.0)
        assert http_res.status_code == 200
        assert "PoolForge" in http_res.text

    # Stop app
    stopped = await runner.stop_app("static_site")
    assert stopped is True
    assert len(runner.list_apps()) == 0


@pytest.mark.asyncio
async def test_coordinator_deploy_api_picks_highest_ram_node():
    """Verify coordinator picks node with most free RAM and coordinates deployment."""
    config = uvicorn.Config(app, host="127.0.0.1", port=8005, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    await asyncio.sleep(0.4)
    base_url = "http://127.0.0.1:8005"

    async with httpx.AsyncClient(base_url=base_url) as client:
        # Create room
        room_data = (await client.post("/api/rooms")).json()
        room_code = room_data["room_code"]
        admin_token = room_data["admin_token"]

        # Connect Node A (1 GB RAM cap) and Node B (4 GB RAM cap)
        a1 = AgentWorker(room_code=room_code, server_url=base_url, node_id="low-ram-node", ram_cap_gb=1.0)
        a2 = AgentWorker(room_code=room_code, server_url=base_url, node_id="high-ram-node", ram_cap_gb=4.0)
        t1 = asyncio.create_task(a1.start())
        t2 = asyncio.create_task(a2.start())

        # Wait for both nodes to join
        for _ in range(25):
            await asyncio.sleep(0.2)
            if (await client.get(f"/api/rooms/{room_code}")).json()["node_count"] == 2:
                break

        # Trigger deployment without specifying node_id
        deploy_res = await client.post(
            f"/api/rooms/{room_code}/deploy",
            headers={"X-Admin-Token": admin_token},
            json={"app_id": "flask_api"},
        )
        assert deploy_res.status_code == 200
        deploy_data = deploy_res.json()
        # MUST automatically pick node with most free RAM
        assert deploy_data["node_id"] == "high-ram-node"
        assert deploy_data["selection_reason"] == "most_free_ram"

        # Wait for agent to start app and report URL
        await asyncio.sleep(0.8)

        apps_list = (await client.get(f"/api/rooms/{room_code}/apps")).json().get("apps", [])
        assert len(apps_list) > 0
        app_entry = apps_list[0]
        assert app_entry["node_id"] == "high-ram-node"
        assert app_entry["status"] == "running"

        # Test the deployed microservice API
        service_port = app_entry["port"]
        service_res = await client.get(f"http://127.0.0.1:{service_port}")
        assert service_res.status_code == 200
        service_json = service_res.json()
        assert service_json["service"] == "poolforge-edge-microservice"
        assert service_json["status"] == "healthy"

        # Stop deployed app
        stop_res = await client.post(
            f"/api/rooms/{room_code}/apps/flask_api/stop",
            headers={"X-Admin-Token": admin_token},
        )
        assert stop_res.status_code == 200

        # Clean shutdown
        await a1.app_runner.stop_app("flask_api")
        await a2.app_runner.stop_app("flask_api")
        await a1.stop()
        await a2.stop()
        t1.cancel()
        t2.cancel()
        server.should_exit = True
        await server_task
