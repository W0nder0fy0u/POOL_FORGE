"""Phase 4: the coordinator serves the built dashboard and streams live data to it."""

import asyncio
import json

import pytest
import websockets

from coordinator.server import DIST_DIR


@pytest.mark.skipif(not (DIST_DIR / "index.html").exists(), reason="dashboard not built")
async def test_spa_routes_serve_index(coordinator):
    for path in ("/", "/join/ABC123", "/room/ABC123/training"):
        r = await coordinator.http.get(path)
        assert r.status_code == 200 and '<div id="root">' in r.text
    js = [p.name for p in (DIST_DIR / "assets").glob("*.js")][0]
    r = await coordinator.http.get(f"/assets/{js}")
    assert r.status_code == 200 and "javascript" in r.headers["content-type"]


async def test_no_path_traversal(coordinator):
    for path in ("/../coordinator/db.py", "/%2e%2e/%2e%2e/etc/passwd", "/assets/../../pyproject.toml"):
        r = await coordinator.http.get(path)
        assert "CREATE TABLE" not in r.text and "root:" not in r.text and "[project]" not in r.text


async def test_unknown_api_is_404_json(coordinator):
    r = await coordinator.http.get("/api/nope")
    assert r.status_code == 404


async def test_dashboard_websocket_streams_live_training(coordinator, run_agent):
    room = await coordinator.create_room()
    code, h = room["room_code"], {"X-Admin-Token": room["admin_token"]}
    ws_url = coordinator.url.replace("http", "ws") + f"/ws/dashboard/{code}"
    async with websockets.connect(ws_url) as ws:
        snap = json.loads(await ws.recv())
        assert snap["type"] == "snapshot" and snap["nodes"] == [] and snap["job"] is None

        run_agent(code, coordinator.url, node_id="a")
        run_agent(code, coordinator.url, node_id="b")
        await coordinator.wait_nodes(code, 2)
        cfg = {"model": "tiny", "batch_size": 16, "epochs": 1, "steps_per_epoch": 8}
        job = (await coordinator.http.post(f"/api/rooms/{code}/jobs", json=cfg, headers=h)).json()

        seen = set()
        steps = []
        while True:
            ev = json.loads(await asyncio.wait_for(ws.recv(), 30))
            seen.add(ev["type"])
            if ev["type"] == "step":
                steps.append(ev["step"])
                assert ev["loss"] > 0 and ev["total_steps"] == 8
            if ev["type"] == "job" and ev["job"]["status"] == "completed":
                break
        assert {"node", "job", "step", "node_step"} <= seen
        assert steps == list(range(1, 9))
        assert job["job_id"]

    # unknown room is refused
    with pytest.raises(websockets.InvalidStatus):
        async with websockets.connect(coordinator.url.replace("http", "ws") + "/ws/dashboard/ZZZZZZ"):
            pass
