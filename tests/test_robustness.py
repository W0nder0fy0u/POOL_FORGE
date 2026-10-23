"""Phase 5: reconnect, checkpoint/resume, node-drop handling, pause/stop, validation."""

import asyncio
import time

import pytest
import uvicorn
import websockets

from coordinator.server import create_app
from tests.conftest import Coordinator, free_port

CFG = {"model": "mnist_mlp", "dataset": "synthetic", "batch_size": 32, "epochs": 1, "steps_per_epoch": 60,
       "lr": 0.003, "checkpoint_every": 5}


async def setup_pool(coordinator, run_agent, n=2, **kw):
    room = await coordinator.create_room()
    code, h = room["room_code"], {"X-Admin-Token": room["admin_token"]}
    agents = [run_agent(code, coordinator.url, node_id=f"n{i}", **kw) for i in range(n)]
    await coordinator.wait_nodes(code, n)
    return code, h, agents


async def job_status(c, code):
    return (await c.room(code))["job"]


async def wait_step(c, code, step, timeout=30):
    async def ok():
        job = await job_status(c, code)
        return job and job["last_step"] >= step
    await c.wait_for(ok, timeout)


async def test_pause_resume_and_stop(coordinator, run_agent):
    code, h, _ = await setup_pool(coordinator, run_agent)
    job = (await coordinator.http.post(f"/api/rooms/{code}/jobs", json=CFG, headers=h)).json()
    jid = job["job_id"]
    await wait_step(coordinator, code, 3)

    r = await coordinator.http.post(f"/api/rooms/{code}/jobs/{jid}/pause", headers=h)
    assert r.json()["status"] == "paused"
    await asyncio.sleep(1.0)  # let the in-flight step finish
    s1 = (await job_status(coordinator, code))["last_step"]
    await asyncio.sleep(1.5)
    assert (await job_status(coordinator, code))["last_step"] == s1  # really paused

    await coordinator.http.post(f"/api/rooms/{code}/jobs/{jid}/resume", headers=h)
    await wait_step(coordinator, code, s1 + 3)

    r = await coordinator.http.post(f"/api/rooms/{code}/jobs/{jid}/stop", headers=h)
    assert r.json()["status"] == "stopped"
    # a second stop is a clear error, not a crash
    assert (await coordinator.http.post(f"/api/rooms/{code}/jobs/{jid}/stop", headers=h)).status_code == 409

    async def all_idle():
        nodes = (await coordinator.room(code))["nodes"]
        return all(n["status"] == "idle" for n in nodes)
    await coordinator.wait_for(all_idle, 10)


async def test_only_one_active_job(coordinator, run_agent):
    code, h, _ = await setup_pool(coordinator, run_agent)
    assert (await coordinator.http.post(f"/api/rooms/{code}/jobs", json=CFG, headers=h)).status_code == 200
    assert (await coordinator.http.post(f"/api/rooms/{code}/jobs", json=CFG, headers=h)).status_code == 409


async def test_network_blip_resumes_from_checkpoint(coordinator, run_agent):
    """The control connection of one node drops; it reconnects by itself and the job resumes."""
    code, h, agents = await setup_pool(coordinator, run_agent)
    job = (await coordinator.http.post(f"/api/rooms/{code}/jobs", json=CFG, headers=h)).json()
    await wait_step(coordinator, code, 12)

    await agents[1].ws.close()  # simulate Wi-Fi dropping for a moment

    async def resumed():
        j = await job_status(coordinator, code)
        return j["status"] == "running" and (j["missing_nodes"] == [])
    async def waited():
        return (await job_status(coordinator, code))["status"] == "waiting"
    await coordinator.wait_for(waited, 5, 0.02)
    await coordinator.wait_for(resumed, 15)

    done = await coordinator.wait_job(code, job["job_id"], timeout=60)
    assert done["status"] == "completed", done
    metrics = (await coordinator.http.get(f"/api/rooms/{code}/jobs/{job['job_id']}")).json()["metrics"]
    steps = sorted({m["step"] for m in metrics})
    assert steps == list(range(1, 61))  # every step done, some twice after the resume


async def test_agent_process_restart_resumes(coordinator, run_agent, tmp_path):
    """A node's agent is killed and started again (same --node-id, same folder): training continues."""
    code, h, agents = await setup_pool(coordinator, run_agent)
    job = (await coordinator.http.post(f"/api/rooms/{code}/jobs", json=CFG, headers=h)).json()
    await wait_step(coordinator, code, 12)

    agents[0]._task.cancel()  # kill the agent of node n0
    await asyncio.gather(agents[0]._task, return_exceptions=True)
    await asyncio.sleep(0.5)
    assert (await job_status(coordinator, code))["status"] == "waiting"

    run_agent(code, coordinator.url, node_id="n0")  # restart it
    done = await coordinator.wait_job(code, job["job_id"], timeout=60)
    assert done["status"] == "completed", done


async def test_node_that_never_returns_fails_job_cleanly(coordinator, run_agent):
    code, h, agents = await setup_pool(coordinator, run_agent)
    job = (await coordinator.http.post(f"/api/rooms/{code}/jobs", json=CFG, headers=h)).json()
    await wait_step(coordinator, code, 3)
    agents[1]._task.cancel()
    done = await coordinator.wait_job(code, job["job_id"], timeout=20)
    assert done["status"] == "failed" and "did not come back" in done["detail"]

    async def survivor_idle():
        nodes = {n["node_id"]: n for n in (await coordinator.room(code))["nodes"]}
        return nodes["n0"]["status"] == "idle"
    await coordinator.wait_for(survivor_idle, 10)


async def test_heartbeat_timeout_marks_node_offline(coordinator, run_agent):
    code, h, agents = await setup_pool(coordinator, run_agent, n=1)
    hub = coordinator.app.state.hub
    node = hub.rooms[code].nodes["n0"]
    node.last_seen = time.time() - 60  # pretend the laptop froze a minute ago
    await hub.sweep()
    assert node.ws is None or node.last_seen > time.time() - 5  # dropped (and maybe already reconnected)


async def test_agent_waits_for_coordinator_with_backoff(tmp_path):
    """Agent started before the coordinator: it retries, then joins once the server is up."""
    from agent.worker import Agent

    port = free_port()
    app = create_app(str(tmp_path / "late.db"))
    url = f"http://127.0.0.1:{port}"
    agent_task = None
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", ws="websockets-sansio"))
    try:
        # Create the room through the app object before the server listens.
        app.state.db.create_room("LATE01", "admintoken")
        agent = Agent("LATE01", url, node_id="early", workdir=str(tmp_path))
        agent_task = asyncio.create_task(agent.run())
        await asyncio.sleep(2.5)  # agent is retrying now
        asyncio.create_task(server.serve())
        while not server.started:
            await asyncio.sleep(0.05)
        c = Coordinator(url, app)
        await c.wait_nodes("LATE01", 1, timeout=20)
        await c.http.aclose()
    finally:
        if agent_task:
            agent_task.cancel()
            await asyncio.gather(agent_task, return_exceptions=True)
        server.should_exit = True
        await asyncio.sleep(0.3)


async def test_garbage_control_messages_are_ignored(coordinator):
    code = (await coordinator.create_room())["room_code"]
    r = await coordinator.http.post(f"/api/rooms/{code}/join", json={"hostname": "h", "os": "o", "node_id": "g"})
    token = r.json()["node_token"]
    url = coordinator.url.replace("http", "ws") + f"/ws/agent/{code}/g?token={token}"
    async with websockets.connect(url) as ws:
        for junk in ("not json", '{"type":"exec","cmd":"rm -rf /"}', '{"type":"stats"}', "[]" * 10):
            await ws.send(junk)
        # still alive and usable after the junk:
        await ws.send('{"type":"stats","hostname":"h","os":"o","ram_total_gb":8,"ram_free_gb":4,"proc_ram_mb":1,'
                      '"cpu_percent":1,"status":"idle","transport":"relay"}')
        await asyncio.sleep(0.3)
    nodes = (await coordinator.room(code))["nodes"]
    assert nodes[0]["ram_free_gb"] == 4


async def test_create_room_rate_limit(coordinator):
    codes = [(await coordinator.http.post("/api/rooms")).status_code for _ in range(12)]
    assert codes.count(200) == 10 and codes[-1] == 429
