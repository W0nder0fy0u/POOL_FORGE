"""Phase 3: rooms, join, auth, relay, scheduler via API, and the acceptance test:
ONE API call trains a model across 2 local agents with an automatic split."""

import asyncio

import pytest
import torch
import websockets

from common.protocol import deserialize_tensor, serialize_tensor
from common.transports import RelayTransport


async def join(c, code, node_id=None):
    r = await c.http.post(f"/api/rooms/{code}/join", json={"hostname": "test", "os": "Linux", "node_id": node_id})
    return r


async def test_create_room_and_join(coordinator):
    room = await coordinator.create_room()
    code = room["room_code"]
    assert len(code) == 6 and code.isalnum() and code.isupper()
    assert room["join_url"].endswith(f"/join/{code}")
    assert room["join_command"] == f"poolforge join {code} --server {coordinator.url}"
    r = await join(coordinator, code, "laptop-a")
    assert r.status_code == 200 and r.json()["node_id"] == "laptop-a" and len(r.json()["node_token"]) > 20
    assert (await join(coordinator, "ZZZZZZ")).status_code == 404
    assert (await join(coordinator, "bad")).status_code == 404


async def test_admin_token_required(coordinator):
    code = (await coordinator.create_room())["room_code"]
    r = await coordinator.http.post(f"/api/rooms/{code}/jobs", json={"model": "tiny"})
    assert r.status_code == 403
    r = await coordinator.http.post(f"/api/rooms/{code}/jobs", json={"model": "tiny"}, headers={"X-Admin-Token": "nope"})
    assert r.status_code == 403


async def test_input_validation(coordinator):
    room = await coordinator.create_room()
    h = {"X-Admin-Token": room["admin_token"]}
    code = room["room_code"]
    bad = [{"model": "tiny", "batch_size": -1}, {"model": "tiny", "rm -rf": 1}, {"model": "tiny", "optimizer": "evil"}]
    for body in bad:
        assert (await coordinator.http.post(f"/api/rooms/{code}/jobs", json=body, headers=h)).status_code == 422
    r = await coordinator.http.post(f"/api/rooms/{code}/jobs", json={"model": "nope"}, headers=h)
    assert r.status_code == 400
    r = await coordinator.http.post(f"/api/rooms/{code}/join", json={"hostname": "x", "os": "y", "node_id": "../../etc"})
    assert r.status_code == 422


async def test_lock_room(coordinator):
    room = await coordinator.create_room()
    code = room["room_code"]
    r = await coordinator.http.post(f"/api/rooms/{code}/lock", json={"locked": True}, headers={"X-Admin-Token": room["admin_token"]})
    assert r.status_code == 200
    assert (await join(coordinator, code)).status_code == 403


async def test_join_rate_limited(coordinator):
    code = (await coordinator.create_room())["room_code"]
    codes = [(await join(coordinator, code)).status_code for _ in range(25)]
    assert 429 in codes and codes[0] == 200


async def test_websockets_need_valid_token(coordinator):
    code = (await coordinator.create_room())["room_code"]
    await join(coordinator, code, "n1")
    ws_url = coordinator.url.replace("http", "ws")
    for path in (f"/ws/agent/{code}/n1?token=wrong", f"/ws/relay/{code}/n1?token=wrong", f"/ws/relay/{code}/stranger"):
        with pytest.raises(websockets.InvalidStatus):
            async with websockets.connect(ws_url + path):
                pass


async def test_relay_50mb_bit_exact_between_room_members(coordinator):
    code = (await coordinator.create_room())["room_code"]
    a = (await join(coordinator, code, "a")).json()
    b = (await join(coordinator, code, "b")).json()
    ta = RelayTransport("a", coordinator.url, code, a["node_token"])
    tb = RelayTransport("b", coordinator.url, code, b["node_token"])
    await ta.start()
    await tb.start()
    try:
        t = torch.randn(50 * 1024 * 1024 // 4)
        await ta.send("b", serialize_tensor(t, job_id="j"))
        sender, frame = await tb.recv(timeout=60)
        assert sender == "a" and torch.equal(deserialize_tensor(frame)[0], t)
        assert ta.bytes_sent > 50 * 1024 * 1024
    finally:
        await ta.stop()
        await tb.stop()


async def test_relay_does_not_cross_rooms(coordinator):
    code1 = (await coordinator.create_room())["room_code"]
    code2 = (await coordinator.create_room())["room_code"]
    a = (await join(coordinator, code1, "a")).json()
    b = (await join(coordinator, code2, "b")).json()
    ta = RelayTransport("a", coordinator.url, code1, a["node_token"])
    tb = RelayTransport("b", coordinator.url, code2, b["node_token"])
    await ta.start()
    await tb.start()
    try:
        await ta.send("b", b"sneaky")
        with pytest.raises(asyncio.TimeoutError):
            await tb.recv(timeout=0.5)
    finally:
        await ta.stop()
        await tb.stop()


async def test_one_api_call_trains_across_two_agents(coordinator, run_agent):
    """Acceptance criterion of Phase 3."""
    room = await coordinator.create_room()
    code, h = room["room_code"], {"X-Admin-Token": room["admin_token"]}
    run_agent(code, coordinator.url, node_id="small", ram_cap_gb=0.5)
    run_agent(code, coordinator.url, node_id="big", ram_cap_gb=2.0)
    await coordinator.wait_nodes(code, 2)

    cfg = {"model": "mnist_mlp", "dataset": "synthetic", "batch_size": 32, "epochs": 2, "steps_per_epoch": 15,
           "lr": 0.003, "checkpoint_every": 10}
    r = await coordinator.http.post(f"/api/rooms/{code}/jobs", json=cfg, headers=h)
    assert r.status_code == 200, r.text
    job = r.json()
    plan = job["plan"]
    assert [p["node_id"] for p in plan] == ["big", "small"]  # auto split, biggest node first
    assert plan[0]["end_layer"] == plan[1]["start_layer"] and plan[1]["end_layer"] == 6
    assert job["transport"] == "relay"

    done = await coordinator.wait_job(code, job["job_id"])
    assert done["status"] == "completed", done
    detail = (await coordinator.http.get(f"/api/rooms/{code}/jobs/{job['job_id']}")).json()
    losses = [m["loss"] for m in detail["metrics"]]
    assert len(losses) == 30
    assert sum(losses[-5:]) < sum(losses[:5])  # it learned
    nodes_reporting = {m["node_id"] for m in detail["node_metrics"]}
    assert nodes_reporting == {"big", "small"}
    assert all(m["bytes_sent"] > 0 for m in detail["node_metrics"])

    csv_text = (await coordinator.http.get(f"/api/rooms/{code}/jobs/{job['job_id']}/export?format=csv")).text
    assert csv_text.startswith("step,epoch,loss") and len(csv_text.splitlines()) == 1 + 30 * 2


async def test_pool_too_small_via_api(coordinator, run_agent):
    room = await coordinator.create_room()
    code, h = room["room_code"], {"X-Admin-Token": room["admin_token"]}
    run_agent(code, coordinator.url, node_id="tiny-laptop", ram_cap_gb=0.05)
    await coordinator.wait_nodes(code, 1)
    cfg = {"model": "large_mlp", "model_options": {"width": 1024, "depth": 8}}
    r = await coordinator.http.post(f"/api/rooms/{code}/plan", json=cfg, headers=h)
    assert r.status_code == 400 and "pool too small" in r.json()["detail"]


async def test_direct_tcp_transport_end_to_end(coordinator, run_agent):
    room = await coordinator.create_room()
    code, h = room["room_code"], {"X-Admin-Token": room["admin_token"]}
    for name in ("t1", "t2", "t3"):
        run_agent(code, coordinator.url, node_id=name, transport="tcp", advertise_host="127.0.0.1")
    await coordinator.wait_nodes(code, 3)
    cfg = {"model": "tiny", "batch_size": 16, "microbatches": 2, "epochs": 1, "steps_per_epoch": 10, "lr": 0.01}
    job = (await coordinator.http.post(f"/api/rooms/{code}/jobs", json=cfg, headers=h)).json()
    assert job["transport"] == "tcp" and len(job["plan"]) == 3
    done = await coordinator.wait_job(code, job["job_id"])
    assert done["status"] == "completed", done
