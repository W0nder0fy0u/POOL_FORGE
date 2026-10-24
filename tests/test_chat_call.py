"""Chat and voice-call signalling over the dashboard WebSocket."""

import asyncio
import json

import pytest
import websockets


async def open_dashboard(c, code):
    """Connect like a browser tab; returns (ws, snapshot, my peer_id)."""
    ws = await websockets.connect(c.url.replace("http", "ws") + f"/ws/dashboard/{code}")
    snap = json.loads(await ws.recv())
    hello = json.loads(await ws.recv())
    assert snap["type"] == "snapshot" and hello["type"] == "hello"
    return ws, snap, hello["peer_id"]


async def next_of(ws, kind, timeout=3.0):
    """Next event of one type (skips others, e.g. node updates)."""
    async def find():
        while True:
            ev = json.loads(await ws.recv())
            if ev["type"] == kind:
                return ev
    return await asyncio.wait_for(find(), timeout)


async def test_chat_is_broadcast_and_kept_in_snapshot(coordinator):
    code = (await coordinator.create_room())["room_code"]
    a, snap, _ = await open_dashboard(coordinator, code)
    b, _, _ = await open_dashboard(coordinator, code)
    assert snap["chat"] == [] and snap["call"] == []
    try:
        await a.send(json.dumps({"type": "chat", "name": "Asha", "text": "hello pool"}))
        for ws in (a, b):
            ev = await next_of(ws, "chat")
            assert ev["name"] == "Asha" and ev["text"] == "hello pool"
        c, snap, _ = await open_dashboard(coordinator, code)  # a late joiner sees the history
        assert [m["text"] for m in snap["chat"]] == ["hello pool"]
        await c.close()
    finally:
        await a.close()
        await b.close()


async def test_bad_chat_messages_are_ignored(coordinator):
    code = (await coordinator.create_room())["room_code"]
    a, _, _ = await open_dashboard(coordinator, code)
    try:
        for bad in ({"type": "chat", "name": "x", "text": ""}, {"type": "chat", "name": "x", "text": "y" * 501},
                    {"type": "chat", "name": "x", "text": "hi", "admin": True}, {"type": "start_job"}):
            await a.send(json.dumps(bad))
        await a.send("not json")
        with pytest.raises(asyncio.TimeoutError):
            await next_of(a, "chat", timeout=0.5)
    finally:
        await a.close()


async def test_chat_rate_limited(coordinator):
    code = (await coordinator.create_room())["room_code"]
    a, _, _ = await open_dashboard(coordinator, code)
    try:
        for i in range(30):
            await a.send(json.dumps({"type": "chat", "name": "spam", "text": str(i)}))
        await asyncio.sleep(0.5)
        snap_ws, snap, _ = await open_dashboard(coordinator, code)
        assert len(snap["chat"]) == 20
        await snap_ws.close()
    finally:
        await a.close()


async def test_call_signals_only_reach_the_target_in_the_same_room(coordinator):
    code = (await coordinator.create_room())["room_code"]
    other = (await coordinator.create_room())["room_code"]
    a, _, pa = await open_dashboard(coordinator, code)
    b, _, pb = await open_dashboard(coordinator, code)
    watcher, _, _ = await open_dashboard(coordinator, code)  # in the room but not in the call
    x, _, px = await open_dashboard(coordinator, other)
    try:
        for ws, name in ((a, "A"), (b, "B"), (x, "X")):
            await ws.send(json.dumps({"type": "call_join", "name": name}))
        call = await next_of(watcher, "call")
        while len(call["call"]) < 2:
            call = await next_of(watcher, "call")
        assert {p["name"] for p in call["call"]} == {"A", "B"}  # X is in another room

        offer = {"sdp": "v=0...", "kind": "offer"}
        await a.send(json.dumps({"type": "signal", "to": pb, "data": offer}))
        ev = await next_of(b, "signal")
        assert ev["from"] == pa and ev["data"] == offer

        await a.send(json.dumps({"type": "signal", "to": px, "data": offer}))  # other room: dropped
        with pytest.raises(asyncio.TimeoutError):
            await next_of(x, "signal", timeout=0.5)

        await b.close()  # closing the tab leaves the call
        call = await next_of(watcher, "call")
        while len(call["call"]) != 1:
            call = await next_of(watcher, "call")
        assert call["call"][0]["peer_id"] == pa
    finally:
        for ws in (a, watcher, x):
            await ws.close()


async def test_ice_servers_turn_only_when_configured(coordinator, monkeypatch):
    ice = (await coordinator.http.get("/api/ice")).json()["iceServers"]
    assert len(ice) == 1 and ice[0]["urls"].startswith("stun:")
    monkeypatch.setenv("POOLFORGE_TURN_URL", "turn:turn.example.com:3478")
    monkeypatch.setenv("POOLFORGE_TURN_USER", "u")
    monkeypatch.setenv("POOLFORGE_TURN_PASS", "p")
    ice = (await coordinator.http.get("/api/ice")).json()["iceServers"]
    assert ice[1] == {"urls": ["turn:turn.example.com:3478"], "username": "u", "credential": "p"}
