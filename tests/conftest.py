"""Shared fixtures: a real coordinator (uvicorn on a free port) + helper to run agents."""

import asyncio
import os
import socket

os.environ["POOLFORGE_NO_DEFAULT_APP"] = "1"  # don't create data/poolforge.db on import

import httpx
import pytest
import uvicorn

from coordinator.server import create_app


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Coordinator:
    def __init__(self, url: str, app):
        self.url = url
        self.app = app
        self.http = httpx.AsyncClient(base_url=url, timeout=30)

    async def create_room(self):
        r = await self.http.post("/api/rooms")
        r.raise_for_status()
        return r.json()

    async def room(self, code):
        return (await self.http.get(f"/api/rooms/{code}")).json()

    async def wait_for(self, predicate, timeout=30.0, interval=0.1):
        loop = asyncio.get_running_loop()
        end = loop.time() + timeout
        while loop.time() < end:
            value = await predicate()
            if value:
                return value
            await asyncio.sleep(interval)
        raise AssertionError("condition not met in time")

    async def wait_nodes(self, code, n, timeout=30.0):
        async def ok():
            nodes = [x for x in (await self.room(code))["nodes"] if x["online"] and "ram_free_gb" in x]
            return len(nodes) >= n
        await self.wait_for(ok, timeout)

    async def wait_job(self, code, job_id, statuses=("completed", "failed", "stopped"), timeout=60.0):
        async def done():
            jobs = (await self.http.get(f"/api/rooms/{code}/jobs")).json()
            job = next((j for j in jobs if j["job_id"] == job_id), None)
            return job if job and job["status"] in statuses else None
        return await self.wait_for(done, timeout, 0.2)


@pytest.fixture
async def coordinator(tmp_path):
    app = create_app(str(tmp_path / "test.db"), drop_grace_s=8)
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", ws_max_size=600 * 1024 * 1024, ws="websockets-sansio"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    c = Coordinator(f"http://127.0.0.1:{port}", app)
    yield c
    await c.http.aclose()
    server.should_exit = True
    await asyncio.wait_for(task, 10)


@pytest.fixture
async def run_agent(tmp_path):
    """Start an Agent as a background task; all are stopped at test end."""
    from agent.worker import Agent

    tasks = []

    def start(code, server, **kw):
        kw.setdefault("workdir", str(tmp_path))
        agent = Agent(code, server, **kw)
        agent.apps.has_cloudflared = False  # never open real tunnels in tests
        task = asyncio.create_task(agent.run())
        tasks.append(task)
        agent._task = task
        return agent

    yield start
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
