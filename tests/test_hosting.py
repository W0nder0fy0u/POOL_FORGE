"""Phase 6: hosting allow-listed apps on pool nodes (Deploy tab)."""

import os
import stat
import sys

import httpx
import pytest

from agent.hosting import TUNNEL_RE, AppManager
from common.apps import get_app


async def deploy_and_wait(c, code, h, body, statuses=("running", "error")):
    dep = (await c.http.post(f"/api/rooms/{code}/deployments", json=body, headers=h)).json()

    async def done():
        deps = (await c.http.get(f"/api/rooms/{code}/deployments")).json()
        d = next(x for x in deps if x["id"] == dep["id"])
        return d if d["status"] in statuses else None
    return await c.wait_for(done, 40, 0.2)


async def pool(c, run_agent, caps=(None,)):
    room = await c.create_room()
    code, h = room["room_code"], {"X-Admin-Token": room["admin_token"]}
    agents = [run_agent(code, c.url, node_id=f"h{i}", ram_cap_gb=cap) for i, cap in enumerate(caps)]
    await c.wait_nodes(code, len(caps))
    return code, h, agents


def test_allow_list_blocks_everything_else():
    assert get_app("static_site")["kind"] == "sample"
    for bad in ("rm -rf /", "../../bin/sh", "nginx; curl evil", ""):
        with pytest.raises(PermissionError):
            get_app(bad)


def test_tunnel_url_regex():
    line = "2026-10-23T10:00:00Z INF |  https://quiet-river-1234.trycloudflare.com  |"
    assert TUNNEL_RE.search(line).group(0) == "https://quiet-river-1234.trycloudflare.com"
    assert TUNNEL_RE.search("https://evil.com/?x=.trycloudflare.com") is None


async def test_api_rejects_unlisted_app_and_needs_admin(coordinator, run_agent):
    code, h, _ = await pool(coordinator, run_agent)
    r = await coordinator.http.post(f"/api/rooms/{code}/deployments", json={"app_id": "bash"}, headers=h)
    assert r.status_code == 403
    r = await coordinator.http.post(f"/api/rooms/{code}/deployments", json={"app_id": "static_site"})
    assert r.status_code == 403


async def test_deploy_static_site_stop_restart(coordinator, run_agent):
    code, h, agents = await pool(coordinator, run_agent)
    d = await deploy_and_wait(coordinator, code, h, {"app_id": "static_site"})
    assert d["status"] == "running", d
    assert "cloudflared not installed" in d["detail"]  # clear fallback message
    port = d["url"].rsplit(":", 1)[1]
    async with httpx.AsyncClient() as client:
        page = await client.get(f"http://127.0.0.1:{port}/")
    assert page.status_code == 200 and "<html" in page.text.lower()

    async def node_hosting():
        return (await coordinator.room(code))["nodes"][0]["status"] == "hosting"
    await coordinator.wait_for(node_hosting, 10)

    await coordinator.http.post(f"/api/rooms/{code}/deployments/{d['id']}/restart", headers=h)  # same id

    async def running_again():
        deps = (await coordinator.http.get(f"/api/rooms/{code}/deployments")).json()
        return deps[0]["status"] == "running" and deps[0]["detail"] != "restarting"
    await coordinator.wait_for(running_again, 30)

    await coordinator.http.post(f"/api/rooms/{code}/deployments/{d['id']}/stop", headers=h)

    async def stopped():
        deps = (await coordinator.http.get(f"/api/rooms/{code}/deployments")).json()
        return deps[0]["status"] == "stopped"
    await coordinator.wait_for(stopped, 10)
    assert not agents[0].apps.apps


async def test_deploy_flask_app(coordinator, run_agent):
    pytest.importorskip("flask")
    code, h, agents = await pool(coordinator, run_agent)
    d = await deploy_and_wait(coordinator, code, h, {"app_id": "flask_app"})
    assert d["status"] == "running", d
    port = d["url"].rsplit(":", 1)[1]
    async with httpx.AsyncClient() as client:
        info = (await client.get(f"http://127.0.0.1:{port}/api/info")).json()
    assert "host" in info
    await agents[0].apps.stop_all()


async def test_docker_missing_gives_clear_error(coordinator, run_agent):
    code, h, agents = await pool(coordinator, run_agent)
    agents[0].apps.has_docker = False
    d = await deploy_and_wait(coordinator, code, h, {"app_id": "nginx"})
    assert d["status"] == "error" and "Docker is not installed" in d["detail"]


async def test_auto_picks_node_with_most_free_ram(coordinator, run_agent):
    code, h, agents = await pool(coordinator, run_agent, caps=(0.5, 3.0))
    d = await deploy_and_wait(coordinator, code, h, {"app_id": "static_site"})
    assert d["node_id"] == "h1"
    for a in agents:
        await a.apps.stop_all()


@pytest.mark.skipif(sys.platform == "win32", reason="fake cloudflared is a shell script")
async def test_cloudflared_tunnel_url_is_reported(tmp_path, monkeypatch):
    """A fake `cloudflared` prints a quick-tunnel URL like the real one does."""
    fake = tmp_path / "cloudflared"
    fake.write_text("#!/bin/sh\necho 'INF Requesting new quick Tunnel...' >&2\n"
                    "echo 'INF |  https://happy-cat-42.trycloudflare.com  |' >&2\nsleep 30\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    reports = []

    async def report(msg):
        reports.append(msg)

    mgr = AppManager(report)
    assert mgr.has_cloudflared
    await mgr.start("app-test", "static_site")
    try:
        final = reports[-1]
        assert final.status == "running" and final.url == "https://happy-cat-42.trycloudflare.com"
        assert "Cloudflare" in final.detail
    finally:
        procs = [p for a in mgr.apps.values() for p in a.procs]
        await mgr.stop_all()
    assert procs and all(p.returncode is not None for p in procs)  # app + tunnel stopped
