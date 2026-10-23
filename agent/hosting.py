"""App hosting on a pool node (the Deploy tab).

The coordinator can only say "run allow-listed app X" (see common/apps.py).
We start it as a subprocess (sample apps) or a Docker container (docker
apps), then try to make it public with a Cloudflare quick tunnel:

    cloudflared tunnel --url http://localhost:PORT   ->  https://xyz.trycloudflare.com

If cloudflared is missing we fall back to http://<this laptop's IP>:PORT.
If Docker is missing, docker apps fail with a clear message (no crash).
"""

import asyncio
import importlib.util
import logging
import re
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional

from common.apps import get_app
from common.schemas import AppStateMsg

logger = logging.getLogger("poolforge.hosting")

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "deploy" / "sample_apps"
TUNNEL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")

Report = Callable[[object], Awaitable[None]]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def lan_ip() -> str:
    """Best guess of this machine's LAN address (no packets are actually sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=5).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def cloudflared_available() -> bool:
    return shutil.which("cloudflared") is not None


@dataclass
class RunningApp:
    deployment_id: str
    app_id: str
    port: int
    procs: List[asyncio.subprocess.Process] = field(default_factory=list)
    container: Optional[str] = None
    url: str = ""
    watcher: Optional[asyncio.Task] = None
    stopping: bool = False


class AppManager:
    def __init__(self, report: Report):
        self.report = report
        self.apps: Dict[str, RunningApp] = {}
        self.has_docker = docker_available()
        self.has_cloudflared = cloudflared_available()

    async def _state(self, dep_id: str, status: str, url: Optional[str] = None, detail: str = "") -> None:
        await self.report(AppStateMsg(type="app_state", deployment_id=dep_id, status=status, url=url, detail=detail[:1900]))

    def _command(self, cfg: dict, port: int, dep_id: str, bind: str) -> List[str]:
        """Build the command line ONLY from allow-listed config (never from network input)."""
        if cfg["kind"] == "docker":
            if not self.has_docker:
                raise RuntimeError("Docker is not installed or not running on this node. Install Docker Desktop or pick a sample app.")
            return ["docker", "run", "--rm", "--name", f"poolforge-{dep_id}", "--memory", "256m",
                    "-p", f"{bind}:{port}:{cfg['container_port']}", cfg["image"]]
        app_dir = SAMPLES_DIR / cfg["sample"]
        if cfg["sample"] == "static_site":
            return [sys.executable, "-m", "http.server", str(port), "--bind", bind, "--directory", str(app_dir)]
        if cfg["sample"] == "flask_app":
            if importlib.util.find_spec("flask") is None:
                raise RuntimeError("Flask is not installed on this node. Run: pip install flask")
            return [sys.executable, str(app_dir / "app.py"), "--port", str(port), "--host", bind]
        raise RuntimeError(f"Unknown sample {cfg['sample']}")

    async def start(self, dep_id: str, app_id: str) -> None:
        if dep_id in self.apps:
            await self.stop(dep_id, report=False)
        try:
            cfg = get_app(app_id)
        except PermissionError as exc:
            await self._state(dep_id, "error", detail=str(exc))
            return
        port = free_port()
        # With a tunnel the app only needs to listen on localhost; without one
        # it must listen on all interfaces so others on the network can reach it.
        bind = "127.0.0.1" if self.has_cloudflared else "0.0.0.0"
        try:
            cmd = self._command(cfg, port, dep_id, bind)
        except RuntimeError as exc:
            await self._state(dep_id, "error", detail=str(exc))
            return
        await self._state(dep_id, "starting", detail=f"starting {cfg['name']} on port {port}")
        app = RunningApp(dep_id, app_id, port, container=f"poolforge-{dep_id}" if cfg["kind"] == "docker" else None)
        self.apps[dep_id] = app
        try:
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            app.procs.append(proc)
            # docker may need to pull the image first, so allow a generous wait
            await self._wait_for_port(port, proc, timeout=120 if cfg["kind"] == "docker" else 20)
            app.url, detail = await self._expose(app)
        except Exception as exc:
            await self.stop(dep_id, report=False)
            await self._state(dep_id, "error", detail=str(exc))
            return
        app.watcher = asyncio.create_task(self._watch(app))
        logger.info("App %s (%s) running at %s", dep_id, app_id, app.url)
        await self._state(dep_id, "running", url=app.url, detail=detail)

    async def _wait_for_port(self, port: int, proc: asyncio.subprocess.Process, timeout: float) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if proc.returncode is not None:
                err = (await proc.stderr.read()).decode(errors="replace")[-500:] if proc.stderr else ""
                raise RuntimeError(f"app exited with code {proc.returncode}: {err.strip()}")
            try:
                _r, w = await asyncio.open_connection("127.0.0.1", port)
                w.close()
                return
            except OSError:
                await asyncio.sleep(0.3)
        raise RuntimeError(f"app did not open port {port} within {timeout:.0f}s")

    async def _expose(self, app: RunningApp) -> "tuple[str, str]":
        """Return (public url, explanation)."""
        local = f"http://{lan_ip()}:{app.port}"
        if not self.has_cloudflared:
            return local, "cloudflared not installed: reachable only on this node's network (install cloudflared for a public URL)"
        tunnel = await asyncio.create_subprocess_exec(
            "cloudflared", "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{app.port}",
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        app.procs.append(tunnel)
        try:
            async def find_url() -> str:
                while True:
                    line = await tunnel.stderr.readline()  # cloudflared logs to stderr
                    if not line:
                        raise RuntimeError("cloudflared exited")
                    m = TUNNEL_RE.search(line.decode(errors="replace"))
                    if m:
                        return m.group(0)

            url = await asyncio.wait_for(find_url(), timeout=30)
            # keep draining stderr so cloudflared never blocks on a full pipe
            asyncio.create_task(self._drain(tunnel))
            return url, "public via Cloudflare quick tunnel"
        except Exception as exc:
            logger.warning("Tunnel failed (%s); falling back to local URL", exc)
            return local, f"tunnel failed ({exc}); reachable only on this node's network"

    @staticmethod
    async def _drain(proc: asyncio.subprocess.Process) -> None:
        while proc.stderr and await proc.stderr.readline():
            pass

    async def _watch(self, app: RunningApp) -> None:
        """Report an error if the app process dies on its own."""
        code = await app.procs[0].wait()
        if not app.stopping:
            self.apps.pop(app.deployment_id, None)
            await self.stop_processes(app)
            await self._state(app.deployment_id, "error", detail=f"app exited unexpectedly (code {code})")

    async def stop_processes(self, app: RunningApp) -> None:
        app.stopping = True
        if app.container and self.has_docker:
            await asyncio.to_thread(subprocess.run, ["docker", "stop", "-t", "2", app.container], capture_output=True)
        for p in app.procs:
            if p.returncode is None:
                p.terminate()
                try:
                    await asyncio.wait_for(p.wait(), 5)
                except asyncio.TimeoutError:
                    p.kill()

    async def stop(self, dep_id: str, report: bool = True) -> None:
        app = self.apps.pop(dep_id, None)
        if app:
            if app.watcher:
                app.watcher.cancel()
            await self.stop_processes(app)
        if report:
            await self._state(dep_id, "stopped", detail="stopped")

    async def stop_all(self) -> None:
        for dep_id in list(self.apps):
            await self.stop(dep_id, report=False)
