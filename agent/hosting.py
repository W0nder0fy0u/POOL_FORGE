"""PoolForge Edge Application Hosting Manager.

Spawns allow-listed apps in lightweight containers or isolated subprocesses,
exposes them through Cloudflare quick tunnels (or local IP fallback),
and reports live URLs to the Coordinator.
"""

import asyncio
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from common.apps import get_allow_listed_app

logger = logging.getLogger("poolforge.hosting")


def find_free_port(start_port: int = 8080) -> int:
    """Find an available TCP port on localhost."""
    for port in range(start_port, start_port + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start_port


class AppRunner:
    """Manages hosted application subprocesses and public tunnels."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        # app_id -> metadata dict
        self._running_apps: Dict[str, Dict[str, Any]] = {}

    def is_docker_available(self) -> bool:
        """Check if Docker CLI binary is installed and daemon is responding."""
        docker_bin = shutil.which("docker")
        if not docker_bin:
            return False
        try:
            res = subprocess.run([docker_bin, "info"], capture_output=True, timeout=2.0)
            return res.returncode == 0
        except Exception:
            return False

    def is_cloudflared_available(self) -> bool:
        """Check if Cloudflare quick tunnel CLI is installed."""
        return shutil.which("cloudflared") is not None

    async def start_app(self, app_id: str, requested_port: Optional[int] = None) -> Dict[str, Any]:
        """Launch an allow-listed application and establish tunnel/local URL."""
        app_cfg = get_allow_listed_app(app_id)
        port = requested_port or find_free_port(app_cfg["default_port"])

        # Stop existing instance if already running
        if app_id in self._running_apps:
            await self.stop_app(app_id)

        logger.info(f"Starting hosted app '{app_id}' on port {port}...")

        # Determine launch mechanism
        entry_type = app_cfg.get("entrypoint_type")
        env = dict(os.environ)
        env["PORT"] = str(port)

        proc = None
        if entry_type == "python_module":
            static_dir = Path(__file__).resolve().parent.parent / "deploy" / "sample_apps" / "static_site"
            proc = subprocess.Popen(
                [sys.executable, "-m", app_cfg["module"], str(port)],
                cwd=str(static_dir),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif entry_type == "python_script":
            script_path = Path(__file__).resolve().parent.parent / app_cfg["script_path"]
            proc = subprocess.Popen(
                [sys.executable, str(script_path)],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        # Wait briefly for process to bind port
        await asyncio.sleep(0.3)

        # Establish public tunnel or local IP fallback
        tunnel_url = None
        tunnel_proc = None

        if self.is_cloudflared_available():
            logger.info("cloudflared detected! Spawning Cloudflare quick tunnel...")
            try:
                tunnel_proc = subprocess.Popen(
                    ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                # Try to parse generated trycloudflare.com URL from output
                for _ in range(15):
                    await asyncio.sleep(0.4)
                    line = tunnel_proc.stdout.readline() if tunnel_proc.stdout else ""
                    match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
                    if match:
                        tunnel_url = match.group(0)
                        logger.info(f"Cloudflare quick tunnel ready: {tunnel_url}")
                        break
            except Exception as e:
                logger.warning(f"Failed to start Cloudflare tunnel: {e}")

        if not tunnel_url:
            # Fallback to local machine IP:port
            logger.info(f"cloudflared missing or inactive; using local direct endpoint http://127.0.0.1:{port}")
            tunnel_url = f"http://127.0.0.1:{port}"

        app_info = {
            "app_id": app_id,
            "name": app_cfg["name"],
            "node_id": self.node_id,
            "port": port,
            "url": tunnel_url,
            "status": "running",
            "process": proc,
            "tunnel_process": tunnel_proc,
        }
        self._running_apps[app_id] = app_info
        return {
            "app_id": app_id,
            "name": app_cfg["name"],
            "node_id": self.node_id,
            "port": port,
            "url": tunnel_url,
            "status": "running",
        }

    async def stop_app(self, app_id: str) -> bool:
        """Stop a running hosted app and its tunnel."""
        info = self._running_apps.pop(app_id, None)
        if not info:
            return False

        proc = info.get("process")
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except Exception:
                proc.kill()

        tunnel_proc = info.get("tunnel_process")
        if tunnel_proc and tunnel_proc.poll() is None:
            tunnel_proc.terminate()
            try:
                tunnel_proc.wait(timeout=1.0)
            except Exception:
                tunnel_proc.kill()

        logger.info(f"Stopped hosted app '{app_id}'")
        return True

    def list_apps(self) -> list[Dict[str, Any]]:
        """List active apps on this node."""
        result = []
        for a_id, info in self._running_apps.items():
            result.append(
                {
                    "app_id": a_id,
                    "name": info["name"],
                    "node_id": self.node_id,
                    "port": info["port"],
                    "url": info["url"],
                    "status": info["status"],
                }
            )
        return result
