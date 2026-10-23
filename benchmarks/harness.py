"""Benchmark harness: runs the REAL system (coordinator + agents as separate
processes) on this machine and drives it through the REST API, exactly like
the dashboard does. Every benchmark builds on Pool.run_job().
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
THREADS_PER_NODE = 4  # each simulated laptop gets 4 CPU threads (fair single-vs-pool comparison)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http(method: str, url: str, body: Optional[dict] = None, token: str = "") -> Any:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("X-Admin-Token", token)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(json.loads(e.read()).get("detail", str(e))) from None


class Pool:
    """A coordinator process plus any number of agent processes."""

    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="poolforge-bench-"))
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        env = {**os.environ, "POOLFORGE_PORT": str(self.port), "POOLFORGE_DB_PATH": str(self.tmp / "bench.db"),
               "LOG_LEVEL": "WARNING", "POOLFORGE_ROOMS_PER_MIN": "1000", "POOLFORGE_JOINS_PER_MIN": "1000"}
        self.coordinator = subprocess.Popen([sys.executable, "-m", "coordinator.server"], cwd=ROOT, env=env,
                                            stdout=open(self.tmp / "coordinator.log", "w"), stderr=subprocess.STDOUT)
        self.agents: List[subprocess.Popen] = []
        for _ in range(100):
            try:
                http("GET", f"{self.url}/health")
                break
            except Exception:
                time.sleep(0.2)

    def new_room(self) -> None:
        """Stop old agents and open a fresh room (each benchmark config gets its own)."""
        self.stop_agents()
        room = http("POST", f"{self.url}/api/rooms")
        self.code, self.token = room["room_code"], room["admin_token"]

    def add_agent(self, node_id: str, ram_cap_gb: Optional[float] = None, transport: str = "relay",
                  latency_ms: float = 0.0) -> None:
        cmd = [sys.executable, "-m", "agent.cli", "join", self.code, "--server", self.url, "--node-id", node_id,
               "--threads", str(THREADS_PER_NODE), "--workdir", str(self.tmp / node_id), "--transport", transport,
               "--simulate-latency-ms", str(latency_ms)]
        if transport == "tcp":
            cmd += ["--advertise-host", "127.0.0.1"]
        if ram_cap_gb is not None:
            cmd += ["--ram-cap-gb", str(ram_cap_gb)]
        log = open(self.tmp / f"{node_id}.log", "a")
        self.agents.append(subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT))

    def wait_nodes(self, n: int, timeout: float = 60) -> List[Dict[str, Any]]:
        end = time.time() + timeout
        while time.time() < end:
            nodes = [x for x in http("GET", f"{self.url}/api/rooms/{self.code}")["nodes"] if x["online"] and "ram_free_gb" in x]
            if len(nodes) >= n:
                return nodes
            time.sleep(0.3)
        raise RuntimeError(f"only some of {n} agents came online")

    def plan(self, cfg: dict) -> Dict[str, Any]:
        return http("POST", f"{self.url}/api/rooms/{self.code}/plan", cfg, self.token)

    def run_job(self, cfg: dict, warmup: int = 3, timeout: float = 900) -> Dict[str, Any]:
        """Start a job, wait for it, and summarise it. Steps < warmup are excluded from timing."""
        idle = {n["node_id"]: n["proc_ram_mb"] for n in self.wait_nodes(1)}
        job = http("POST", f"{self.url}/api/rooms/{self.code}/jobs", cfg, self.token)
        end = time.time() + timeout
        while time.time() < end:
            d = http("GET", f"{self.url}/api/rooms/{self.code}/jobs/{job['job_id']}")
            if d["status"] not in ("running", "paused", "waiting"):
                break
            time.sleep(0.5)
        if d["status"] != "completed":
            raise RuntimeError(f"job ended as {d['status']}: {d['detail']}")
        timed = [m for m in d["metrics"] if m["step"] > warmup]
        avg_ms = sum(m["step_time_ms"] for m in timed) / len(timed)
        nodes = sorted({m["node_id"] for m in d["node_metrics"]})
        per_node = {}
        for nid in nodes:
            rows = [m for m in d["node_metrics"] if m["node_id"] == nid and m["step"] > warmup]
            per_node[nid] = {
                "peak_ram_mb": max(m["proc_ram_mb"] for m in rows),
                "idle_ram_mb": idle.get(nid, 0.0),
                "bytes_sent_per_step": sum(m["bytes_sent"] for m in rows) / len(rows),
                "avg_send_ms": sum(m["avg_send_ms"] for m in rows) / len(rows),
            }
        return {
            "job_id": job["job_id"],
            "plan": job["plan"],
            "transport": job["transport"],
            "avg_step_ms": avg_ms,
            "steps_per_s": 1000.0 / avg_ms,
            "final_loss": d["metrics"][-1]["loss"],
            "per_node": per_node,
        }

    def stop_agents(self) -> None:
        for p in self.agents:
            p.terminate()
        for p in self.agents:
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()
        self.agents = []

    def close(self) -> None:
        self.stop_agents()
        self.coordinator.terminate()
        try:
            self.coordinator.wait(10)
        except subprocess.TimeoutExpired:
            self.coordinator.kill()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
