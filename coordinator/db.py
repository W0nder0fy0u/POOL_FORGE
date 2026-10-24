"""SQLite persistence for the coordinator (rooms, nodes, jobs, metrics, apps).

Live things (open WebSockets, latest RAM/CPU numbers) are kept in memory in
coordinator/state.py; this database keeps what should survive a restart and
what the Reports page exports.

Secrets (admin tokens, node tokens) are stored only as SHA-256 hashes, so a
leaked database file does not leak working tokens.
"""

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS rooms (
    code TEXT PRIMARY KEY,
    admin_hash TEXT NOT NULL,
    locked INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS nodes (
    node_id TEXT NOT NULL,
    room TEXT NOT NULL,
    token_hash TEXT NOT NULL,
    hostname TEXT NOT NULL,
    os TEXT NOT NULL,
    joined_at REAL NOT NULL,
    PRIMARY KEY (room, node_id)
);
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    room TEXT NOT NULL,
    model TEXT NOT NULL,
    config TEXT NOT NULL,
    plan TEXT NOT NULL,
    transport TEXT NOT NULL,
    status TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    finished_at REAL
);
CREATE TABLE IF NOT EXISTS metrics (
    job_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    epoch INTEGER NOT NULL,
    loss REAL,
    step_time_ms REAL NOT NULL,
    samples_per_s REAL NOT NULL,
    t REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS node_metrics (
    job_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    stage INTEGER NOT NULL,
    step INTEGER NOT NULL,
    step_time_ms REAL NOT NULL,
    bytes_sent INTEGER NOT NULL,
    bytes_received INTEGER NOT NULL,
    avg_send_ms REAL NOT NULL,
    proc_ram_mb REAL NOT NULL,
    t REAL NOT NULL,
    gpu_mem_mb REAL
);
CREATE TABLE IF NOT EXISTS deployments (
    id TEXT PRIMARY KEY,
    room TEXT NOT NULL,
    node_id TEXT NOT NULL,
    app_id TEXT NOT NULL,
    status TEXT NOT NULL,
    url TEXT,
    detail TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_metrics_job ON metrics(job_id, step);
CREATE INDEX IF NOT EXISTS idx_node_metrics_job ON node_metrics(job_id, step);
"""


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class DB:
    """Tiny wrapper around one sqlite3 connection (guarded by a lock)."""

    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        if path != ":memory:":
            self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        # Databases made before GPU support lack this column: add it once.
        cols = [r["name"] for r in self.conn.execute("PRAGMA table_info(node_metrics)")]
        if "gpu_mem_mb" not in cols:
            self.conn.execute("ALTER TABLE node_metrics ADD COLUMN gpu_mem_mb REAL")
        self.lock = threading.Lock()
        # Jobs that were running when the coordinator stopped can't continue.
        self.execute(
            "UPDATE jobs SET status='interrupted', finished_at=? WHERE status IN ('running','paused','waiting')",
            (time.time(),),
        )

    def execute(self, sql: str, args: tuple = ()) -> None:
        with self.lock, self.conn:
            self.conn.execute(sql, args)

    def query(self, sql: str, args: tuple = ()) -> List[Dict[str, Any]]:
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def one(self, sql: str, args: tuple = ()) -> Optional[Dict[str, Any]]:
        rows = self.query(sql, args)
        return rows[0] if rows else None

    # ---------------------------------------------------------------- rooms
    def create_room(self, code: str, admin_token: str) -> None:
        self.execute("INSERT INTO rooms VALUES (?, ?, 0, ?)", (code, hash_token(admin_token), time.time()))

    def get_room(self, code: str) -> Optional[Dict[str, Any]]:
        return self.one("SELECT * FROM rooms WHERE code=?", (code,))

    def set_locked(self, code: str, locked: bool) -> None:
        self.execute("UPDATE rooms SET locked=? WHERE code=?", (int(locked), code))

    # ---------------------------------------------------------------- nodes
    def add_node(self, room: str, node_id: str, token: str, hostname: str, os_name: str) -> None:
        self.execute(
            "INSERT OR REPLACE INTO nodes VALUES (?, ?, ?, ?, ?, ?)",
            (node_id, room, hash_token(token), hostname, os_name, time.time()),
        )

    def get_node(self, room: str, node_id: str) -> Optional[Dict[str, Any]]:
        return self.one("SELECT * FROM nodes WHERE room=? AND node_id=?", (room, node_id))

    def list_nodes(self, room: str) -> List[Dict[str, Any]]:
        return self.query("SELECT node_id, hostname, os, joined_at FROM nodes WHERE room=?", (room,))

    # ----------------------------------------------------------------- jobs
    def create_job(self, job_id: str, room: str, model: str, config: dict, plan: list, transport: str) -> None:
        self.execute(
            "INSERT INTO jobs (job_id, room, model, config, plan, transport, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 'running', ?)",
            (job_id, room, model, json.dumps(config), json.dumps(plan), transport, time.time()),
        )

    def set_job_status(self, job_id: str, status: str, detail: str = "") -> None:
        finished = time.time() if status in ("completed", "stopped", "failed") else None
        self.execute(
            "UPDATE jobs SET status=?, detail=?, finished_at=COALESCE(?, finished_at) WHERE job_id=?",
            (status, detail, finished, job_id),
        )

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        row = self.one("SELECT * FROM jobs WHERE job_id=?", (job_id,))
        if row:
            row["config"] = json.loads(row["config"])
            row["plan"] = json.loads(row["plan"])
        return row

    def list_jobs(self, room: str) -> List[Dict[str, Any]]:
        return self.query(
            "SELECT job_id, model, transport, status, detail, created_at, finished_at FROM jobs "
            "WHERE room=? ORDER BY created_at DESC",
            (room,),
        )

    # -------------------------------------------------------------- metrics
    def add_metric(self, job_id: str, step: int, epoch: int, loss: Optional[float], step_ms: float, sps: float) -> None:
        self.execute("INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?, ?)", (job_id, step, epoch, loss, step_ms, sps, time.time()))

    def add_node_metric(self, job_id: str, node_id: str, m: Any) -> None:
        self.execute(
            "INSERT INTO node_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (job_id, node_id, m.stage, m.step, m.step_time_ms, m.bytes_sent, m.bytes_received,
             m.avg_send_ms, m.proc_ram_mb, time.time(), m.gpu_mem_mb),
        )

    def job_metrics(self, job_id: str) -> List[Dict[str, Any]]:
        return self.query("SELECT step, epoch, loss, step_time_ms, samples_per_s, t FROM metrics WHERE job_id=? ORDER BY step", (job_id,))

    def job_node_metrics(self, job_id: str) -> List[Dict[str, Any]]:
        return self.query(
            "SELECT node_id, stage, step, step_time_ms, bytes_sent, bytes_received, avg_send_ms, proc_ram_mb, gpu_mem_mb, t "
            "FROM node_metrics WHERE job_id=? ORDER BY step, stage",
            (job_id,),
        )

    # ---------------------------------------------------------- deployments
    def add_deployment(self, dep_id: str, room: str, node_id: str, app_id: str) -> None:
        self.execute(
            "INSERT INTO deployments (id, room, node_id, app_id, status, created_at) VALUES (?, ?, ?, ?, 'starting', ?)",
            (dep_id, room, node_id, app_id, time.time()),
        )

    def update_deployment(self, dep_id: str, status: str, url: Optional[str], detail: str) -> None:
        self.execute("UPDATE deployments SET status=?, url=COALESCE(?, url), detail=? WHERE id=?", (status, url, detail, dep_id))

    def get_deployment(self, dep_id: str) -> Optional[Dict[str, Any]]:
        return self.one("SELECT * FROM deployments WHERE id=?", (dep_id,))

    def list_deployments(self, room: str) -> List[Dict[str, Any]]:
        return self.query("SELECT * FROM deployments WHERE room=? ORDER BY created_at DESC", (room,))
