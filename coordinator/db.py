"""PoolForge Coordinator SQLite Database.

Stores persistent room state, connected nodes, training jobs, and historical metrics.
Uses standard Python sqlite3 with WAL mode for fast concurrent reads and writes.
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from common.schemas import NodeTelemetry


class CoordinatorDB:
    """Thread-safe SQLite database manager for coordinator state."""

    def __init__(self, db_path: str = "poolforge.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for high concurrency
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self) -> None:
        """Create database tables if they do not exist."""
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS rooms (
                    room_code TEXT PRIMARY KEY,
                    admin_token TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'idle'
                );

                CREATE TABLE IF NOT EXISTS nodes (
                    node_id TEXT PRIMARY KEY,
                    room_code TEXT NOT NULL,
                    hostname TEXT NOT NULL,
                    os_name TEXT NOT NULL,
                    ram_total_gb REAL NOT NULL,
                    ram_free_gb REAL NOT NULL,
                    cpu_percent REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'idle',
                    transport_type TEXT NOT NULL DEFAULT 'relay',
                    tcp_host TEXT,
                    tcp_port INTEGER,
                    assigned_layers TEXT,
                    last_heartbeat REAL NOT NULL,
                    FOREIGN KEY (room_code) REFERENCES rooms(room_code)
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    room_code TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'running',
                    created_at REAL NOT NULL,
                    finished_at REAL,
                    config_json TEXT NOT NULL,
                    FOREIGN KEY (room_code) REFERENCES rooms(room_code)
                );

                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    epoch INTEGER NOT NULL,
                    loss REAL NOT NULL,
                    step_time_ms REAL NOT NULL,
                    throughput REAL NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
                );
                """
            )

    def create_room(self, room_code: str, admin_token: str) -> None:
        """Insert a newly created room."""
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO rooms (room_code, admin_token, created_at, status) VALUES (?, ?, ?, 'idle')",
                (room_code, admin_token, time.time()),
            )

    def get_room(self, room_code: str) -> Optional[Dict[str, Any]]:
        """Fetch room record by code."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM rooms WHERE room_code = ?", (room_code,)).fetchone()
            return dict(row) if row else None

    def upsert_node(self, room_code: str, telemetry: NodeTelemetry) -> None:
        """Register or update telemetry for a worker node."""
        with self._get_connection() as conn:
            assigned_json = json.dumps(telemetry.assigned_layers) if telemetry.assigned_layers else None
            conn.execute(
                """
                INSERT INTO nodes (
                    node_id, room_code, hostname, os_name, ram_total_gb, ram_free_gb,
                    cpu_percent, status, transport_type, tcp_host, tcp_port, assigned_layers, last_heartbeat
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    room_code = excluded.room_code,
                    hostname = excluded.hostname,
                    os_name = excluded.os_name,
                    ram_total_gb = excluded.ram_total_gb,
                    ram_free_gb = excluded.ram_free_gb,
                    cpu_percent = excluded.cpu_percent,
                    status = excluded.status,
                    transport_type = excluded.transport_type,
                    tcp_host = excluded.tcp_host,
                    tcp_port = excluded.tcp_port,
                    assigned_layers = excluded.assigned_layers,
                    last_heartbeat = excluded.last_heartbeat
                """,
                (
                    telemetry.node_id,
                    room_code,
                    telemetry.hostname,
                    telemetry.os_name,
                    telemetry.ram_total_gb,
                    telemetry.ram_free_gb,
                    telemetry.cpu_percent,
                    telemetry.status,
                    telemetry.transport_type,
                    telemetry.tcp_host,
                    telemetry.tcp_port,
                    assigned_json,
                    time.time(),
                ),
            )

    def get_room_nodes(self, room_code: str, max_age_seconds: float = 10.0) -> List[NodeTelemetry]:
        """Retrieve all active nodes in a room with recent heartbeats."""
        with self._get_connection() as conn:
            cutoff = time.time() - max_age_seconds
            rows = conn.execute(
                "SELECT * FROM nodes WHERE room_code = ? AND last_heartbeat >= ? ORDER BY node_id ASC",
                (room_code, cutoff),
            ).fetchall()

            result = []
            for r in rows:
                assigned = json.loads(r["assigned_layers"]) if r["assigned_layers"] else None
                result.append(
                    NodeTelemetry(
                        node_id=r["node_id"],
                        hostname=r["hostname"],
                        os_name=r["os_name"],
                        ram_total_gb=r["ram_total_gb"],
                        ram_free_gb=r["ram_free_gb"],
                        cpu_percent=r["cpu_percent"],
                        status=r["status"],
                        transport_type=r["transport_type"],
                        tcp_host=r["tcp_host"],
                        tcp_port=r["tcp_port"],
                        assigned_layers=assigned,
                    )
                )
            return result

    def remove_node(self, node_id: str) -> None:
        """Remove a disconnected node."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM nodes WHERE node_id = ?", (node_id,))

    def create_job(self, job_id: str, room_code: str, model_name: str, config: Dict[str, Any]) -> None:
        """Create a new training job entry."""
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO jobs (job_id, room_code, model_name, status, created_at, config_json) VALUES (?, ?, ?, 'running', ?, ?)",
                (job_id, room_code, model_name, time.time(), json.dumps(config)),
            )
            conn.execute("UPDATE rooms SET status = 'running' WHERE room_code = ?", (room_code,))

    def update_job_status(self, job_id: str, status: str) -> None:
        """Update job status (e.g. paused, completed, stopped)."""
        with self._get_connection() as conn:
            finished = time.time() if status in ("completed", "stopped", "error") else None
            conn.execute(
                "UPDATE jobs SET status = ?, finished_at = ? WHERE job_id = ?",
                (status, finished, job_id),
            )

    def record_metric(
        self,
        job_id: str,
        step: int,
        epoch: int,
        loss: float,
        step_time_ms: float,
        throughput: float,
    ) -> None:
        """Record step training metric."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO metrics (job_id, step, epoch, loss, step_time_ms, throughput, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, step, epoch, loss, step_time_ms, throughput, time.time()),
            )

    def get_job_metrics(self, job_id: str) -> List[Dict[str, Any]]:
        """Fetch all historical step metrics for a job."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT step, epoch, loss, step_time_ms, throughput, created_at FROM metrics WHERE job_id = ? ORDER BY step ASC",
                (job_id,),
            ).fetchall()
            return [dict(r) for r in rows]
