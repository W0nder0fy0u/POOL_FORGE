"""Live coordinator state and the training-job lifecycle.

The Hub keeps everything that only exists while the coordinator runs: open
WebSockets, the latest stats of every node, the active job of every room.
Durable data (rooms, jobs, metrics) goes to SQLite through coordinator/db.py.

Job lifecycle:

    start --> running <--> paused            (admin Pause / Resume)
                 |
                 | a node in the job disconnects
                 v
              waiting --(node back within grace)--> running  (all stages
                 |                                   reload the newest common
                 | grace period expires              checkpoint and continue)
                 v
              failed
    running --(last stage reports done)--> completed
    any active state --(admin Stop)--> stopped
"""

import asyncio
import json
import logging
import secrets
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from common.model_zoo import get_model
from common.schemas import (
    AssignMsg,
    CallJoin,
    CallLeave,
    ChatIn,
    ControlMsg,
    HyperParams,
    JobStateMsg,
    Signal,
    StepMsg,
    TrainingConfig,
)
from common.specs import validate_specs
from coordinator.db import DB
from coordinator.scheduler import NodeCapacity, plan_split

logger = logging.getLogger("poolforge.hub")

ACTIVE = ("running", "paused", "waiting")


class ApiError(Exception):
    """An error with an HTTP status, raised by hub logic and turned into a response."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


@dataclass
class NodeLive:
    node_id: str
    hostname: str
    os: str
    ws: Any = None  # control WebSocket (None = offline)
    relay: Any = None  # data-plane WebSocket
    stats: Dict[str, Any] = field(default_factory=dict)
    last_seen: float = 0.0

    def public(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "hostname": self.stats.get("hostname", self.hostname),
            "os": self.stats.get("os", self.os),
            "online": self.ws is not None,
            "relay_connected": self.relay is not None,
            "last_seen": self.last_seen,
            **{k: v for k, v in self.stats.items() if k not in ("type", "hostname", "os")},
        }


def node_capacity(n: NodeLive) -> NodeCapacity:
    """What the planner may use on this node: its GPU's free VRAM, or else its free RAM."""
    if n.stats.get("device") == "cuda" and n.stats.get("vram_free_gb"):
        return NodeCapacity(n.node_id, float(n.stats["vram_free_gb"]), "cuda")
    device = "mps" if n.stats.get("device") == "mps" else "cpu"  # Apple GPUs share RAM
    return NodeCapacity(n.node_id, float(n.stats["ram_free_gb"]), device)


@dataclass
class JobLive:
    job_id: str
    config: TrainingConfig
    model: Dict[str, Any]
    dataset: str
    plan: List[Dict[str, Any]]
    transport: str
    status: str = "running"
    attempt: int = 0
    last_step: int = 0
    paused_by_user: bool = False
    saved_steps: Dict[int, Set[int]] = field(default_factory=dict)  # stage -> checkpoint steps
    done_stages: Set[int] = field(default_factory=set)
    missing: Set[str] = field(default_factory=set)
    drop_timer: Optional[asyncio.Task] = None
    data_detail: str = ""

    @property
    def total_steps(self) -> int:
        return self.config.epochs * self.config.steps_per_epoch

    def node_ids(self) -> List[str]:
        return [s["node_id"] for s in self.plan]

    def public(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "config": self.config.model_dump(),
            "model": self.model["name"],
            "dataset": self.data_detail or self.dataset,
            "plan": self.plan,
            "num_layers": len(self.model["specs"]),
            "layer_types": [s["type"] for s in self.model["specs"]],
            "transport": self.transport,
            "last_step": self.last_step,
            "total_steps": self.total_steps,
            "missing_nodes": sorted(self.missing),
        }


@dataclass
class Room:
    code: str
    nodes: Dict[str, NodeLive] = field(default_factory=dict)
    dashboards: Set[Any] = field(default_factory=set)
    job: Optional[JobLive] = None
    # Frames for a node whose relay is (briefly) disconnected: node_id -> deque[(time, bytes)]
    pending: Dict[str, Deque[Tuple[float, bytes]]] = field(default_factory=dict)
    # Chat + voice call (in memory only; gone when the coordinator restarts)
    peers: Dict[str, Any] = field(default_factory=dict)  # peer_id -> dashboard WebSocket
    chat: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=100))
    call: Dict[str, str] = field(default_factory=dict)  # peer_id -> name of everyone in the call


class RateLimiter:
    """Sliding-window limiter: at most `limit` hits per `window` seconds per key."""

    def __init__(self, limit: int, window: float):
        self.limit, self.window = limit, window
        self.hits: Dict[str, Deque[float]] = {}

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        q = self.hits.setdefault(key, deque())
        while q and now - q[0] > self.window:
            q.popleft()
        if len(q) >= self.limit:
            return False
        q.append(now)
        return True


class Hub:
    PENDING_MAX_FRAMES = 64
    PENDING_MAX_AGE_S = 60.0

    def __init__(self, db: DB, drop_grace_s: float = 30.0, heartbeat_s: float = 2.0):
        self.db = db
        self.drop_grace_s = drop_grace_s
        self.heartbeat_s = heartbeat_s
        self.rooms: Dict[str, Room] = {}

    # ------------------------------------------------------------- helpers
    def room(self, code: str) -> Room:
        """Return the live Room, loading it from the DB; 404 if it doesn't exist."""
        if code not in self.rooms:
            if not self.db.get_room(code):
                raise ApiError(404, f"Room '{code}' not found")
            room = Room(code)
            for n in self.db.list_nodes(code):
                room.nodes[n["node_id"]] = NodeLive(n["node_id"], n["hostname"], n["os"])
            self.rooms[code] = room
        return self.rooms[code]

    async def send(self, node: NodeLive, msg: Any) -> bool:
        """Send a pydantic model or dict to an agent's control socket."""
        if node.ws is None:
            return False
        text = msg.model_dump_json() if hasattr(msg, "model_dump_json") else json.dumps(msg)
        try:
            await node.ws.send_text(text)
            return True
        except Exception:
            return False

    async def broadcast(self, room: Room, event: Dict[str, Any]) -> None:
        """Push an event to every open dashboard of this room."""
        text = json.dumps(event)
        for ws in list(room.dashboards):
            try:
                await ws.send_text(text)
            except Exception:
                room.dashboards.discard(ws)

    def snapshot(self, room: Room) -> Dict[str, Any]:
        """Everything a freshly opened dashboard needs."""
        job = room.job
        data: Dict[str, Any] = {
            "type": "snapshot",
            "room": room.code,
            "locked": bool(self.db.get_room(room.code)["locked"]),
            "nodes": [n.public() for n in room.nodes.values()],
            "job": job.public() if job else None,
            "metrics": self.db.job_metrics(job.job_id) if job else [],
            "node_metrics": self.db.job_node_metrics(job.job_id)[-2000:] if job else [],
            "deployments": self.db.list_deployments(room.code),
            "chat": list(room.chat),
            "call": self.call_list(room),
        }
        return data

    # ------------------------------------------------------ chat + call
    @staticmethod
    def call_list(room: Room) -> List[Dict[str, str]]:
        return [{"peer_id": pid, "name": name} for pid, name in room.call.items()]

    async def on_browser(self, room: Room, peer_id: str, msg: Any) -> None:
        """Chat and voice-call messages from one dashboard (already validated)."""
        if isinstance(msg, ChatIn):
            entry = {"name": msg.name, "text": msg.text, "t": time.time(), "peer_id": peer_id}
            room.chat.append(entry)
            await self.broadcast(room, {"type": "chat", **entry})
        elif isinstance(msg, CallJoin):
            room.call[peer_id] = msg.name
            await self.broadcast(room, {"type": "call", "call": self.call_list(room)})
        elif isinstance(msg, CallLeave):
            await self.leave_call(room, peer_id)
        elif isinstance(msg, Signal):
            # Only forward between two peers that are both in THIS room's call.
            target = room.peers.get(msg.to)
            if target is None or peer_id not in room.call or msg.to not in room.call:
                return
            try:
                await target.send_text(json.dumps({"type": "signal", "from": peer_id, "data": msg.data}))
            except Exception:
                pass

    async def leave_call(self, room: Room, peer_id: str) -> None:
        if room.call.pop(peer_id, None) is not None:
            await self.broadcast(room, {"type": "call", "call": self.call_list(room)})

    # ------------------------------------------------------- node presence
    async def node_online(self, room: Room, node: NodeLive) -> None:
        await self.broadcast(room, {"type": "node", "node": node.public()})
        job = room.job
        if job and job.status == "waiting" and node.node_id in job.missing:
            job.missing.discard(node.node_id)
            if not job.missing:
                await self._resume_after_drop(room, job)

    async def node_offline(self, room: Room, node: NodeLive) -> None:
        await self.broadcast(room, {"type": "node", "node": node.public()})
        job = room.job
        if not job or job.status not in ACTIVE or node.node_id not in job.node_ids():
            return
        job.missing.add(node.node_id)
        if job.status == "waiting":
            return
        job.status = "waiting"
        self.db.set_job_status(job.job_id, "waiting", f"node {node.node_id} dropped")
        logger.warning("Node %s dropped during %s; pausing for %.0fs", node.node_id, job.job_id, self.drop_grace_s)
        await self._control_all(room, job, "pause", f"node {node.node_id} dropped")
        await self.broadcast(room, {"type": "job", "job": job.public(),
                                    "message": f"Node {node.node_id} dropped. Waiting {self.drop_grace_s:.0f}s for it to return..."})
        job.drop_timer = asyncio.create_task(self._drop_timeout(room, job))

    async def _drop_timeout(self, room: Room, job: JobLive) -> None:
        await asyncio.sleep(self.drop_grace_s)
        if room.job is job and job.status == "waiting":
            await self._finish(room, job, "failed", f"node(s) {', '.join(sorted(job.missing))} did not come back in time")

    async def _resume_after_drop(self, room: Room, job: JobLive) -> None:
        if job.drop_timer:
            job.drop_timer.cancel()
        # Newest checkpoint step that EVERY stage has saved (0 = start over).
        common = set.intersection(*(job.saved_steps.get(i, set()) for i in range(len(job.plan)))) if job.plan else set()
        resume = max(common | {0})
        job.attempt += 1
        job.done_stages.clear()
        job.status = "paused" if job.paused_by_user else "running"
        self.db.set_job_status(job.job_id, job.status, f"resumed from step {resume}")
        logger.info("All nodes back for %s; resuming from step %d", job.job_id, resume)
        await self._assign_all(room, job, resume)
        if job.paused_by_user:
            await self._control_all(room, job, "pause", "paused by admin")
        await self.broadcast(room, {"type": "job", "job": job.public(), "message": f"Nodes are back. Resuming from checkpoint step {resume}."})

    async def sweep(self) -> None:
        """Close control sockets that stopped sending heartbeats (hung laptops, dead Wi-Fi)."""
        now = time.time()
        for room in list(self.rooms.values()):
            for node in list(room.nodes.values()):
                if node.ws is not None and now - node.last_seen > 5 * self.heartbeat_s:
                    logger.warning("Node %s missed heartbeats; disconnecting it", node.node_id)
                    ws, node.ws = node.ws, None
                    try:
                        await ws.close()
                    except Exception:
                        pass
                    await self.node_offline(room, node)

    # ------------------------------------------------------------ jobs
    def make_plan(self, room: Room, cfg: TrainingConfig) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]], str]:
        """Validate the config and compute (model, dataset, plan, transport) without starting anything."""
        try:
            model = get_model(cfg.model, **cfg.model_options)
            validate_specs(model["specs"], tuple(model["input_shape"]))
        except (KeyError, ValueError, TypeError) as exc:
            raise ApiError(400, str(exc))
        if cfg.batch_size < cfg.microbatches:
            raise ApiError(400, "batch_size must be at least the number of micro-batches")
        dataset = cfg.dataset or model["dataset"]
        if dataset == "mnist" and model["input_shape"] != [1, 28, 28]:
            raise ApiError(400, f"Model '{cfg.model}' does not take MNIST images")
        nodes = [n for n in room.nodes.values() if n.ws is not None and n.stats]
        caps = [node_capacity(n) for n in nodes]
        try:
            plan = [p.to_dict() for p in plan_split(model["specs"], model["input_shape"], caps, cfg.batch_size)]
        except ValueError as exc:
            raise ApiError(400, str(exc))
        # Use direct TCP only if every stage's agent was started with --transport tcp.
        tcp_ok = all(room.nodes[p["node_id"]].stats.get("transport") == "tcp" and room.nodes[p["node_id"]].stats.get("tcp_port")
                     for p in plan)
        return model, dataset, plan, ("tcp" if tcp_ok else "relay")

    async def start_job(self, room: Room, cfg: TrainingConfig) -> JobLive:
        if room.job and room.job.status in ACTIVE:
            raise ApiError(409, f"Job {room.job.job_id} is still {room.job.status}; stop it first")
        model, dataset, plan, transport = self.make_plan(room, cfg)
        job = JobLive(f"job-{secrets.token_hex(4)}", cfg, model, dataset, plan, transport)
        self.db.create_job(job.job_id, room.code, cfg.model, cfg.model_dump(), plan, transport)
        room.job = job
        await self._assign_all(room, job, resume_step=0)
        await self.broadcast(room, {"type": "job", "job": job.public(), "message": f"Started {job.job_id} on {len(plan)} node(s)"})
        logger.info("Started %s (%s) on %s via %s", job.job_id, cfg.model, [p["node_id"] for p in plan], transport)
        return job

    async def _assign_all(self, room: Room, job: JobLive, resume_step: int) -> None:
        n = len(job.plan)
        peers = {
            p["node_id"]: [room.nodes[p["node_id"]].stats.get("tcp_host"), room.nodes[p["node_id"]].stats.get("tcp_port")]
            for p in job.plan
        } if job.transport == "tcp" else {}
        cfg = job.config
        hp = HyperParams(
            batch_size=cfg.batch_size, microbatches=cfg.microbatches, epochs=cfg.epochs,
            steps_per_epoch=cfg.steps_per_epoch, lr=cfg.lr, optimizer=cfg.optimizer,
            checkpoint_every=cfg.checkpoint_every, seed=cfg.seed, dataset=job.dataset,
        )
        for i, p in enumerate(job.plan):
            msg = AssignMsg(
                type="assign", job_id=job.job_id, stage_index=i, num_stages=n,
                start_layer=p["start_layer"], end_layer=p["end_layer"], specs=job.model["specs"],
                input_shape=job.model["input_shape"], num_classes=job.model["num_classes"],
                prev_node=job.plan[i - 1]["node_id"] if i > 0 else None,
                next_node=job.plan[i + 1]["node_id"] if i < n - 1 else None,
                transport=job.transport, peers=peers, hp=hp, resume_step=resume_step, attempt=job.attempt,
                device=p.get("device", "cpu"),
            )
            await self.send(room.nodes[p["node_id"]], msg)

    async def _control_all(self, room: Room, job: JobLive, action: str, reason: str = "") -> None:
        for nid in job.node_ids():
            node = room.nodes.get(nid)
            if node:
                await self.send(node, ControlMsg(type="control", action=action, job_id=job.job_id, reason=reason))

    def active_job(self, room: Room, job_id: str) -> JobLive:
        job = room.job
        if not job or job.job_id != job_id:
            raise ApiError(404, f"Job '{job_id}' is not the room's current job")
        return job

    async def control(self, room: Room, job_id: str, action: str) -> JobLive:
        job = self.active_job(room, job_id)
        if job.status not in ACTIVE:
            raise ApiError(409, f"Job is already {job.status}")
        if action == "stop":
            await self._finish(room, job, "stopped", "stopped by admin")
            return job
        job.paused_by_user = action == "pause"
        if job.status != "waiting":  # while waiting for a node, just remember the wish
            job.status = "paused" if action == "pause" else "running"
            self.db.set_job_status(job.job_id, job.status)
            await self._control_all(room, job, action, f"{action}d by admin")
        await self.broadcast(room, {"type": "job", "job": job.public(), "message": f"Job {action}d"})
        return job

    async def _finish(self, room: Room, job: JobLive, status: str, detail: str) -> None:
        if job.drop_timer:
            job.drop_timer.cancel()
        job.status = status
        self.db.set_job_status(job.job_id, status, detail)
        if status != "completed":
            await self._control_all(room, job, "stop", detail)
        logger.info("Job %s %s: %s", job.job_id, status, detail)
        await self.broadcast(room, {"type": "job", "job": job.public(), "message": f"Job {status}: {detail}"})

    # ------------------------------------------------- reports from agents
    async def on_step(self, room: Room, node: NodeLive, m: StepMsg) -> None:
        job = room.job
        if not job or job.job_id != m.job_id:
            return
        self.db.add_node_metric(job.job_id, node.node_id, m)
        event: Dict[str, Any] = {**m.model_dump(exclude={"type"}), "type": "node_step", "node_id": node.node_id}
        await self.broadcast(room, event)
        if m.loss is not None:  # the last stage reports the loss for the whole pipeline
            job.last_step = max(job.last_step, m.step)
            sps = job.config.batch_size / (m.step_time_ms / 1000.0) if m.step_time_ms > 0 else 0.0
            self.db.add_metric(job.job_id, m.step, m.epoch, m.loss, m.step_time_ms, sps)
            await self.broadcast(room, {
                "type": "step", "job_id": job.job_id, "step": m.step, "epoch": m.epoch, "loss": m.loss,
                "step_time_ms": m.step_time_ms, "samples_per_s": sps, "total_steps": job.total_steps,
            })

    async def on_checkpoint(self, room: Room, job_id: str, stage: int, step: int) -> None:
        job = room.job
        if job and job.job_id == job_id:
            steps = job.saved_steps.setdefault(stage, set())
            steps.add(step)
            # agents keep their newest 3 checkpoints; mirror that here
            for old in sorted(steps)[:-3]:
                steps.discard(old)

    async def on_job_state(self, room: Room, m: JobStateMsg) -> None:
        job = room.job
        if not job or job.job_id != m.job_id or job.status not in ACTIVE:
            return
        if m.state == "started" and m.stage == 0 and m.detail:
            job.data_detail = m.detail
        elif m.state == "error":
            await self._finish(room, job, "failed", f"stage {m.stage}: {m.detail}")
        elif m.state == "done":
            job.done_stages.add(m.stage)
            if len(job.done_stages) == len(job.plan):
                await self._finish(room, job, "completed", f"{job.last_step} steps")

    # ------------------------------------------------------------ relay
    def queue_pending(self, room: Room, target: str, frame: bytes) -> None:
        q = room.pending.setdefault(target, deque(maxlen=self.PENDING_MAX_FRAMES))
        q.append((time.monotonic(), frame))

    async def flush_pending(self, room: Room, node: NodeLive) -> None:
        q = room.pending.pop(node.node_id, None)
        now = time.monotonic()
        while q and node.relay is not None:
            t, frame = q.popleft()
            if now - t <= self.PENDING_MAX_AGE_S:
                await node.relay.send_bytes(frame)
