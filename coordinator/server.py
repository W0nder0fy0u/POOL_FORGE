"""PoolForge coordinator: FastAPI REST + WebSockets + tensor relay + dashboard.

One process (one Docker container) serves everything:

  REST        /api/...                        rooms, join, jobs, deployments, exports
  WebSocket   /ws/agent/{room}/{node}?token   control plane (small JSON messages)
  WebSocket   /ws/relay/{room}/{node}?token   data plane (binary tensor frames)
  WebSocket   /ws/dashboard/{room}            live events for the browser
  Static      /                               the built React dashboard

Run:  python -m coordinator.server     (or: uvicorn coordinator.server:app)
"""

import asyncio
import csv
import hmac
import io
import json
import logging
import os
import re
import secrets
import string
import struct
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Header, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import ValidationError

from common import __version__
from common.apps import catalog as app_catalog
from common.apps import get_app
from common.model_zoo import catalog as model_catalog
from common.schemas import (
    ROOM_CODE_PATTERN,
    AppStateMsg,
    CheckpointMsg,
    DeployRequest,
    JobStateMsg,
    JoinRequest,
    JoinResponse,
    LockRequest,
    RunAppMsg,
    StatsMsg,
    StepMsg,
    StopAppMsg,
    TrainingConfig,
    agent_message_adapter,
)
from coordinator.db import DB, hash_token
from coordinator.hub import ApiError, Hub, NodeLive, RateLimiter, Room

logger = logging.getLogger("poolforge.coordinator")

DIST_DIR = Path(__file__).resolve().parent.parent / "dashboard" / "dist"
MAX_RELAY_FRAME = 600 * 1024 * 1024
UINT32 = struct.Struct(">I")  # same 4-byte length prefix as common/protocol.py (kept here so no torch import)
ROOM_RE = re.compile(ROOM_CODE_PATTERN)


def create_app(db_path: Optional[str] = None, drop_grace_s: Optional[float] = None) -> FastAPI:
    """Build the app. Tests call this to get a fresh coordinator with its own DB."""
    db = DB(db_path or os.environ.get("POOLFORGE_DB_PATH", "data/poolforge.db"))
    hub = Hub(db, drop_grace_s=drop_grace_s if drop_grace_s is not None else float(os.environ.get("POOLFORGE_DROP_GRACE_S", 30)))
    create_limiter = RateLimiter(limit=10, window=60)  # rooms created per IP per minute
    join_limiter = RateLimiter(limit=20, window=60)  # join attempts per IP per minute

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        async def sweeper():
            while True:
                await asyncio.sleep(hub.heartbeat_s)
                try:
                    await hub.sweep()
                except Exception:
                    logger.exception("sweeper error")

        task = asyncio.create_task(sweeper())
        yield
        task.cancel()

    app = FastAPI(title="PoolForge Coordinator", version=__version__, lifespan=lifespan)
    app.state.hub = hub
    app.state.db = db

    @app.exception_handler(ApiError)
    async def api_error(_req: Request, exc: ApiError):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status)

    # ------------------------------------------------------------ helpers
    def client_ip(request: Request) -> str:
        # Behind Render/Railway the real client is the first X-Forwarded-For entry.
        fwd = request.headers.get("x-forwarded-for")
        return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?")

    def public_url(request: Request) -> str:
        env = os.environ.get("POOLFORGE_PUBLIC_URL", "").rstrip("/")
        if env:
            return env
        proto = request.headers.get("x-forwarded-proto", request.url.scheme)
        return f"{proto}://{request.headers.get('host', 'localhost:8000')}"

    def get_room(code: str) -> Room:
        if not ROOM_RE.match(code):
            raise ApiError(404, "Invalid room code")
        return hub.room(code)

    def require_admin(code: str, x_admin_token: str = Header(default="")) -> Room:
        """FastAPI dependency: the request must carry the room's admin token."""
        room = get_room(code)
        stored = db.get_room(code)["admin_hash"]
        if not hmac.compare_digest(stored, hash_token(x_admin_token)):
            raise ApiError(403, "Admin token missing or wrong")
        return room

    def node_auth(code: str, node_id: str, token: str) -> Optional[NodeLive]:
        """Check a node's token for WebSocket connections."""
        if not ROOM_RE.match(code):
            return None
        row = db.get_node(code, node_id)
        if not row or not hmac.compare_digest(row["token_hash"], hash_token(token)):
            return None
        room = hub.room(code)
        if node_id not in room.nodes:
            room.nodes[node_id] = NodeLive(node_id, row["hostname"], row["os"])
        return room.nodes[node_id]

    # --------------------------------------------------------------- REST
    @app.get("/health")
    async def health():
        return {"status": "ok", "version": __version__}

    @app.get("/api/models")
    async def models():
        return model_catalog()

    @app.get("/api/apps")
    async def apps():
        return app_catalog()

    @app.post("/api/rooms")
    async def create_room(request: Request):
        if not create_limiter.allow(client_ip(request)):
            raise ApiError(429, "Too many rooms created from your IP; wait a minute")
        alphabet = string.ascii_uppercase + string.digits
        while True:
            code = "".join(secrets.choice(alphabet) for _ in range(6))
            if not db.get_room(code):
                break
        admin_token = secrets.token_urlsafe(24)
        db.create_room(code, admin_token)
        base = public_url(request)
        logger.info("Created room %s", code)
        return {
            "room_code": code,
            "admin_token": admin_token,
            "join_url": f"{base}/join/{code}",
            "join_command": f"poolforge join {code} --server {base}",
        }

    @app.get("/api/rooms/{code}")
    async def room_info(code: str, request: Request):
        room = get_room(code)
        snap = hub.snapshot(room)
        snap["join_command"] = f"poolforge join {code} --server {public_url(request)}"
        snap["server"] = public_url(request)
        return snap

    @app.post("/api/rooms/{code}/join", response_model=JoinResponse)
    async def join(code: str, body: JoinRequest, request: Request):
        """An agent asks to join. It gets a node id + a secret node token."""
        if not join_limiter.allow(client_ip(request)):
            raise ApiError(429, "Too many join attempts; wait a minute")
        room = get_room(code)
        if db.get_room(code)["locked"]:
            raise ApiError(403, "This pool is locked by its admin; no new nodes can join")
        node_id = body.node_id or f"node-{secrets.token_hex(3)}"
        existing = room.nodes.get(node_id)
        if existing is not None and existing.ws is not None:
            raise ApiError(409, f"Node id '{node_id}' is already online in this room")
        token = secrets.token_urlsafe(24)
        db.add_node(code, node_id, token, body.hostname, body.os)
        room.nodes[node_id] = NodeLive(node_id, body.hostname, body.os)
        # Per-room TCP secret, derived so we don't have to store it.
        tcp_secret = hash_token(db.get_room(code)["admin_hash"] + ":tcp")[:32]
        logger.info("Node %s (%s) joined room %s", node_id, body.hostname, code)
        return JoinResponse(node_id=node_id, node_token=token, tcp_secret=tcp_secret, heartbeat_s=hub.heartbeat_s)

    @app.post("/api/rooms/{code}/admin/check")
    async def admin_check(room: Room = Depends(require_admin)):
        return {"ok": True, "room": room.code}

    @app.post("/api/rooms/{code}/lock")
    async def lock(body: LockRequest, room: Room = Depends(require_admin)):
        db.set_locked(room.code, body.locked)
        await hub.broadcast(room, {"type": "locked", "locked": body.locked})
        return {"locked": body.locked}

    @app.post("/api/rooms/{code}/plan")
    async def preview_plan(cfg: TrainingConfig, room: Room = Depends(require_admin)):
        """Show how the model would be split, without starting anything."""
        model, dataset, plan, transport = hub.make_plan(room, cfg)
        return {"plan": plan, "transport": transport, "dataset": dataset,
                "num_layers": len(model["specs"]), "layer_types": [s["type"] for s in model["specs"]]}

    @app.post("/api/rooms/{code}/jobs")
    async def start_job(cfg: TrainingConfig, room: Room = Depends(require_admin)):
        job = await hub.start_job(room, cfg)
        return job.public()

    @app.post("/api/rooms/{code}/jobs/{job_id}/{action}")
    async def job_action(job_id: str, action: str, room: Room = Depends(require_admin)):
        if action not in ("pause", "resume", "stop"):
            raise ApiError(404, "Unknown action")
        job = await hub.control(room, job_id, action)
        return job.public()

    @app.get("/api/rooms/{code}/jobs")
    async def list_jobs(code: str):
        get_room(code)
        return db.list_jobs(code)

    @app.get("/api/rooms/{code}/jobs/{job_id}")
    async def job_detail(code: str, job_id: str):
        get_room(code)
        job = db.get_job(job_id)
        if not job or job["room"] != code:
            raise ApiError(404, "Job not found")
        job["metrics"] = db.job_metrics(job_id)
        job["node_metrics"] = db.job_node_metrics(job_id)
        return job

    @app.get("/api/rooms/{code}/jobs/{job_id}/export")
    async def export(code: str, job_id: str, format: str = "csv"):
        """Reports page: download a run's per-step results."""
        job = await job_detail(code, job_id)
        if format == "json":
            return JSONResponse(job, headers={"Content-Disposition": f'attachment; filename="{job_id}.json"'})
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["step", "epoch", "loss", "step_time_ms", "samples_per_s", "node_id", "stage",
                         "bytes_sent", "bytes_received", "avg_send_ms", "proc_ram_mb"])
        by_step = {}
        for nm in job["node_metrics"]:
            by_step.setdefault(nm["step"], []).append(nm)
        for m in job["metrics"]:
            for nm in by_step.get(m["step"], [{}]):
                writer.writerow([m["step"], m["epoch"], m["loss"], round(m["step_time_ms"], 2), round(m["samples_per_s"], 2),
                                 nm.get("node_id"), nm.get("stage"), nm.get("bytes_sent"), nm.get("bytes_received"),
                                 nm.get("avg_send_ms"), nm.get("proc_ram_mb")])
        return PlainTextResponse(buf.getvalue(), media_type="text/csv",
                                 headers={"Content-Disposition": f'attachment; filename="{job_id}.csv"'})

    # ---------------------------------------------------------- hosting
    @app.get("/api/rooms/{code}/deployments")
    async def list_deployments(code: str):
        get_room(code)
        return db.list_deployments(code)

    @app.post("/api/rooms/{code}/deployments")
    async def deploy(body: DeployRequest, room: Room = Depends(require_admin)):
        try:
            get_app(body.app_id)  # only allow-listed apps
        except PermissionError as exc:
            raise ApiError(403, str(exc))
        online = [n for n in room.nodes.values() if n.ws is not None and n.stats]
        if body.node_id:
            node = room.nodes.get(body.node_id)
            if not node or node.ws is None:
                raise ApiError(404, f"Node '{body.node_id}' is not online")
        elif online:
            node = max(online, key=lambda n: n.stats.get("ram_free_gb", 0))  # most free RAM
        else:
            raise ApiError(400, "No online nodes to deploy on")
        dep_id = f"app-{secrets.token_hex(3)}"
        db.add_deployment(dep_id, room.code, node.node_id, body.app_id)
        await hub.send(node, RunAppMsg(type="run_app", deployment_id=dep_id, app_id=body.app_id))
        await hub.broadcast(room, {"type": "deployment", "deployment": db.get_deployment(dep_id)})
        return db.get_deployment(dep_id)

    @app.post("/api/rooms/{code}/deployments/{dep_id}/{action}")
    async def deployment_action(dep_id: str, action: str, room: Room = Depends(require_admin)):
        dep = db.get_deployment(dep_id)
        if not dep or dep["room"] != room.code:
            raise ApiError(404, "Deployment not found")
        node = room.nodes.get(dep["node_id"])
        if node is None or node.ws is None:
            raise ApiError(409, f"Node {dep['node_id']} is offline")
        if action == "stop":
            await hub.send(node, StopAppMsg(type="stop_app", deployment_id=dep_id))
        elif action == "restart":
            await hub.send(node, StopAppMsg(type="stop_app", deployment_id=dep_id))
            db.update_deployment(dep_id, "starting", None, "restarting")
            await hub.send(node, RunAppMsg(type="run_app", deployment_id=dep_id, app_id=dep["app_id"]))
        else:
            raise ApiError(404, "Unknown action")
        return db.get_deployment(dep_id)

    # -------------------------------------------------------- WebSockets
    @app.websocket("/ws/agent/{code}/{node_id}")
    async def agent_ws(ws: WebSocket, code: str, node_id: str, token: str = ""):
        node = node_auth(code, node_id, token)
        if node is None:
            await ws.close(code=4403, reason="bad room, node or token")
            return
        await ws.accept()
        room = hub.room(code)
        if node.ws is not None:  # a stale connection from the same node: replace it
            try:
                await node.ws.close()
            except Exception:
                pass
        node.ws = ws
        node.last_seen = time.time()
        logger.info("Agent %s connected (room %s)", node_id, code)
        first_stats = True
        try:
            while True:
                text = await ws.receive_text()
                if len(text) > 64 * 1024:
                    continue
                try:
                    msg = agent_message_adapter.validate_json(text)
                except ValidationError as exc:
                    logger.warning("Invalid message from %s: %s", node_id, exc.errors()[:1])
                    continue
                node.last_seen = time.time()
                if isinstance(msg, StatsMsg):
                    node.stats = msg.model_dump()
                    await hub.broadcast(room, {"type": "node", "node": node.public()})
                    if first_stats:  # node is fully "online" once we know its RAM
                        first_stats = False
                        await hub.node_online(room, node)
                elif isinstance(msg, StepMsg):
                    await hub.on_step(room, node, msg)
                elif isinstance(msg, CheckpointMsg):
                    await hub.on_checkpoint(room, msg.job_id, msg.stage, msg.step)
                elif isinstance(msg, JobStateMsg):
                    await hub.on_job_state(room, msg)
                elif isinstance(msg, AppStateMsg):
                    dep = db.get_deployment(msg.deployment_id)
                    if dep and dep["node_id"] == node_id:
                        db.update_deployment(msg.deployment_id, msg.status, msg.url, msg.detail)
                        await hub.broadcast(room, {"type": "deployment", "deployment": db.get_deployment(msg.deployment_id)})
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            if node.ws is ws:  # (not if we were replaced by a newer connection)
                node.ws = None
                logger.info("Agent %s disconnected", node_id)
                await hub.node_offline(room, node)

    @app.websocket("/ws/relay/{code}/{node_id}")
    async def relay_ws(ws: WebSocket, code: str, node_id: str, token: str = ""):
        """Forward binary tensor frames between members of the SAME room only."""
        node = node_auth(code, node_id, token)
        if node is None:
            await ws.close(code=4403, reason="bad room, node or token")
            return
        await ws.accept()
        room = hub.room(code)
        node.relay = ws
        await hub.flush_pending(room, node)
        sender_prefix = UINT32.pack(len(node_id.encode())) + node_id.encode()
        try:
            while True:
                data = await ws.receive_bytes()
                if len(data) < UINT32.size or len(data) > MAX_RELAY_FRAME:
                    continue
                (n,) = UINT32.unpack_from(data, 0)
                if n > 64:
                    continue
                target_id = data[UINT32.size:UINT32.size + n].decode("utf-8", "replace")
                frame = sender_prefix + data[UINT32.size + n:]
                target = room.nodes.get(target_id)  # lookup is per room: no cross-room relaying
                if target is None:
                    continue
                if target.relay is not None:
                    try:
                        await target.relay.send_bytes(frame)
                        continue
                    except Exception:
                        pass
                hub.queue_pending(room, target_id, frame)  # deliver when it reconnects
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            if node.relay is ws:
                node.relay = None

    @app.websocket("/ws/dashboard/{code}")
    async def dashboard_ws(ws: WebSocket, code: str):
        """Read-only live feed for browsers. Commands go through REST with the admin token."""
        if not ROOM_RE.match(code) or not db.get_room(code):
            await ws.close(code=4404, reason="room not found")
            return
        await ws.accept()
        room = hub.room(code)
        room.dashboards.add(ws)
        try:
            await ws.send_text(json.dumps(hub.snapshot(room)))
            while True:
                await ws.receive_text()  # browser pings; content ignored
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            room.dashboards.discard(ws)

    # ---------------------------------------------------- dashboard files
    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str):
        index = DIST_DIR / "index.html"
        if path.startswith(("api/", "ws/")):
            raise ApiError(404, "Not found")
        if path:
            target = (DIST_DIR / path).resolve()
            # Only serve files that really live inside dashboard/dist (no ../ tricks).
            if target.is_file() and DIST_DIR.resolve() in target.parents:
                return FileResponse(target)
        if index.exists():
            return FileResponse(index)
        return PlainTextResponse("Dashboard not built. Run: cd dashboard && npm install && npm run build", status_code=503)

    return app


app = create_app() if os.environ.get("POOLFORGE_NO_DEFAULT_APP") != "1" else None


class RedactTokens(logging.Filter):
    """Hide ?token=... in log lines (uvicorn logs every WebSocket URL)."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "token=" in msg:
            record.msg, record.args = re.sub(r"token=[^\s&\"]+", "token=***", msg), ()
        return True


def main() -> None:
    import uvicorn

    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    port = int(os.environ.get("PORT") or os.environ.get("POOLFORGE_PORT", 8000))
    host = os.environ.get("POOLFORGE_HOST", "0.0.0.0")
    config = uvicorn.Config(app, host=host, port=port, ws="websockets-sansio", ws_max_size=MAX_RELAY_FRAME,
                            log_level="info", access_log=False)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).addFilter(RedactTokens())
    print(f"PoolForge coordinator on http://{host}:{port}  (dashboard: http://localhost:{port})")
    uvicorn.Server(config).run()


if __name__ == "__main__":
    main()
