"""PoolForge Coordinator Server.

FastAPI REST API + WebSocket Control Plane + Binary Tensor Relay.
Serves dashboard static files and orchestrates distributed model training.
"""

import asyncio
import json
import logging
import secrets
import string
import struct
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set
from fastapi import FastAPI, HTTPException, Header, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from common.model_zoo import get_model_specs
from common.schemas import (
    AssignJobMessage,
    CreateRoomResponse,
    NodeTelemetry,
    RoomStatus,
    StartTrainingRequest,
    StepMetric,
)
from coordinator.db import CoordinatorDB
from coordinator.scheduler import schedule_pipeline_split

logger = logging.getLogger("poolforge.coordinator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

app = FastAPI(title="PoolForge Coordinator", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Database
db = CoordinatorDB("poolforge.db")

# Active WebSocket connections
# (room_code, node_id) -> WebSocket
agent_sockets: Dict[tuple[str, str], WebSocket] = {}
# (room_code, node_id) -> WebSocket
relay_sockets: Dict[tuple[str, str], WebSocket] = {}
# room_code -> Set[WebSocket]
dashboard_sockets: Dict[str, Set[WebSocket]] = {}

UINT32_STRUCT = struct.Struct(">I")

# In-memory active job tracker: room_code -> job_id
active_room_jobs: Dict[str, str] = {}
# Active job metadata: job_id -> dict
job_registry: Dict[str, Dict[str, Any]] = {}
# Drop grace timers: (room_code, node_id) -> asyncio.Task
node_drop_tasks: Dict[tuple[str, str], asyncio.Task] = {}
# Deployed edge apps registry: app_id -> metadata dict
deployed_apps_registry: Dict[str, Dict[str, Any]] = {}

# Simple in-memory rate limiter: ip -> list of timestamps
rate_limit_history: Dict[str, list[float]] = {}


def check_rate_limit(client_ip: str, max_requests: int = 20, window_seconds: float = 10.0) -> None:
    """Enforce request rate limiting per IP address."""
    now = time.time()
    history = rate_limit_history.get(client_ip, [])
    # Filter within window
    history = [t for t in history if now - t < window_seconds]
    if len(history) >= max_requests:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: max {max_requests} requests per {window_seconds}s",
        )
    history.append(now)
    rate_limit_history[client_ip] = history


def generate_room_code(length: int = 6) -> str:
    """Generate random uppercase/numeric 6-character room code."""
    chars = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(chars) for _ in range(length))


async def broadcast_to_dashboard(room_code: str, message: Dict[str, Any]) -> None:
    """Broadcast JSON event to all connected dashboard clients for this room."""
    clients = dashboard_sockets.get(room_code, set())
    if not clients:
        return
    text = json.dumps(message)
    dead_clients = set()
    for ws in list(clients):
        try:
            await ws.send_text(text)
        except Exception:
            dead_clients.add(ws)
    for ws in dead_clients:
        clients.discard(ws)


# ---------------------------------------------------------
# REST APIs
# ---------------------------------------------------------

@app.post("/api/rooms", response_model=CreateRoomResponse)
async def create_room(request: Request):
    """Create a new collaborative pool room."""
    room_code = generate_room_code()
    admin_token = secrets.token_hex(16)
    db.create_room(room_code, admin_token)

    host = request.headers.get("host", "localhost:8000")
    scheme = "https" if request.url.scheme == "https" else "http"
    join_url = f"{scheme}://{host}/join/{room_code}"

    logger.info(f"Created pool room {room_code}")
    return CreateRoomResponse(
        room_code=room_code,
        admin_token=admin_token,
        join_url=join_url,
        created_at=time.time(),
    )


@app.get("/api/rooms/{room_code}", response_model=RoomStatus)
async def get_room(room_code: str):
    """Get status and participating nodes for a room."""
    room = db.get_room(room_code)
    if not room:
        raise HTTPException(status_code=404, detail=f"Room '{room_code}' not found")

    nodes = db.get_room_nodes(room_code)
    active_job = active_room_jobs.get(room_code)
    return RoomStatus(
        room_code=room_code,
        node_count=len(nodes),
        nodes=nodes,
        active_job_id=active_job,
        job_status=room.get("status", "idle"),
    )


@app.post("/api/rooms/{room_code}/start")
async def start_training(
    room_code: str,
    req: StartTrainingRequest,
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
    admin_token: Optional[str] = Query(None),
):
    """Start distributed training across all connected pool nodes."""
    room = db.get_room(room_code)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    token = x_admin_token or admin_token
    if token != room["admin_token"]:
        raise HTTPException(status_code=403, detail="Invalid admin token")

    nodes = db.get_room_nodes(room_code)
    if len(nodes) < 1:
        raise HTTPException(status_code=400, detail="Cannot start: 0 worker nodes in pool")

    # Load layer specs
    try:
        specs = get_model_specs(req.model_name)
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unknown model: {req.model_name}")

    # Compute contiguous pipeline partitioning with memory check
    try:
        split_plan = schedule_pipeline_split(specs, nodes, batch_size=req.batch_size)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job_id = f"job_{room_code}_{int(time.time())}"
    active_room_jobs[room_code] = job_id
    job_registry[job_id] = {
        "job_id": job_id,
        "room_code": room_code,
        "split_plan": split_plan,
        "nodes": [s.node_id for s in split_plan],
        "status": "running",
        "last_step": 0,
        "config": req.model_dump(),
    }
    db.create_job(job_id, room_code, req.model_name, req.model_dump())

    # Build peer TCP address dictionary for DirectTCP transports
    peer_tcp_addrs = {}
    for n in nodes:
        if n.tcp_host and n.tcp_port:
            peer_tcp_addrs[n.node_id] = [n.tcp_host, n.tcp_port]

    # Send assign messages to each node's control socket
    num_stages = len(split_plan)
    for idx, stage in enumerate(split_plan):
        prev_id = split_plan[idx - 1].node_id if idx > 0 else None
        next_id = split_plan[idx + 1].node_id if idx < num_stages - 1 else None

        assign_msg = AssignJobMessage(
            job_id=job_id,
            stage_index=idx,
            num_stages=num_stages,
            start_layer=stage.start_layer,
            end_layer=stage.end_layer,
            layer_specs=specs,
            prev_node_id=prev_id,
            next_node_id=next_id,
            peer_tcp_addresses=peer_tcp_addrs,
            hyperparameters=req.model_dump(),
        )

        node_ws = agent_sockets.get((room_code, stage.node_id))
        if node_ws:
            await node_ws.send_text(
                json.dumps({"type": "assign", "payload": assign_msg.model_dump()})
            )

    await broadcast_to_dashboard(
        room_code,
        {
            "type": "job_started",
            "job_id": job_id,
            "split_plan": [s.model_dump() for s in split_plan],
            "config": req.model_dump(),
        },
    )

    return {
        "status": "started",
        "job_id": job_id,
        "split_plan": [s.model_dump() for s in split_plan],
    }


@app.post("/api/rooms/{room_code}/stop")
async def stop_training(
    room_code: str,
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
    admin_token: Optional[str] = Query(None),
):
    """Stop currently executing training job."""
    room = db.get_room(room_code)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    token = x_admin_token or admin_token
    if token != room["admin_token"]:
        raise HTTPException(status_code=403, detail="Invalid admin token")

    job_id = active_room_jobs.pop(room_code, None)
    if job_id:
        db.update_job_status(job_id, "stopped")

    # Broadcast stop to all connected agents
    for (r_code, node_id), ws in list(agent_sockets.items()):
        if r_code == room_code:
            try:
                await ws.send_text(json.dumps({"type": "stop", "job_id": job_id}))
            except Exception:
                pass

    await broadcast_to_dashboard(room_code, {"type": "job_stopped", "job_id": job_id})
    return {"status": "stopped", "job_id": job_id}


@app.post("/api/rooms/{room_code}/pause")
async def pause_training(
    room_code: str,
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
    admin_token: Optional[str] = Query(None),
):
    """Pause currently executing training job."""
    room = db.get_room(room_code)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    token = x_admin_token or admin_token
    if token != room["admin_token"]:
        raise HTTPException(status_code=403, detail="Invalid admin token")

    job_id = active_room_jobs.get(room_code)
    if not job_id:
        raise HTTPException(status_code=400, detail="No active job to pause")

    if job_id in job_registry:
        job_registry[job_id]["status"] = "paused"
    db.update_job_status(job_id, "paused")

    for (r_code, node_id), ws in list(agent_sockets.items()):
        if r_code == room_code:
            try:
                await ws.send_text(json.dumps({"type": "pause", "job_id": job_id}))
            except Exception:
                pass

    await broadcast_to_dashboard(room_code, {"type": "job_paused", "job_id": job_id})
    return {"status": "paused", "job_id": job_id}


@app.post("/api/rooms/{room_code}/resume")
async def resume_training(
    room_code: str,
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
    admin_token: Optional[str] = Query(None),
):
    """Resume paused training job."""
    room = db.get_room(room_code)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    token = x_admin_token or admin_token
    if token != room["admin_token"]:
        raise HTTPException(status_code=403, detail="Invalid admin token")

    job_id = active_room_jobs.get(room_code)
    if not job_id:
        raise HTTPException(status_code=400, detail="No active job to resume")

    if job_id in job_registry:
        job_registry[job_id]["status"] = "running"
    db.update_job_status(job_id, "running")

    for (r_code, node_id), ws in list(agent_sockets.items()):
        if r_code == room_code:
            try:
                await ws.send_text(json.dumps({"type": "resume", "job_id": job_id}))
            except Exception:
                pass

    await broadcast_to_dashboard(room_code, {"type": "job_resumed", "job_id": job_id})
    return {"status": "resumed", "job_id": job_id}


@app.post("/api/rooms/{room_code}/report_metric")
async def report_metric(room_code: str, metric: StepMetric):
    """Receive step telemetry emitted by the last pipeline stage."""
    db.record_metric(
        job_id=metric.job_id,
        step=metric.step,
        epoch=metric.epoch,
        loss=metric.loss,
        step_time_ms=metric.step_time_ms,
        throughput=metric.throughput_samples_sec,
    )
    # Broadcast live step metrics to dashboard
    await broadcast_to_dashboard(
        room_code,
        {"type": "step_metric", "payload": metric.model_dump()},
    )
    return {"status": "ok"}


@app.get("/api/rooms/{room_code}/metrics")
async def get_metrics(room_code: str, job_id: Optional[str] = None):
    """Fetch time-series training metrics for dashboard visualization."""
    target_job = job_id or active_room_jobs.get(room_code)
    if not target_job:
        return {"job_id": None, "metrics": []}
    metrics = db.get_job_metrics(target_job)
    return {"job_id": target_job, "metrics": metrics}


# ---------------------------------------------------------
# Hosting APIs (Edge Microservice Deployment)
# ---------------------------------------------------------

@app.post("/api/rooms/{room_code}/deploy")
async def deploy_app(
    room_code: str,
    payload: Dict[str, Any],
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
    admin_token: Optional[str] = Query(None),
):
    """Deploy an allow-listed edge web application onto a pool node."""
    room = db.get_room(room_code)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    token = x_admin_token or admin_token
    if token != room["admin_token"]:
        raise HTTPException(status_code=403, detail="Invalid admin token")

    app_id = payload.get("app_id", "static_site")
    requested_node_id = payload.get("node_id")
    port = payload.get("port")

    nodes = db.get_room_nodes(room_code)
    if not nodes:
        raise HTTPException(status_code=400, detail="No nodes connected to host application")

    # If no node specified, automatically pick the node with the MOST free RAM
    if requested_node_id:
        target_node = next((n for n in nodes if n.node_id == requested_node_id), None)
        if not target_node:
            raise HTTPException(status_code=404, detail=f"Node '{requested_node_id}' not found in room")
    else:
        target_node = max(nodes, key=lambda n: n.ram_free_gb)

    target_ws = agent_sockets.get((room_code, target_node.node_id))
    if not target_ws:
        raise HTTPException(status_code=500, detail=f"Target node '{target_node.node_id}' control socket not active")

    # Send run_app command to agent
    await target_ws.send_text(
        json.dumps(
            {
                "type": "run_app",
                "app_id": app_id,
                "port": port,
            }
        )
    )

    logger.info(f"Sent deploy command for '{app_id}' to node [{target_node.node_id}]")
    return {
        "status": "deploying",
        "app_id": app_id,
        "node_id": target_node.node_id,
        "selection_reason": "most_free_ram" if not requested_node_id else "explicit_selection",
    }


@app.get("/api/rooms/{room_code}/apps")
async def list_deployed_apps(room_code: str):
    """Return all currently active hosted apps for this room."""
    return {"apps": list(deployed_apps_registry.values())}


@app.post("/api/rooms/{room_code}/apps/{app_id}/stop")
async def stop_deployed_app(
    room_code: str,
    app_id: str,
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
    admin_token: Optional[str] = Query(None),
):
    """Stop a hosted edge application."""
    room = db.get_room(room_code)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    token = x_admin_token or admin_token
    if token != room["admin_token"]:
        raise HTTPException(status_code=403, detail="Invalid admin token")

    app_info = deployed_apps_registry.get(app_id)
    if not app_info:
        raise HTTPException(status_code=404, detail=f"App '{app_id}' not found")

    node_id = app_info["node_id"]
    target_ws = agent_sockets.get((room_code, node_id))
    if target_ws:
        await target_ws.send_text(json.dumps({"type": "stop_app", "app_id": app_id}))

    deployed_apps_registry.pop(app_id, None)
    await broadcast_to_dashboard(room_code, {"type": "app_stopped", "app_id": app_id})
    return {"status": "stopped", "app_id": app_id}


# ---------------------------------------------------------
# WebSocket Endpoints
# ---------------------------------------------------------

@app.websocket("/ws/agent/{room_code}/{node_id}")
async def agent_control_websocket(websocket: WebSocket, room_code: str, node_id: str):
    """Control plane connection between worker agent and coordinator."""
    await websocket.accept()
    agent_sockets[(room_code, node_id)] = websocket
    logger.info(f"Agent [{node_id}] connected to control plane for room {room_code}")

    # Check if this node was missing for an active paused job
    job_id = active_room_jobs.get(room_code)
    drop_key = (room_code, node_id)
    if drop_key in node_drop_tasks:
        drop_task = node_drop_tasks.pop(drop_key)
        drop_task.cancel()
        logger.info(f"Node [{node_id}] returned before timeout! Resuming job {job_id}")
        if job_id and job_id in job_registry:
            job_registry[job_id]["status"] = "running"
            db.update_job_status(job_id, "running")
            # Notify agents to resume
            for (r_code, n_id), ws in list(agent_sockets.items()):
                if r_code == room_code:
                    try:
                        await ws.send_text(json.dumps({"type": "resume", "job_id": job_id}))
                    except Exception:
                        pass
            await broadcast_to_dashboard(
                room_code,
                {"type": "job_resumed", "job_id": job_id, "returned_node": node_id},
            )

    try:
        while True:
            text_data = await websocket.receive_text()
            msg = json.loads(text_data)
            mtype = msg.get("type")

            if mtype == "register" or mtype == "heartbeat":
                telemetry = NodeTelemetry(**msg["telemetry"])
                db.upsert_node(room_code, telemetry)
                await broadcast_to_dashboard(
                    room_code,
                    {"type": "node_update", "telemetry": telemetry.model_dump()},
                )

            elif mtype == "app_started":
                app_info = msg["app_info"]
                deployed_apps_registry[app_info["app_id"]] = app_info
                logger.info(f"App '{app_info['app_id']}' started on [{node_id}]: {app_info['url']}")
                await broadcast_to_dashboard(
                    room_code,
                    {"type": "app_deployed", "app_info": app_info},
                )

            elif mtype == "app_stopped":
                app_id = msg["app_id"]
                deployed_apps_registry.pop(app_id, None)
                await broadcast_to_dashboard(
                    room_code,
                    {"type": "app_stopped", "app_id": app_id},
                )

    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        agent_sockets.pop((room_code, node_id), None)
        db.remove_node(node_id)
        logger.info(f"Agent [{node_id}] disconnected from control plane")
        await broadcast_to_dashboard(
            room_code,
            {"type": "node_left", "node_id": node_id},
        )

        # Robust Node-Drop Handling: If an active job is running, pause it and wait with a grace timer
        current_job_id = active_room_jobs.get(room_code)
        if current_job_id and current_job_id in job_registry:
            job_info = job_registry[current_job_id]
            if node_id in job_info.get("nodes", []):
                logger.warning(
                    f"Participating node [{node_id}] dropped during active job {current_job_id}! Pausing job..."
                )
                job_info["status"] = "paused"
                db.update_job_status(current_job_id, "paused")

                # Instruct remaining agents to pause
                for (r_code, n_id), ws in list(agent_sockets.items()):
                    if r_code == room_code:
                        try:
                            await ws.send_text(
                                json.dumps(
                                    {
                                        "type": "pause",
                                        "job_id": current_job_id,
                                        "reason": f"Node {node_id} dropped",
                                    }
                                )
                            )
                        except Exception:
                            pass

                await broadcast_to_dashboard(
                    room_code,
                    {
                        "type": "job_paused",
                        "job_id": current_job_id,
                        "dropped_node": node_id,
                        "grace_seconds": 10.0,
                    },
                )

                async def _grace_timeout_handler():
                    try:
                        await asyncio.sleep(10.0)
                        if drop_key in node_drop_tasks:
                            logger.error(f"Node [{node_id}] failed to return within 10s grace period. Failing job {current_job_id}")
                            job_info["status"] = "failed"
                            db.update_job_status(current_job_id, "failed")
                            active_room_jobs.pop(room_code, None)
                            node_drop_tasks.pop(drop_key, None)
                            await broadcast_to_dashboard(
                                room_code,
                                {
                                    "type": "job_failed",
                                    "job_id": current_job_id,
                                    "reason": f"Node {node_id} disconnected and timed out",
                                },
                            )
                    except asyncio.CancelledError:
                        pass

                drop_task = asyncio.create_task(_grace_timeout_handler())
                node_drop_tasks[drop_key] = drop_task


@app.websocket("/ws/relay/{room_code}/{node_id}")
async def binary_relay_websocket(websocket: WebSocket, room_code: str, node_id: str):
    """Data plane: routes binary tensor frames between pool nodes."""
    await websocket.accept()
    relay_sockets[(room_code, node_id)] = websocket
    logger.info(f"RelayTransport [{node_id}] connected in room {room_code}")

    try:
        while True:
            message = await websocket.receive_bytes()
            if len(message) < UINT32_STRUCT.size:
                continue

            # Envelope from sender: [4-byte target_len][target_id][payload]
            (target_len,) = UINT32_STRUCT.unpack_from(message, 0)
            target_id = message[UINT32_STRUCT.size : UINT32_STRUCT.size + target_len].decode("utf-8")
            raw_payload = message[UINT32_STRUCT.size + target_len :]

            target_ws = relay_sockets.get((room_code, target_id))
            if target_ws:
                # Forward envelope to recipient: [4-byte sender_len][sender_id][payload]
                sender_bytes = node_id.encode("utf-8")
                forward_frame = (
                    UINT32_STRUCT.pack(len(sender_bytes)) + sender_bytes + raw_payload
                )
                await target_ws.send_bytes(forward_frame)
            else:
                logger.warning(f"Relay dropped frame: target node [{target_id}] not connected")

    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        relay_sockets.pop((room_code, node_id), None)
        logger.info(f"RelayTransport [{node_id}] disconnected")


@app.websocket("/ws/dashboard/{room_code}")
async def dashboard_websocket(websocket: WebSocket, room_code: str):
    """Live streaming WebSocket for the React web dashboard."""
    await websocket.accept()
    if room_code not in dashboard_sockets:
        dashboard_sockets[room_code] = set()
    dashboard_sockets[room_code].add(websocket)

    # Immediately send current state snapshot
    room = db.get_room(room_code)
    nodes = db.get_room_nodes(room_code)
    await websocket.send_text(
        json.dumps(
            {
                "type": "snapshot",
                "room": room,
                "nodes": [n.model_dump() for n in nodes],
                "active_job": active_room_jobs.get(room_code),
            }
        )
    )

    try:
        while True:
            # Keep-alive ping/pong
            await websocket.receive_text()
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        dashboard_sockets[room_code].discard(websocket)


# ---------------------------------------------------------
# Static File Hosting (Serves built React dashboard)
# ---------------------------------------------------------
dist_dir = Path(__file__).resolve().parent.parent / "dashboard" / "dist"
if dist_dir.exists():
    assets_dir = dist_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = dist_dir / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(dist_dir / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("coordinator.server:app", host="0.0.0.0", port=8000, reload=False)
