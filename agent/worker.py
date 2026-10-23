"""The PoolForge agent: what `poolforge join CODE --server URL` runs.

Life of an agent:
  1. POST /api/rooms/CODE/join  -> gets node_id + secret node_token
  2. opens the relay WebSocket (data plane) and, with --transport tcp, a TCP listener
  3. opens the control WebSocket and sends stats every 2 s
  4. obeys messages: assign (train a stage), control (pause/resume/stop),
     run_app / stop_app (hosting)
  If the coordinator connection drops, it reconnects with exponential backoff
  (1, 2, 4, 8, 10, 10 ... seconds). Reports produced while offline are queued
  and sent after reconnecting. A re-assign after a node drop resumes training
  from the last checkpoint.

All connections are OUTBOUND from the laptop: no port forwarding needed.
"""

import asyncio
import json
import logging
import platform
import socket
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional

import psutil
import websockets

from agent.hosting import AppManager, lan_ip
from agent.trainer import proc_ram_mb, run_job
from common.schemas import AssignMsg, ControlMsg, RunAppMsg, StatsMsg, StopAppMsg, coordinator_message_adapter
from common.transports import DirectTCPTransport, RelayTransport, Transport

logger = logging.getLogger("poolforge.agent")
GB = 1024 ** 3


class JoinError(Exception):
    """The coordinator refused us (bad room, locked room, ...): don't retry forever."""


class Agent:
    def __init__(
        self,
        room: str,
        server: str,
        node_id: Optional[str] = None,
        ram_cap_gb: Optional[float] = None,
        transport: str = "relay",
        advertise_host: Optional[str] = None,
        tcp_port: int = 0,
        simulate_latency_ms: float = 0.0,
        encrypt_passphrase: Optional[str] = None,
        workdir: str = ".",
    ):
        self.room = room.strip().upper()
        self.server = server.rstrip("/")
        self.requested_id = node_id
        self.ram_cap_gb = ram_cap_gb
        self.transport_mode = transport
        self.advertise_host = advertise_host
        self.tcp_port = tcp_port
        self.latency = simulate_latency_ms
        self.workdir = Path(workdir)
        self.key = None
        if encrypt_passphrase:
            from common.crypto import derive_room_key

            self.key = derive_room_key(encrypt_passphrase, self.room)

        self.node_id: Optional[str] = None
        self.token: Optional[str] = None
        self.tcp_secret = ""
        self.heartbeat_s = 2.0
        self.relay: Optional[RelayTransport] = None
        self.tcp: Optional[DirectTCPTransport] = None
        self.ws = None
        self.outbox: Deque[str] = deque(maxlen=5000)  # reports waiting for a connection
        self._send_lock = asyncio.Lock()
        self.job_task: Optional[asyncio.Task] = None
        self.job_id: Optional[str] = None
        self.gate = asyncio.Event()  # set = job may run, clear = paused
        self.apps = AppManager(self.report)
        self.stopped = asyncio.Event()

    # ------------------------------------------------------------- joining
    def _http_join(self) -> Dict[str, Any]:
        body = json.dumps({"hostname": socket.gethostname()[:100], "os": f"{platform.system()} {platform.release()}"[:100],
                           "node_id": self.requested_id}).encode()
        req = urllib.request.Request(f"{self.server}/api/rooms/{self.room}/join", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            try:
                detail = json.loads(detail)["detail"]
            except Exception:
                pass
            if exc.code in (400, 403, 404, 409, 422):
                raise JoinError(f"Coordinator refused to join room {self.room}: {detail}")
            raise ConnectionError(f"HTTP {exc.code}: {detail}")

    async def join(self) -> None:
        backoff = 1.0
        while True:
            try:
                data = await asyncio.to_thread(self._http_join)
                break
            except (ConnectionError, OSError) as exc:  # coordinator not reachable yet
                logger.warning("Cannot reach %s (%s); retrying in %.0fs", self.server, exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 10.0)
        self.node_id, self.token = data["node_id"], data["node_token"]
        self.tcp_secret, self.heartbeat_s = data["tcp_secret"], data["heartbeat_s"]
        logger.info("Joined room %s as %s", self.room, self.node_id)

    # ----------------------------------------------------------- reporting
    def stats(self) -> StatsMsg:
        vm = psutil.virtual_memory()
        free = vm.available / GB
        if self.ram_cap_gb is not None:
            free = min(free, self.ram_cap_gb)  # pretend to be a smaller laptop
        if self.job_task and not self.job_task.done():
            status = "training" if self.gate.is_set() else "paused"
        else:
            status = "hosting" if self.apps.apps else "idle"
        return StatsMsg(
            type="stats", hostname=socket.gethostname()[:100], os=f"{platform.system()} {platform.release()}"[:100],
            ram_total_gb=round(vm.total / GB, 2), ram_free_gb=round(free, 3), ram_cap_gb=self.ram_cap_gb,
            proc_ram_mb=round(proc_ram_mb(), 1), cpu_percent=psutil.cpu_percent(None), status=status,
            transport=self.transport_mode, tcp_host=(self.advertise_host or lan_ip()) if self.tcp else None,
            tcp_port=self.tcp.port if self.tcp else None,
            docker=self.apps.has_docker, cloudflared=self.apps.has_cloudflared,
        )

    async def report(self, msg: Any) -> None:
        """Send a message to the coordinator, or queue it while disconnected.

        Stats are not queued (old RAM numbers are useless); training reports are,
        so no step or checkpoint is lost during a short disconnect.
        """
        if isinstance(msg, StatsMsg) and self.ws is None:
            return
        self.outbox.append(msg.model_dump_json())
        await self._flush()

    async def _flush(self) -> None:
        async with self._send_lock:  # one sender at a time keeps messages in order
            while self.outbox and self.ws is not None:
                try:
                    await self.ws.send(self.outbox[0])
                    self.outbox.popleft()
                except Exception:
                    return

    async def _heartbeat(self) -> None:
        while True:
            await self.report(self.stats())
            await asyncio.sleep(self.heartbeat_s)

    # ----------------------------------------------------------- main loop
    async def run(self) -> None:
        await self.join()
        self.relay = RelayTransport(self.node_id, self.server, self.room, self.token, self.latency)
        await self.relay.start()
        if self.transport_mode == "tcp":
            self.tcp = DirectTCPTransport(self.node_id, "0.0.0.0", self.tcp_port, self.tcp_secret, self.latency)
            await self.tcp.start()
        heartbeat = asyncio.create_task(self._heartbeat())
        try:
            await self._control_loop()
        finally:
            heartbeat.cancel()
            await self.shutdown()

    async def _control_loop(self) -> None:
        base = self.server.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
        backoff = 1.0
        while not self.stopped.is_set():
            url = f"{base}/ws/agent/{self.room}/{self.node_id}?token={self.token}"
            try:
                async with websockets.connect(url, max_size=8 * 1024 * 1024, ping_interval=20) as ws:
                    self.ws = ws
                    backoff = 1.0
                    logger.info("Connected to coordinator %s", self.server)
                    await self.report(self.stats())
                    async for raw in ws:
                        await self._handle(raw)
            except websockets.InvalidStatus as exc:
                # 403 = our token is no longer valid (e.g. coordinator database reset): join again.
                logger.warning("Coordinator rejected our connection (%s); re-joining", exc)
                await self.join()
                self.relay.url = self.relay.url.rsplit("token=", 1)[0] + f"token={self.token}"
            except (OSError, websockets.ConnectionClosed, asyncio.TimeoutError) as exc:
                logger.warning("Lost coordinator connection (%s); reconnecting in %.0fs", exc, backoff)
            finally:
                self.ws = None
            if self.stopped.is_set():
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 10.0)

    async def _handle(self, raw: str) -> None:
        try:
            msg = coordinator_message_adapter.validate_json(raw)
        except Exception as exc:
            logger.warning("Ignoring invalid message from coordinator: %s", exc)
            return
        if isinstance(msg, AssignMsg):
            await self._start_job(msg)
        elif isinstance(msg, ControlMsg):
            if msg.job_id != self.job_id:
                return
            if msg.action == "pause":
                self.gate.clear()
                logger.info("Paused (%s)", msg.reason)
            elif msg.action == "resume":
                self.gate.set()
                logger.info("Resumed")
            else:
                logger.info("Stopping job (%s)", msg.reason)
                await self._cancel_job()
        elif isinstance(msg, RunAppMsg):
            asyncio.create_task(self.apps.start(msg.deployment_id, msg.app_id))
        elif isinstance(msg, StopAppMsg):
            asyncio.create_task(self.apps.stop(msg.deployment_id))

    def _transport_for(self, msg: AssignMsg) -> Transport:
        if msg.transport == "tcp" and self.tcp is not None:
            for nid, (host, port) in msg.peers.items():
                if nid != self.node_id and host and port:
                    self.tcp.register_peer(nid, host, int(port))
            return self.tcp
        return self.relay

    async def _cancel_job(self) -> None:
        if self.job_task and not self.job_task.done():
            self.job_task.cancel()
            try:
                await self.job_task
            except (asyncio.CancelledError, Exception):
                pass
        self.job_task = None

    async def _start_job(self, msg: AssignMsg) -> None:
        await self._cancel_job()  # a re-assign replaces the running attempt
        self.job_id = msg.job_id
        self.gate.set()
        transport = self._transport_for(msg)
        self.job_task = asyncio.create_task(
            run_job(msg, transport, self.report, self.gate, self.key,
                    str(self.workdir / "checkpoints"), str(self.workdir / "data"))
        )

    async def shutdown(self) -> None:
        self.stopped.set()
        await self._cancel_job()
        await self.apps.stop_all()
        for t in (self.relay, self.tcp):
            if t:
                await t.stop()
