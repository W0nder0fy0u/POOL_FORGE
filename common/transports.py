"""PoolForge data-plane transports (how tensor frames move between nodes).

Every transport implements the same small interface, so the training code
never knows (or cares) how bytes travel:

    await t.start()
    await t.send(target_node_id, frame_bytes)
    sender_id, frame_bytes = await t.recv()
    await t.stop()

Implementations:
  * RelayTransport     - default. One outbound WebSocket to the coordinator,
                         which forwards frames to the target node. Works
                         behind any home router / NAT (no port forwarding).
  * DirectTCPTransport - node-to-node TCP sockets. Faster, but peers must be
                         reachable (same LAN, hotspot or Tailscale).
  * LocalTransport     - in-process queues, used by tests and benchmarks.

Simulated latency (--simulate-latency-ms) is applied to every INCOMING frame:
a frame only becomes visible to recv() `latency` ms after it arrived. This is
like a slow network link and, unlike sleeping in send(), it does not block the
sender (so micro-batching can still hide the latency, as on a real network).
"""

import abc
import asyncio
import hmac
import logging
import time
from typing import Dict, Optional, Tuple

from common.protocol import UINT32

logger = logging.getLogger("poolforge.transport")

MAX_FRAME_BYTES = 600 * 1024 * 1024  # header + 512 MB tensor + envelope


def _pack_id(node_id: str) -> bytes:
    """Encode a node id as [4-byte length][utf-8 bytes]."""
    raw = node_id.encode("utf-8")
    return UINT32.pack(len(raw)) + raw


def _unpack_id(data: bytes) -> Tuple[str, bytes]:
    """Inverse of _pack_id: returns (node_id, rest_of_data)."""
    if len(data) < UINT32.size:
        raise ValueError("Envelope too short")
    (n,) = UINT32.unpack_from(data, 0)
    if n > 256 or len(data) < UINT32.size + n:
        raise ValueError("Bad node id length in envelope")
    node_id = bytes(data[UINT32.size:UINT32.size + n]).decode("utf-8")
    return node_id, data[UINT32.size + n:]


class Transport(abc.ABC):
    """Common base: an inbox queue, byte/latency counters and simulated latency."""

    name = "base"

    def __init__(self, node_id: str, simulate_latency_ms: float = 0.0):
        self.node_id = node_id
        self.simulate_latency_ms = float(simulate_latency_ms)
        # Inbox items: (time when deliverable, sender_id, payload)
        self._inbox: "asyncio.Queue[Tuple[float, str, bytes]]" = asyncio.Queue()
        # Counters (cumulative) + a "since last snapshot" copy for per-step stats.
        self.bytes_sent = 0
        self.bytes_received = 0
        self.frames_sent = 0
        self.send_time_ms = 0.0  # total time spent inside send()
        self._snap = (0, 0, 0, 0.0)

    # ----- interface -------------------------------------------------------
    @abc.abstractmethod
    async def start(self) -> None: ...

    @abc.abstractmethod
    async def stop(self) -> None: ...

    @abc.abstractmethod
    async def _send_raw(self, target_node_id: str, payload: bytes) -> int:
        """Actually send; return number of bytes put on the wire."""

    async def send(self, target_node_id: str, payload: bytes) -> None:
        t0 = time.perf_counter()
        wire_bytes = await self._send_raw(target_node_id, payload)
        self.send_time_ms += (time.perf_counter() - t0) * 1000.0
        self.bytes_sent += wire_bytes
        self.frames_sent += 1

    async def recv(self, timeout: Optional[float] = None) -> Tuple[str, bytes]:
        """Wait for the next frame. Raises asyncio.TimeoutError on timeout."""
        deliver_at, sender, payload = await asyncio.wait_for(self._inbox.get(), timeout)
        wait = deliver_at - time.monotonic()
        if wait > 0:  # simulated network latency
            await asyncio.sleep(wait)
        return sender, payload

    # ----- helpers for subclasses -----------------------------------------
    def _deliver(self, sender: str, payload: bytes, wire_bytes: int) -> None:
        self.bytes_received += wire_bytes
        self._inbox.put_nowait((time.monotonic() + self.simulate_latency_ms / 1000.0, sender, payload))

    def take_step_stats(self) -> Dict[str, float]:
        """Return traffic since the previous call (used for per-step metrics)."""
        s0, r0, f0, t0 = self._snap
        stats = {
            "bytes_sent": self.bytes_sent - s0,
            "bytes_received": self.bytes_received - r0,
            "frames_sent": self.frames_sent - f0,
            "avg_send_ms": (self.send_time_ms - t0) / max(1, self.frames_sent - f0),
        }
        self._snap = (self.bytes_sent, self.bytes_received, self.frames_sent, self.send_time_ms)
        return stats

    def clear_inbox(self) -> None:
        """Drop any queued frames (e.g. leftovers of a cancelled job)."""
        while not self._inbox.empty():
            self._inbox.get_nowait()


# ---------------------------------------------------------------------------
# Local (in-process) transport
# ---------------------------------------------------------------------------
class LocalHub:
    """A fake network: a dict of node_id -> LocalTransport living in one process."""

    def __init__(self):
        self.nodes: Dict[str, "LocalTransport"] = {}


class LocalTransport(Transport):
    name = "local"

    def __init__(self, node_id: str, hub: LocalHub, simulate_latency_ms: float = 0.0):
        super().__init__(node_id, simulate_latency_ms)
        self.hub = hub

    async def start(self) -> None:
        self.hub.nodes[self.node_id] = self

    async def stop(self) -> None:
        self.hub.nodes.pop(self.node_id, None)

    async def _send_raw(self, target_node_id: str, payload: bytes) -> int:
        target = self.hub.nodes.get(target_node_id)
        if target is None:
            raise ConnectionError(f"Local node '{target_node_id}' not found")
        target._deliver(self.node_id, payload, len(payload))
        return len(payload)


# ---------------------------------------------------------------------------
# Direct TCP transport
# ---------------------------------------------------------------------------
class DirectTCPTransport(Transport):
    """Point-to-point TCP. Each frame: [4B length][4B id-len][sender id][payload].

    The first frame on every new connection is a handshake carrying the room's
    shared `tcp_secret` (given to members by the coordinator). Connections that
    don't present it are closed, so strangers on the LAN can't inject tensors.
    """

    name = "tcp"

    def __init__(
        self,
        node_id: str,
        listen_host: str = "0.0.0.0",
        port: int = 0,
        secret: str = "",
        simulate_latency_ms: float = 0.0,
    ):
        super().__init__(node_id, simulate_latency_ms)
        self.listen_host = listen_host
        self.port = port
        self.secret = secret
        self._server: Optional[asyncio.base_events.Server] = None
        self._peers: Dict[str, Tuple[str, int]] = {}
        self._writers: Dict[str, asyncio.StreamWriter] = {}
        self._incoming: set = set()
        self._send_lock = asyncio.Lock()

    def register_peer(self, node_id: str, host: str, port: int) -> None:
        if self._peers.get(node_id) != (host, port):
            self._peers[node_id] = (host, port)
            old = self._writers.pop(node_id, None)
            if old:
                old.close()

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle_conn, self.listen_host, self.port)
        self.port = self._server.sockets[0].getsockname()[1]
        logger.info("TCP transport %s listening on %s:%d", self.node_id, self.listen_host, self.port)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
        for w in list(self._writers.values()) + list(self._incoming):
            w.close()
        self._writers.clear()
        self._incoming.clear()

    @staticmethod
    async def _read_frame(reader: asyncio.StreamReader) -> bytes:
        (n,) = UINT32.unpack(await reader.readexactly(UINT32.size))
        if n > MAX_FRAME_BYTES:
            raise ValueError("TCP frame too large")
        return await reader.readexactly(n)

    async def _handle_conn(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._incoming.add(writer)
        try:
            hello = await asyncio.wait_for(self._read_frame(reader), timeout=5.0)
            sender, secret = _unpack_id(hello)
            if not hmac.compare_digest(bytes(secret), self.secret.encode("utf-8")):
                logger.warning("Rejected TCP connection with bad secret (claimed id %s)", sender)
                return
            while True:
                frame = await self._read_frame(reader)
                frame_sender, payload = _unpack_id(frame)
                self._deliver(frame_sender, payload, len(frame) + UINT32.size)
        except (asyncio.IncompleteReadError, ConnectionError, asyncio.TimeoutError, ValueError):
            pass
        finally:
            self._incoming.discard(writer)
            writer.close()

    async def _connect(self, target: str) -> asyncio.StreamWriter:
        if target not in self._peers:
            raise ConnectionError(f"No TCP address known for peer '{target}'")
        host, port = self._peers[target]
        _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=10.0)
        hello = _pack_id(self.node_id) + self.secret.encode("utf-8")
        writer.write(UINT32.pack(len(hello)) + hello)
        self._writers[target] = writer
        return writer

    async def _send_raw(self, target_node_id: str, payload: bytes) -> int:
        body = _pack_id(self.node_id) + payload
        frame = UINT32.pack(len(body)) + body
        async with self._send_lock:  # keep frames from interleaving on one socket
            for attempt in range(2):  # one retry with a fresh connection
                writer = self._writers.get(target_node_id)
                try:
                    if writer is None or writer.is_closing():
                        writer = await self._connect(target_node_id)
                    writer.write(frame)
                    await writer.drain()
                    return len(frame)
                except (ConnectionError, OSError):
                    self._writers.pop(target_node_id, None)
                    if attempt == 1:
                        raise
        return 0


# ---------------------------------------------------------------------------
# Relay transport (through the coordinator)
# ---------------------------------------------------------------------------
class RelayTransport(Transport):
    """Binary WebSocket to the coordinator at /ws/relay/{room}/{node}?token=...

    Outgoing message: [4B target-id-len][target id][frame]
    Incoming message: [4B sender-id-len][sender id][frame]

    The connection is kept open for the agent's whole life and is re-opened
    automatically if it drops (sends wait until it is back).
    """

    name = "relay"

    def __init__(
        self,
        node_id: str,
        server_url: str,
        room_code: str,
        token: str,
        simulate_latency_ms: float = 0.0,
    ):
        super().__init__(node_id, simulate_latency_ms)
        base = server_url.rstrip("/").replace("https://", "wss://", 1).replace("http://", "ws://", 1)
        self.url = f"{base}/ws/relay/{room_code}/{node_id}?token={token}"
        self._ws = None
        self._connected = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._stopping = False

    async def start(self) -> None:
        self._stopping = False
        self._task = asyncio.create_task(self._run())
        await asyncio.wait_for(self._connected.wait(), timeout=15.0)

    async def _run(self) -> None:
        import websockets

        backoff = 0.5
        while not self._stopping:
            try:
                async with websockets.connect(self.url, max_size=MAX_FRAME_BYTES, ping_interval=20) as ws:
                    self._ws = ws
                    self._connected.set()
                    backoff = 0.5
                    async for msg in ws:
                        if isinstance(msg, (bytes, bytearray)):
                            try:
                                sender, payload = _unpack_id(msg)
                            except ValueError:
                                continue
                            self._deliver(sender, payload, len(msg))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # network trouble -> retry with backoff
                if not self._stopping:
                    logger.warning("Relay connection lost (%s); retrying in %.1fs", exc, backoff)
            finally:
                self._ws = None
                self._connected.clear()
            if not self._stopping:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 10.0)

    async def stop(self) -> None:
        self._stopping = True
        if self._ws is not None:
            await self._ws.close()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    async def _send_raw(self, target_node_id: str, payload: bytes) -> int:
        await asyncio.wait_for(self._connected.wait(), timeout=60.0)
        envelope = _pack_id(target_node_id) + payload
        await self._ws.send(envelope)
        return len(envelope)
