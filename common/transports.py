"""PoolForge Transport Implementations.

Defines the pluggable Transport interface and provides:
1. RelayTransport: Default WAN-ready transport routing binary tensor frames through
   the Coordinator via WebSockets (works behind home routers/NATs without port forwarding).
2. DirectTCPTransport: High-throughput raw TCP transport for LAN/Tailscale environments.
"""

import abc
import asyncio
import logging
import struct
import time
from typing import Any, Dict, Optional, Tuple
import websockets

logger = logging.getLogger("poolforge.transports")

# 4-byte big-endian length struct used in framing
UINT32_STRUCT = struct.Struct(">I")


class Transport(abc.ABC):
    """Abstract base class for all PoolForge data plane tensor transports."""

    def __init__(self, node_id: str, simulate_latency_ms: float = 0.0):
        self.node_id = node_id
        self.simulate_latency_ms = simulate_latency_ms
        self.total_bytes_sent = 0
        self.total_bytes_received = 0
        self.send_latencies_ms: list[float] = []
        self.recv_latencies_ms: list[float] = []
        self._running = False

    @abc.abstractmethod
    async def start(self) -> None:
        """Initialize transport listener or client connections."""
        pass

    @abc.abstractmethod
    async def stop(self) -> None:
        """Gracefully shut down transport connections."""
        pass

    @abc.abstractmethod
    async def send_tensor(self, target_node_id: str, payload: bytes) -> None:
        """Send framed tensor payload to a target node.

        Args:
            target_node_id: Identifier of the destination node.
            payload: Binary serialized tensor bytes.
        """
        pass

    @abc.abstractmethod
    async def receive_tensor(self, timeout: Optional[float] = None) -> Tuple[str, bytes]:
        """Receive the next available tensor from the transport queue.

        Args:
            timeout: Maximum seconds to wait before raising TimeoutError.

        Returns:
            Tuple of (sender_node_id, payload_bytes).
        """
        pass

    def get_metrics(self) -> Dict[str, Any]:
        """Return cumulative network transfer and latency metrics."""
        avg_send_lat = (
            sum(self.send_latencies_ms) / len(self.send_latencies_ms)
            if self.send_latencies_ms
            else 0.0
        )
        avg_recv_lat = (
            sum(self.recv_latencies_ms) / len(self.recv_latencies_ms)
            if self.recv_latencies_ms
            else 0.0
        )
        return {
            "node_id": self.node_id,
            "transport_type": self.__class__.__name__,
            "total_bytes_sent": self.total_bytes_sent,
            "total_bytes_received": self.total_bytes_received,
            "total_mb_sent": round(self.total_bytes_sent / (1024 * 1024), 2),
            "total_mb_received": round(self.total_bytes_received / (1024 * 1024), 2),
            "avg_send_latency_ms": round(avg_send_lat, 2),
            "avg_recv_latency_ms": round(avg_recv_lat, 2),
            "simulated_latency_ms": self.simulate_latency_ms,
        }

    async def _apply_simulated_latency(self) -> None:
        """Simulate network latency if configured."""
        if self.simulate_latency_ms > 0:
            await asyncio.sleep(self.simulate_latency_ms / 1000.0)


class DirectTCPTransport(Transport):
    """Direct peer-to-peer TCP transport for LAN or high-speed private networks."""

    def __init__(
        self,
        node_id: str,
        host: str = "127.0.0.1",
        port: int = 0,
        simulate_latency_ms: float = 0.0,
    ):
        super().__init__(node_id=node_id, simulate_latency_ms=simulate_latency_ms)
        self.host = host
        self.port = port
        self.listen_port = port
        self._server: Optional[asyncio.Server] = None
        self._inbox: asyncio.Queue[Tuple[str, bytes]] = asyncio.Queue()
        self._peer_addresses: Dict[str, Tuple[str, int]] = {}
        self._peer_connections: Dict[str, Tuple[asyncio.StreamReader, asyncio.StreamWriter]] = {}

    def register_peer(self, node_id: str, host: str, port: int) -> None:
        """Register the network location of a peer node."""
        self._peer_addresses[node_id] = (host, port)

    async def start(self) -> None:
        """Start listening on the TCP port."""
        self._server = await asyncio.start_server(
            self._handle_client,
            self.host,
            self.port,
        )
        # Capture the dynamically assigned port if 0 was given
        sockets = self._server.sockets
        if sockets:
            self.listen_port = sockets[0].getsockname()[1]
        self._running = True
        logger.info(f"DirectTCPTransport [{self.node_id}] listening on {self.host}:{self.listen_port}")

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle incoming TCP connections from peer nodes."""
        try:
            while self._running:
                # Frame layout: [4-byte total length][4-byte sender_id len][sender_id][payload]
                total_len_data = await reader.readexactly(UINT32_STRUCT.size)
                (total_len,) = UINT32_STRUCT.unpack(total_len_data)

                frame = await reader.readexactly(total_len)
                (sender_len,) = UINT32_STRUCT.unpack_from(frame, 0)
                sender_id = frame[UINT32_STRUCT.size : UINT32_STRUCT.size + sender_len].decode("utf-8")
                payload = frame[UINT32_STRUCT.size + sender_len :]

                t0 = time.perf_counter()
                await self._apply_simulated_latency()
                self.total_bytes_received += len(frame) + UINT32_STRUCT.size
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                self.recv_latencies_ms.append(elapsed_ms)

                await self._inbox.put((sender_id, payload))
        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _get_or_create_connection(
        self, target_node_id: str
    ) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Obtain a reusable TCP connection to a peer, opening one if necessary."""
        if target_node_id in self._peer_connections:
            reader, writer = self._peer_connections[target_node_id]
            if not writer.is_closing():
                return reader, writer

        if target_node_id not in self._peer_addresses:
            raise KeyError(
                f"Peer node '{target_node_id}' address unknown to DirectTCPTransport"
            )

        host, port = self._peer_addresses[target_node_id]
        reader, writer = await asyncio.open_connection(host, port)
        self._peer_connections[target_node_id] = (reader, writer)
        return reader, writer

    async def send_tensor(self, target_node_id: str, payload: bytes) -> None:
        """Send a tensor frame to a peer via direct TCP socket."""
        t0 = time.perf_counter()
        await self._apply_simulated_latency()

        sender_id_bytes = self.node_id.encode("utf-8")
        sender_len = len(sender_id_bytes)
        # Body = [4-byte sender_len][sender_id][payload]
        body = UINT32_STRUCT.pack(sender_len) + sender_id_bytes + payload
        total_len = len(body)
        frame = UINT32_STRUCT.pack(total_len) + body

        reader, writer = await self._get_or_create_connection(target_node_id)
        writer.write(frame)
        await writer.drain()

        self.total_bytes_sent += len(frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.send_latencies_ms.append(elapsed_ms)

    async def receive_tensor(self, timeout: Optional[float] = None) -> Tuple[str, bytes]:
        """Retrieve next tensor from the incoming TCP queue."""
        if timeout is not None:
            return await asyncio.wait_for(self._inbox.get(), timeout=timeout)
        return await self._inbox.get()

    async def stop(self) -> None:
        """Close listening server and all peer TCP connections."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        for target_id, (_, writer) in list(self._peer_connections.items()):
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
        self._peer_connections.clear()


class RelayTransport(Transport):
    """Outbound WebSocket transport routing tensors through the Coordinator.

    Works automatically behind student routers, NATs, and campus firewalls without
    port forwarding or static IP addresses.
    """

    def __init__(
        self,
        node_id: str,
        relay_url: str,
        room_code: str,
        simulate_latency_ms: float = 0.0,
    ):
        super().__init__(node_id=node_id, simulate_latency_ms=simulate_latency_ms)
        self.relay_url = relay_url
        self.room_code = room_code
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._inbox: asyncio.Queue[Tuple[str, bytes]] = asyncio.Queue()
        self._receive_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Establish outbound WebSocket connection to coordinator relay endpoint."""
        # Convert http/https URL scheme to ws/wss if needed
        ws_url = self.relay_url
        if ws_url.startswith("http://"):
            ws_url = "ws://" + ws_url[7:]
        elif ws_url.startswith("https://"):
            ws_url = "wss://" + ws_url[8:]

        # Endpoint convention: /ws/relay/{room_code}/{node_id}
        full_url = f"{ws_url.rstrip('/')}/ws/relay/{self.room_code}/{self.node_id}"
        # Set max_size=100MB to easily accommodate large tensor frames
        self._ws = await websockets.connect(full_url, max_size=100 * 1024 * 1024)
        self._running = True
        self._receive_task = asyncio.create_task(self._listen_loop())
        logger.info(f"RelayTransport [{self.node_id}] connected to coordinator at {full_url}")

    async def _listen_loop(self) -> None:
        """Receive routed binary frames forwarded by the coordinator."""
        try:
            while self._running and self._ws is not None:
                message = await self._ws.recv()
                if not isinstance(message, bytes):
                    continue

                t0 = time.perf_counter()
                await self._apply_simulated_latency()
                # Incoming frame from coordinator: [4-byte sender_id len][sender_id][payload]
                if len(message) < UINT32_STRUCT.size:
                    continue
                (sender_len,) = UINT32_STRUCT.unpack_from(message, 0)
                sender_id = message[UINT32_STRUCT.size : UINT32_STRUCT.size + sender_len].decode("utf-8")
                payload = message[UINT32_STRUCT.size + sender_len :]

                self.total_bytes_received += len(message)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                self.recv_latencies_ms.append(elapsed_ms)

                await self._inbox.put((sender_id, payload))
        except (websockets.ConnectionClosed, asyncio.CancelledError):
            pass

    async def send_tensor(self, target_node_id: str, payload: bytes) -> None:
        """Send tensor frame to coordinator with target node routing envelope."""
        if not self._ws or not self._running:
            raise ConnectionError(f"RelayTransport [{self.node_id}] is not connected")

        t0 = time.perf_counter()
        await self._apply_simulated_latency()

        # Outbound frame: [4-byte target_id len][target_id][payload]
        target_bytes = target_node_id.encode("utf-8")
        target_len = len(target_bytes)
        envelope = UINT32_STRUCT.pack(target_len) + target_bytes + payload

        await self._ws.send(envelope)

        self.total_bytes_sent += len(envelope)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.send_latencies_ms.append(elapsed_ms)

    async def receive_tensor(self, timeout: Optional[float] = None) -> Tuple[str, bytes]:
        """Retrieve next routed tensor from inbox queue."""
        if timeout is not None:
            return await asyncio.wait_for(self._inbox.get(), timeout=timeout)
        return await self._inbox.get()

    async def stop(self) -> None:
        """Gracefully disconnect from coordinator relay."""
        self._running = False
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        if self._ws:
            await self._ws.close()
            self._ws = None
