"""Phase 1 Tests: Tensor wire format, serialization, and transports.

Acceptance criteria:
- Bit-exact round trip for PyTorch tensors.
- Large 50MB tensor serialized and deserialized without corruption.
- DirectTCP transport delivery across local nodes.
- Relay transport routing with binary WebSocket frames.
"""

import asyncio
import struct
import pytest
import torch
import websockets

from common.protocol import serialize_tensor, deserialize_tensor, UINT32_STRUCT
from common.transports import DirectTCPTransport, RelayTransport


def test_bit_exact_roundtrip_various_types():
    """Verify serialization and deserialization is bit-exact across shapes and dtypes."""
    # Test cases: (tensor, kind, step, microbatch)
    test_cases = [
        (torch.randn(10, 20, dtype=torch.float32), "activation", 1, 0),
        (torch.randn(5, 5, 5, dtype=torch.float64), "gradient", 2, 1),
        (torch.randint(0, 100, (64,), dtype=torch.int64), "labels", 3, 0),
        (torch.randint(-50, 50, (4, 8), dtype=torch.int32), "activation", 4, 2),
        (torch.tensor([True, False, True, True], dtype=torch.bool), "metadata", 5, 0),
    ]

    for orig_tensor, kind, step, microbatch in test_cases:
        extra_meta = {"sample_tag": "test_tag", "val": 42}
        payload = serialize_tensor(
            orig_tensor,
            kind=kind,
            job_id="job-phase1-test",
            step=step,
            microbatch=microbatch,
            extra=extra_meta,
        )

        assert isinstance(payload, bytes)
        assert len(payload) > 4

        rebuilt_tensor, header = deserialize_tensor(payload)

        # 1. Structural checks
        assert header["kind"] == kind
        assert header["job_id"] == "job-phase1-test"
        assert header["step"] == step
        assert header["microbatch"] == microbatch
        assert header["extra"] == extra_meta
        assert header["shape"] == list(orig_tensor.shape)

        # 2. Bit-exact equality check
        assert rebuilt_tensor.dtype == orig_tensor.dtype
        assert rebuilt_tensor.shape == orig_tensor.shape
        assert torch.equal(orig_tensor, rebuilt_tensor), "Tensor data corrupted during roundtrip!"


def test_large_50mb_tensor_roundtrip():
    """Verify a large 50 MB tensor serializes and rebuilds bit-exactly."""
    # 50 MB = 50 * 1024 * 1024 bytes = 52,428,800 bytes
    # Float32 is 4 bytes, so 13,107,200 elements
    elements = 13_107_200
    tensor_50mb = torch.randn(elements, dtype=torch.float32)
    raw_size_mb = (tensor_50mb.numel() * tensor_50mb.element_size()) / (1024 * 1024)
    assert abs(raw_size_mb - 50.0) < 0.1

    payload = serialize_tensor(tensor_50mb, kind="activation", job_id="benchmark-50mb")
    assert len(payload) > 52_428_800  # Raw bytes + header

    rebuilt_tensor, header = deserialize_tensor(payload)
    assert header["job_id"] == "benchmark-50mb"
    assert rebuilt_tensor.shape == tensor_50mb.shape
    assert torch.equal(tensor_50mb, rebuilt_tensor), "50MB tensor roundtrip failed bit-exact equality!"


@pytest.mark.asyncio
async def test_direct_tcp_transport():
    """Verify direct peer-to-peer TCP transport delivers tensors reliably."""
    node_a = DirectTCPTransport(node_id="node-a", host="127.0.0.1", port=0)
    node_b = DirectTCPTransport(node_id="node-b", host="127.0.0.1", port=0)

    await node_a.start()
    await node_b.start()

    try:
        # Register each other's dynamically allocated ports
        node_a.register_peer("node-b", "127.0.0.1", node_b.listen_port)
        node_b.register_peer("node-a", "127.0.0.1", node_a.listen_port)

        # Prepare test tensor
        test_tensor = torch.randn(32, 128, dtype=torch.float32)
        payload = serialize_tensor(test_tensor, kind="activation", step=1)

        # Send from A to B
        await node_a.send_tensor(target_node_id="node-b", payload=payload)

        # Receive on B
        sender_id, received_bytes = await node_b.receive_tensor(timeout=5.0)

        assert sender_id == "node-a"
        received_tensor, header = deserialize_tensor(received_bytes)
        assert torch.equal(test_tensor, received_tensor)
        assert header["step"] == 1

        # Check metrics
        metrics_a = node_a.get_metrics()
        metrics_b = node_b.get_metrics()
        assert metrics_a["total_bytes_sent"] > 0
        assert metrics_b["total_bytes_received"] > 0
    finally:
        await node_a.stop()
        await node_b.stop()


@pytest.mark.asyncio
async def test_relay_transport_with_mock_coordinator():
    """Verify RelayTransport routes messages between nodes through a WebSocket relay."""
    # Build a minimal relay server simulating coordinator's /ws/relay/{room_code}/{node_id}
    connected_clients = {}

    async def mock_relay_handler(websocket):
        raw_path = getattr(getattr(websocket, "request", None), "path", getattr(websocket, "path", ""))
        path_parts = raw_path.strip("/").split("/")
        # Path: ws/relay/{room_code}/{node_id}
        if len(path_parts) >= 4 and path_parts[0] == "ws" and path_parts[1] == "relay":
            room = path_parts[2]
            node_id = path_parts[3]
            connected_clients[(room, node_id)] = websocket
            try:
                async for message in websocket:
                    # Message envelope: [4-byte target_id len][target_id][payload]
                    (target_len,) = UINT32_STRUCT.unpack_from(message, 0)
                    target_id = message[UINT32_STRUCT.size : UINT32_STRUCT.size + target_len].decode("utf-8")
                    raw_payload = message[UINT32_STRUCT.size + target_len :]

                    target_ws = connected_clients.get((room, target_id))
                    if target_ws:
                        # Forward envelope: [4-byte sender_id len][sender_id][payload]
                        sender_bytes = node_id.encode("utf-8")
                        forward_frame = (
                            UINT32_STRUCT.pack(len(sender_bytes)) + sender_bytes + raw_payload
                        )
                        await target_ws.send(forward_frame)
            finally:
                connected_clients.pop((room, node_id), None)

    # Start mock relay server on dynamic port
    server = await websockets.serve(mock_relay_handler, "127.0.0.1", 0)
    server_port = server.sockets[0].getsockname()[1]
    relay_url = f"ws://127.0.0.1:{server_port}"

    room_code = "TEST01"
    agent_1 = RelayTransport(node_id="agent-1", relay_url=relay_url, room_code=room_code)
    agent_2 = RelayTransport(node_id="agent-2", relay_url=relay_url, room_code=room_code)

    await agent_1.start()
    await agent_2.start()

    # Small delay for registration in server
    await asyncio.sleep(0.05)

    try:
        sample_tensor = torch.randn(16, 64, dtype=torch.float32)
        payload = serialize_tensor(sample_tensor, kind="gradient", step=42)

        # Agent 1 sends to Agent 2 via Relay
        await agent_1.send_tensor("agent-2", payload)

        sender, recv_payload = await agent_2.receive_tensor(timeout=5.0)
        assert sender == "agent-1"

        rebuilt, header = deserialize_tensor(recv_payload)
        assert header["kind"] == "gradient"
        assert header["step"] == 42
        assert torch.equal(sample_tensor, rebuilt)

        # Verify metrics tracking
        m1 = agent_1.get_metrics()
        m2 = agent_2.get_metrics()
        assert m1["total_bytes_sent"] > 0
        assert m2["total_bytes_received"] > 0
    finally:
        await agent_1.stop()
        await agent_2.stop()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_simulated_latency():
    """Verify that simulate_latency_ms introduces artificial delay."""
    transport = DirectTCPTransport(node_id="lat-test", simulate_latency_ms=30.0)
    t0 = asyncio.get_event_loop().time()
    await transport._apply_simulated_latency()
    elapsed_ms = (asyncio.get_event_loop().time() - t0) * 1000.0
    assert elapsed_ms >= 25.0, f"Expected >= 25ms delay, got {elapsed_ms}ms"
