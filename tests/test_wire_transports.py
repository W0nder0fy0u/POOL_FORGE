"""Phase 1: tensor wire format + transports (bit-exact round trips, 50 MB tensor)."""

import asyncio
import time

import pytest
import torch

from common.protocol import UINT32, deserialize_tensor, serialize_tensor
from common.transports import DirectTCPTransport, LocalHub, LocalTransport


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int64, torch.uint8, torch.bool]
)
def test_round_trip_is_bit_exact(dtype):
    torch.manual_seed(0)
    t = (torch.randn(3, 5, 7) * 100).to(dtype)
    frame = serialize_tensor(t, kind="gradient", job_id="j1", step=4, microbatch=2)
    out, header = deserialize_tensor(frame)
    assert out.dtype == dtype and out.shape == t.shape
    assert torch.equal(out.view(torch.uint8), t.view(torch.uint8))  # same bits
    assert header["kind"] == "gradient" and header["step"] == 4 and header["microbatch"] == 2


def test_frame_layout_matches_spec():
    frame = serialize_tensor(torch.zeros(2, 2), job_id="x")
    (hlen,) = UINT32.unpack_from(frame, 0)
    assert len(frame) == 4 + hlen + 16  # 4 floats of 4 bytes


def test_non_contiguous_and_scalar_tensors():
    t = torch.arange(12.0).reshape(3, 4).t()  # transposed -> non-contiguous
    out, _ = deserialize_tensor(serialize_tensor(t))
    assert torch.equal(out, t)
    out, _ = deserialize_tensor(serialize_tensor(torch.tensor(3.5)))
    assert out.item() == 3.5 and out.shape == ()


def test_rejects_corrupt_frames():
    frame = serialize_tensor(torch.ones(4))
    with pytest.raises(ValueError):
        deserialize_tensor(frame[:-1])  # truncated payload
    with pytest.raises(ValueError):
        deserialize_tensor(b"\x00\x00")  # too short
    bad = UINT32.pack(2) + b"{}"
    with pytest.raises(ValueError):
        deserialize_tensor(bad)  # missing fields
    with pytest.raises(ValueError):
        serialize_tensor(torch.ones(1), kind="pickle_me")


def test_encrypted_round_trip():
    from common.crypto import derive_room_key

    key = derive_room_key("correct horse", "ABC123")
    t = torch.randn(10, 10)
    frame = serialize_tensor(t, encrypt_key=key)
    assert t.numpy().tobytes() not in frame  # ciphertext, not plaintext
    out, header = deserialize_tensor(frame, decrypt_key=key)
    assert header["encrypted"] and torch.equal(out, t)
    with pytest.raises(ValueError):
        deserialize_tensor(frame)  # no key
    with pytest.raises(Exception):
        deserialize_tensor(frame, decrypt_key=derive_room_key("wrong", "ABC123"))


def test_50mb_tensor_round_trip():
    t = torch.randn(50 * 1024 * 1024 // 4)  # 50 MB of float32
    out, _ = deserialize_tensor(serialize_tensor(t))
    assert torch.equal(out, t)


async def test_local_transport_and_stats():
    hub = LocalHub()
    a, b = LocalTransport("a", hub), LocalTransport("b", hub)
    await a.start()
    await b.start()
    await a.send("b", b"hello")
    assert await b.recv(timeout=1) == ("a", b"hello")
    stats = a.take_step_stats()
    assert stats["bytes_sent"] == 5 and stats["frames_sent"] == 1
    assert a.take_step_stats()["bytes_sent"] == 0  # reset after snapshot


async def test_tcp_transport_50mb_bit_exact():
    a = DirectTCPTransport("a", "127.0.0.1", 0, secret="s3cret")
    b = DirectTCPTransport("b", "127.0.0.1", 0, secret="s3cret")
    await a.start()
    await b.start()
    try:
        a.register_peer("b", "127.0.0.1", b.port)
        b.register_peer("a", "127.0.0.1", a.port)
        t = torch.randn(50 * 1024 * 1024 // 4)
        await a.send("b", serialize_tensor(t, kind="activation"))
        sender, frame = await b.recv(timeout=30)
        out, _ = deserialize_tensor(frame)
        assert sender == "a" and torch.equal(out, t)
        # and the other direction
        await b.send("a", serialize_tensor(torch.ones(3), kind="gradient"))
        sender, frame = await a.recv(timeout=5)
        assert sender == "b" and torch.equal(deserialize_tensor(frame)[0], torch.ones(3))
    finally:
        await a.stop()
        await b.stop()


async def test_tcp_rejects_wrong_secret():
    good = DirectTCPTransport("good", "127.0.0.1", 0, secret="right")
    evil = DirectTCPTransport("evil", "127.0.0.1", 0, secret="wrong")
    await good.start()
    try:
        evil.register_peer("good", "127.0.0.1", good.port)
        await evil.send("good", b"injected")
        with pytest.raises(asyncio.TimeoutError):
            await good.recv(timeout=0.5)
    finally:
        await good.stop()
        await evil.stop()


async def test_simulated_latency_delays_delivery_not_sender():
    hub = LocalHub()
    a = LocalTransport("a", hub)
    b = LocalTransport("b", hub, simulate_latency_ms=100)
    await a.start()
    await b.start()
    t0 = time.perf_counter()
    await a.send("b", b"x")
    assert time.perf_counter() - t0 < 0.05  # sender not blocked
    await b.recv(timeout=1)
    assert time.perf_counter() - t0 >= 0.095  # receiver saw ~100 ms delay
