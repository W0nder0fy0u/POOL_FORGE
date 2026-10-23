"""
Latency Test: Effect of inter-node network latency on training throughput.

Simulates real-world latency by injecting asyncio.sleep() delays inside the
transport layer (DirectTCPTransport.simulate_latency_ms).

Tests:
  - DirectTCP transport with latency_ms in [0, 20, 50, 100]
  - MockRelay transport (in-process queue, mimics WebSocket relay overhead)
    with the same latency values.

Usage:  python -m benchmarks.latency_test
Output: benchmarks/results/latency_test.csv
        benchmarks/results/latency_test.png
"""

import asyncio
import time
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common.model_zoo import get_tiny_test_model
from common.datasets import get_dataloader
from common.transports import Transport, DirectTCPTransport
from common.pipeline import PipelineStage

# ── configuration ────────────────────────────────────────────────────────────
NUM_STEPS      = 5
BATCH_SIZE     = 16
INPUT_SHAPE    = (16,)
LATENCY_VALUES = [0, 20, 50, 100]   # milliseconds

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH    = RESULTS_DIR / "latency_test.csv"
PNG_PATH    = RESULTS_DIR / "latency_test.png"


# ── MockRelayTransport ────────────────────────────────────────────────────────
# A minimal in-process "relay" transport that uses asyncio Queues to pass
# tensors — similar in spirit to the real RelayTransport (which bounces data
# through the coordinator WebSocket) but without needing a running server.

class MockRelayTransport(Transport):
    """In-process relay transport backed by asyncio queues.

    Two MockRelayTransport instances are cross-linked by calling `link()`.
    send_tensor() puts a frame on the *peer's* inbox queue (simulating the
    coordinator forwarding the message), while receive_tensor() pops from
    the local inbox queue.  Both operations apply the configured simulated
    latency so timing behaviour mirrors the real RelayTransport.
    """

    def __init__(self, node_id: str, simulate_latency_ms: float = 0.0):
        super().__init__(node_id=node_id, simulate_latency_ms=simulate_latency_ms)
        self._inbox: asyncio.Queue[Tuple[str, bytes]] = asyncio.Queue()
        self._peer: Optional["MockRelayTransport"] = None

    def link(self, peer: "MockRelayTransport") -> None:
        """Cross-link two transports so each can deliver to the other."""
        self._peer = peer
        peer._peer = self

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def send_tensor(self, target_node_id: str, payload: bytes) -> None:
        """Deliver payload to peer's inbox (simulating relay forwarding)."""
        await self._apply_simulated_latency()
        if self._peer is None:
            raise RuntimeError("MockRelayTransport not linked to a peer")
        await self._peer._inbox.put((self.node_id, payload))
        self.total_bytes_sent += len(payload)

    async def receive_tensor(self, timeout: Optional[float] = None) -> Tuple[str, bytes]:
        """Pop next tensor from the local inbox queue."""
        if timeout is not None:
            item = await asyncio.wait_for(self._inbox.get(), timeout=timeout)
        else:
            item = await self._inbox.get()
        await self._apply_simulated_latency()
        self.total_bytes_received += len(item[1])
        return item


# ── benchmark core ───────────────────────────────────────────────────────────

async def _run_two_stage(
    specs: list,
    batches: list,
    t0: Transport,
    t1: Transport,
) -> dict:
    """Run two-stage pipeline with the given pre-configured transports.

    Returns timing statistics.
    """
    split  = len(specs) // 2
    stage0 = PipelineStage(0, 2, specs, 0,     split,       t0, next_node_id=t1.node_id, learning_rate=0.01)
    stage1 = PipelineStage(1, 2, specs, split, len(specs),  t1, prev_node_id=t0.node_id, learning_rate=0.01)

    step_times: list[float] = []

    async def run_s0():
        for i, (x, y) in enumerate(batches):
            stage0.zero_grad()
            t_start = time.perf_counter()
            await stage0.forward_step(x=x, labels=y, step=i)
            await stage0.backward_step(step=i)
            stage0.step_optimizer()
            step_times.append(time.perf_counter() - t_start)

    async def run_s1():
        for i in range(len(batches)):
            stage1.zero_grad()
            await stage1.forward_step(step=i)
            await stage1.backward_step(step=i)
            stage1.step_optimizer()

    await asyncio.gather(run_s0(), run_s1())

    avg_ms = (sum(step_times) / len(step_times)) * 1000
    sps    = 1.0 / (sum(step_times) / len(step_times))
    return {"avg_step_ms": avg_ms, "steps_per_sec": sps}


async def run_async(num_steps: int = NUM_STEPS) -> pd.DataFrame:
    specs   = get_tiny_test_model()
    batches = list(
        get_dataloader("synthetic", batch_size=BATCH_SIZE,
                       num_samples=num_steps * BATCH_SIZE, input_shape=INPUT_SHAPE, seed=42)
    )[:num_steps]

    print("=" * 60)
    print("  PoolForge Latency Test")
    print(f"  Latency values : {LATENCY_VALUES} ms")
    print(f"  Steps          : {num_steps}")
    print("=" * 60)

    rows = []

    for latency_ms in LATENCY_VALUES:
        # ── DirectTCP ────────────────────────────────────────────────────
        t0_tcp = DirectTCPTransport("tcp-s0", port=0, simulate_latency_ms=latency_ms)
        t1_tcp = DirectTCPTransport("tcp-s1", port=0, simulate_latency_ms=latency_ms)
        await t0_tcp.start()
        await t1_tcp.start()
        t0_tcp.register_peer("tcp-s1", "127.0.0.1", t1_tcp.listen_port)
        t1_tcp.register_peer("tcp-s0", "127.0.0.1", t0_tcp.listen_port)

        try:
            tcp_result = await _run_two_stage(specs, batches, t0_tcp, t1_tcp)
        finally:
            await t0_tcp.stop()
            await t1_tcp.stop()

        tcp_row = {
            "transport"  : "DirectTCP",
            "latency_ms" : latency_ms,
            "steps_per_sec": round(tcp_result["steps_per_sec"], 3),
            "avg_step_ms"  : round(tcp_result["avg_step_ms"], 3),
        }
        rows.append(tcp_row)
        print(f"  DirectTCP  latency={latency_ms:>3d} ms → {tcp_result['steps_per_sec']:.2f} steps/s"
              f"  ({tcp_result['avg_step_ms']:.1f} ms/step)")

        # ── MockRelay ─────────────────────────────────────────────────────
        t0_relay = MockRelayTransport("relay-s0", simulate_latency_ms=latency_ms)
        t1_relay = MockRelayTransport("relay-s1", simulate_latency_ms=latency_ms)
        t0_relay.link(t1_relay)
        await t0_relay.start()
        await t1_relay.start()

        try:
            relay_result = await _run_two_stage(specs, batches, t0_relay, t1_relay)
        finally:
            await t0_relay.stop()
            await t1_relay.stop()

        relay_row = {
            "transport"  : "MockRelay",
            "latency_ms" : latency_ms,
            "steps_per_sec": round(relay_result["steps_per_sec"], 3),
            "avg_step_ms"  : round(relay_result["avg_step_ms"], 3),
        }
        rows.append(relay_row)
        print(f"  MockRelay  latency={latency_ms:>3d} ms → {relay_result['steps_per_sec']:.2f} steps/s"
              f"  ({relay_result['avg_step_ms']:.1f} ms/step)")

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"\n  CSV saved → {CSV_PATH}")
    return df


def plot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for transport, grp in df.groupby("transport"):
        color = "#1f77b4" if transport == "DirectTCP" else "#ff7f0e"
        axes[0].plot(grp["latency_ms"], grp["steps_per_sec"], marker="o", label=transport, color=color)
        axes[1].plot(grp["latency_ms"], grp["avg_step_ms"],   marker="o", label=transport, color=color)

    axes[0].set_xlabel("Injected latency (ms)")
    axes[0].set_ylabel("Steps per second")
    axes[0].set_title("Throughput vs Latency\n(higher is better)")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].set_xlabel("Injected latency (ms)")
    axes[1].set_ylabel("Avg step time (ms)")
    axes[1].set_title("Step Latency vs Injected Latency\n(lower is better)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Latency Test: DirectTCP vs MockRelay Transport", fontsize=12)
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=120)
    plt.close(fig)
    print(f"  PNG saved → {PNG_PATH}")


def run(num_steps: int = NUM_STEPS) -> pd.DataFrame:
    """Synchronous entry-point."""
    return asyncio.run(run_async(num_steps))


if __name__ == "__main__":
    df = run()
    plot(df)
    print("\nLatency test complete.")
