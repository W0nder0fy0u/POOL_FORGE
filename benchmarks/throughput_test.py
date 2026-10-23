"""
Throughput Test: Steps/sec for single-stage vs 2-stage pipeline training.

Measures wall-clock time for N training steps in two configurations:
1. Single-stage  — all layers on one "node" (no inter-node transfers).
2. Two-stage     — layers split 50/50 via in-process DirectTCPTransport (loopback TCP).

Both configurations use the same tiny model so that the ONLY difference is the
pipeline-split communication overhead, not model size.

Usage:  python -m benchmarks.throughput_test
Output: benchmarks/results/throughput_test.csv
        benchmarks/results/throughput_test.png
"""

import asyncio
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common.model_zoo import get_tiny_test_model
from common.datasets import get_dataloader
from common.transports import DirectTCPTransport
from common.pipeline import PipelineStage

# ── configuration ────────────────────────────────────────────────────────────
NUM_STEPS   = 10          # steps per mode (keep low so the test is fast)
BATCH_SIZE  = 32
INPUT_SHAPE = (16,)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH    = RESULTS_DIR / "throughput_test.csv"
PNG_PATH    = RESULTS_DIR / "throughput_test.png"


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_batches(num_steps: int, batch_size: int, input_shape: tuple, seed: int = 42):
    """Return exactly num_steps (x, y) batches from the synthetic dataset."""
    loader = get_dataloader(
        dataset_name="synthetic",
        batch_size=batch_size,
        num_samples=num_steps * batch_size,
        input_shape=input_shape,
        seed=seed,
    )
    return list(loader)[:num_steps]


# ── single-stage benchmark ───────────────────────────────────────────────────

async def run_single_stage(specs: list, batches: list) -> dict:
    """Train single-stage pipeline (all layers in one PipelineStage).

    Because there is only one stage (is_first AND is_last), no inter-node
    communication happens at all — this is the baseline.
    """
    t = DirectTCPTransport(node_id="single", port=0)
    await t.start()

    try:
        stage = PipelineStage(
            stage_index=0,
            num_stages=1,
            layer_specs=specs,
            start_layer=0,
            end_layer=len(specs),
            transport=t,
            learning_rate=0.01,
        )

        step_times = []
        for i, (x, y) in enumerate(batches):
            stage.zero_grad()
            t0 = time.perf_counter()
            await stage.forward_step(x=x, labels=y, step=i)
            await stage.backward_step(step=i)
            stage.step_optimizer()
            step_times.append(time.perf_counter() - t0)

        avg_ms = (sum(step_times) / len(step_times)) * 1000
        sps    = 1.0 / (sum(step_times) / len(step_times))
        return {"mode": "single_stage", "avg_step_ms": avg_ms, "steps_per_sec": sps}

    finally:
        await t.stop()


# ── two-stage benchmark ──────────────────────────────────────────────────────

async def run_two_stage(specs: list, batches: list) -> dict:
    """Train two-stage pipeline split via loopback DirectTCPTransport."""
    split = len(specs) // 2

    t0 = DirectTCPTransport(node_id="s0", port=0)
    t1 = DirectTCPTransport(node_id="s1", port=0)
    await t0.start()
    await t1.start()
    t0.register_peer("s1", "127.0.0.1", t1.listen_port)
    t1.register_peer("s0", "127.0.0.1", t0.listen_port)

    try:
        stage0 = PipelineStage(0, 2, specs, 0,     split,       t0, next_node_id="s1", learning_rate=0.01)
        stage1 = PipelineStage(1, 2, specs, split, len(specs),  t1, prev_node_id="s0", learning_rate=0.01)

        step_times = []

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
        return {"mode": "two_stage", "avg_step_ms": avg_ms, "steps_per_sec": sps}

    finally:
        await t0.stop()
        await t1.stop()


# ── main ─────────────────────────────────────────────────────────────────────

async def run_async(num_steps: int = NUM_STEPS) -> pd.DataFrame:
    specs   = get_tiny_test_model()
    batches = _make_batches(num_steps, BATCH_SIZE, INPUT_SHAPE)

    print("=" * 60)
    print("  PoolForge Throughput Test")
    print(f"  Steps per mode : {num_steps}")
    print(f"  Batch size     : {BATCH_SIZE}")
    print("=" * 60)

    r_single = await run_single_stage(specs, batches)
    print(f"  single_stage  → {r_single['steps_per_sec']:.2f} steps/s  ({r_single['avg_step_ms']:.2f} ms/step)")

    r_two = await run_two_stage(specs, batches)
    print(f"  two_stage     → {r_two['steps_per_sec']:.2f} steps/s  ({r_two['avg_step_ms']:.2f} ms/step)")

    # Compute overhead
    overhead_pct = ((r_two["avg_step_ms"] - r_single["avg_step_ms"])
                    / r_single["avg_step_ms"] * 100)
    r_single["overhead_vs_single_pct"] = 0.0
    r_two["overhead_vs_single_pct"]    = round(overhead_pct, 2)

    print(f"\n  Pipeline-split overhead vs single-stage: {overhead_pct:.1f}%")

    df = pd.DataFrame([r_single, r_two])
    df.to_csv(CSV_PATH, index=False)
    print(f"  CSV saved → {CSV_PATH}")
    return df


def plot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    colors = ["#1f77b4", "#ff7f0e"]

    # Left: steps/sec
    axes[0].bar(df["mode"], df["steps_per_sec"], color=colors, edgecolor="black")
    axes[0].set_ylabel("Steps per second")
    axes[0].set_title("Throughput (steps/sec)\nHigher is better")

    # Right: avg step latency
    axes[1].bar(df["mode"], df["avg_step_ms"], color=colors, edgecolor="black")
    axes[1].set_ylabel("Avg step time (ms)")
    axes[1].set_title("Step Latency (ms)\nLower is better")

    for ax in axes:
        ax.set_xticklabels(df["mode"], rotation=10)

    fig.suptitle("Throughput Test: Single-Stage vs Two-Stage Pipeline", fontsize=12)
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=120)
    plt.close(fig)
    print(f"  PNG saved → {PNG_PATH}")


def run(num_steps: int = NUM_STEPS) -> pd.DataFrame:
    """Synchronous entry-point for external callers (e.g. run_all.py, tests)."""
    return asyncio.run(run_async(num_steps))


if __name__ == "__main__":
    df = run()
    plot(df)
    print("\nThroughput test complete.")
