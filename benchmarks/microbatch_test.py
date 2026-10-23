"""
Microbatch Test: Effect of micro-batch size on pipeline utilization.

In a pipeline-parallel setup the TOTAL batch is split into N micro-batches.
Larger micro-batches:
  - Increase GPU utilisation per pass (amortise kernel-launch overhead).
  - Expose more pipeline parallelism (stages can overlap forward passes
    from different micro-batches).
  - But require more per-step activation memory.

This test uses a 2-stage split and varies num_microbatches in [1, 2, 4, 8].
For each configuration it measures:
  - avg step time (ms)
  - effective throughput (samples/sec)  = batch_size / step_time

Usage:  python -m benchmarks.microbatch_test
Output: benchmarks/results/microbatch_test.csv
        benchmarks/results/microbatch_test.png
"""

import asyncio
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from common.model_zoo import get_tiny_test_model
from common.datasets import get_dataloader
from common.transports import DirectTCPTransport
from common.pipeline import PipelineStage

# ── configuration ────────────────────────────────────────────────────────────
NUM_STEPS         = 8
BATCH_SIZE        = 32          # total samples per step (fixed)
MICROBATCH_SIZES  = [1, 2, 4, 8]   # num_microbatches to test
INPUT_SHAPE       = (16,)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH    = RESULTS_DIR / "microbatch_test.csv"
PNG_PATH    = RESULTS_DIR / "microbatch_test.png"


# ── benchmark core ───────────────────────────────────────────────────────────

async def run_with_microbatch(
    specs: list,
    batches: list,
    num_microbatches: int,
) -> dict:
    """Run two-stage pipeline with a given number of micro-batches per step.

    Each step's batch is split into num_microbatches equal chunks.
    All micro-batch forward passes are issued first; then all backward passes.
    This is the standard 1F1B / GPipe-style micro-batch schedule.
    """
    split = len(specs) // 2

    t0 = DirectTCPTransport("mb-s0", port=0)
    t1 = DirectTCPTransport("mb-s1", port=0)
    await t0.start()
    await t1.start()
    t0.register_peer("mb-s1", "127.0.0.1", t1.listen_port)
    t1.register_peer("mb-s0", "127.0.0.1", t0.listen_port)

    try:
        stage0 = PipelineStage(0, 2, specs, 0,     split,       t0, next_node_id="mb-s1", learning_rate=0.01)
        stage1 = PipelineStage(1, 2, specs, split, len(specs),  t1, prev_node_id="mb-s0", learning_rate=0.01)

        # loss_scale normalises each micro-batch gradient so that
        # accumulated gradients equal the full-batch gradient
        loss_scale = 1.0 / num_microbatches
        step_times: list[float] = []

        async def run_s0():
            for step_i, (x_full, y_full) in enumerate(batches):
                stage0.zero_grad()
                # Split batch into micro-batches
                xs = torch.chunk(x_full, num_microbatches)
                ys = torch.chunk(y_full, num_microbatches)

                t_start = time.perf_counter()

                # Forward all micro-batches
                for m in range(num_microbatches):
                    await stage0.forward_step(x=xs[m], labels=ys[m], step=step_i, microbatch=m)

                # Backward all micro-batches
                for m in range(num_microbatches):
                    await stage0.backward_step(step=step_i, microbatch=m)

                stage0.step_optimizer()
                step_times.append(time.perf_counter() - t_start)

        async def run_s1():
            for step_i in range(len(batches)):
                stage1.zero_grad()

                # Forward all micro-batches
                for m in range(num_microbatches):
                    await stage1.forward_step(step=step_i, microbatch=m)

                # Backward all micro-batches (with gradient scaling)
                for m in range(num_microbatches):
                    await stage1.backward_step(step=step_i, microbatch=m, loss_scale=loss_scale)

                stage1.step_optimizer()

        await asyncio.gather(run_s0(), run_s1())

        avg_ms         = (sum(step_times) / len(step_times)) * 1000
        samples_per_s  = BATCH_SIZE / (sum(step_times) / len(step_times))
        return {
            "num_microbatches"   : num_microbatches,
            "avg_step_ms"        : round(avg_ms, 3),
            "samples_per_sec"    : round(samples_per_s, 2),
        }

    finally:
        await t0.stop()
        await t1.stop()


async def run_async(num_steps: int = NUM_STEPS) -> pd.DataFrame:
    specs = get_tiny_test_model()
    batches = list(
        get_dataloader("synthetic", batch_size=BATCH_SIZE,
                       num_samples=num_steps * BATCH_SIZE, input_shape=INPUT_SHAPE, seed=42)
    )[:num_steps]

    print("=" * 60)
    print("  PoolForge Microbatch Test")
    print(f"  Microbatch sizes : {MICROBATCH_SIZES}")
    print(f"  Total batch size : {BATCH_SIZE}")
    print(f"  Steps            : {num_steps}")
    print("=" * 60)

    rows = []
    for nmb in MICROBATCH_SIZES:
        # Ensure batch is evenly divisible
        if BATCH_SIZE % nmb != 0:
            print(f"  Skipping nmb={nmb} (batch size {BATCH_SIZE} not divisible)")
            continue

        result = await run_with_microbatch(specs, batches, num_microbatches=nmb)
        rows.append(result)
        print(f"  microbatches={nmb}  → {result['samples_per_sec']:.1f} samples/s"
              f"  ({result['avg_step_ms']:.1f} ms/step)")

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"\n  CSV saved → {CSV_PATH}")
    return df


def plot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    x = df["num_microbatches"].astype(str)

    axes[0].bar(x, df["samples_per_sec"], color="#2ca02c", edgecolor="black")
    axes[0].set_xlabel("Number of micro-batches")
    axes[0].set_ylabel("Samples per second")
    axes[0].set_title("Effective Throughput\n(higher is better)")

    axes[1].bar(x, df["avg_step_ms"], color="#9467bd", edgecolor="black")
    axes[1].set_xlabel("Number of micro-batches")
    axes[1].set_ylabel("Avg step time (ms)")
    axes[1].set_title("Step Latency\n(lower is better)")

    fig.suptitle("Microbatch Test: Effect of Micro-batch Size on Throughput", fontsize=12)
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
    print("\nMicrobatch test complete.")
