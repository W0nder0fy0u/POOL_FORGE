"""
Capacity Test: Demonstrates that split training across 2 nodes can handle models
that don't fit on a single RAM-capped node.

Grows a large MLP (by width) until single-node (with 0.5 GB RAM cap) fails, then
shows the 2-node pool succeeds on that same model, because each node only holds
HALF the layers and therefore uses HALF the memory.

Usage:  python -m benchmarks.capacity_test
Output: benchmarks/results/capacity_test.csv
        benchmarks/results/capacity_test.png
"""

import os
import sys
import time
from pathlib import Path

# ── headless matplotlib (must be set before any other import of matplotlib) ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd

# ── project imports ──────────────────────────────────────────────────────────
from common.model_zoo import get_large_mlp
from common.models import estimate_pipeline_memory

# ── configuration ────────────────────────────────────────────────────────────
# Simulated per-node RAM cap (bytes).  0.5 GB is a deliberately tight limit
# that a typical student laptop might give to a single training process.
RAM_CAP_BYTES = int(0.5 * 1024 ** 3)          # 512 MB

BATCH_SIZE   = 32
DEPTH        = 4                                # fixed number of hidden layers
WIDTHS       = [128, 256, 512, 1024, 2048, 4096]   # grow width to stress test

RESULTS_DIR  = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH  = RESULTS_DIR / "capacity_test.csv"
PNG_PATH  = RESULTS_DIR / "capacity_test.png"


# ── memory helpers ───────────────────────────────────────────────────────────

def total_model_memory_bytes(specs: list, batch_size: int) -> int:
    """Sum estimated memory across all layers (weights + grads + Adam states + activations)."""
    return sum(estimate_pipeline_memory(specs, batch_size=batch_size))


def single_node_fits(specs: list, batch_size: int, ram_cap: int) -> bool:
    """Return True if the FULL model fits within ram_cap bytes on one node."""
    return total_model_memory_bytes(specs, batch_size) <= ram_cap


def pool_fits(specs: list, batch_size: int, ram_cap: int, num_nodes: int = 2) -> bool:
    """Return True if the model fits when split equally across num_nodes nodes.

    Each node only instantiates len(specs)//num_nodes layers, so its memory
    footprint is roughly 1/num_nodes of the full model.
    """
    per_node_layers = max(1, len(specs) // num_nodes)
    # Worst-case: evaluate the largest contiguous slice any single node holds
    max_slice_bytes = sum(
        estimate_pipeline_memory(specs[i : i + per_node_layers], batch_size=batch_size)
        for i in range(0, len(specs), per_node_layers)
    )
    # max_slice_bytes above is the SUM of all slices; take the maximum slice
    slices = [
        sum(estimate_pipeline_memory(specs[i : i + per_node_layers], batch_size=batch_size))
        for i in range(0, len(specs), per_node_layers)
    ]
    return max(slices) <= ram_cap


# ── main benchmark ───────────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    """Run the capacity test across all configured widths and return results DataFrame."""
    print("=" * 60)
    print("  PoolForge Capacity Test")
    print(f"  RAM cap per node : {RAM_CAP_BYTES / 1024**2:.0f} MB")
    print(f"  Depth            : {DEPTH} hidden layers")
    print(f"  Widths tested    : {WIDTHS}")
    print("=" * 60)

    rows = []
    for width in WIDTHS:
        specs = get_large_mlp(hidden_width=width, depth=DEPTH)
        mem_bytes = total_model_memory_bytes(specs, BATCH_SIZE)
        mem_mb    = mem_bytes / 1024 ** 2
        s_fits    = single_node_fits(specs, BATCH_SIZE, RAM_CAP_BYTES)
        p_fits    = pool_fits(specs, BATCH_SIZE, RAM_CAP_BYTES, num_nodes=2)

        status = "✓ both fit" if s_fits else ("✓ pool only" if p_fits else "✗ pool too small")
        print(f"  width={width:>5d}  model={mem_mb:>8.1f} MB  single={'OK' if s_fits else 'FAIL'}  pool={'OK' if p_fits else 'FAIL'}  {status}")

        rows.append(
            {
                "width"             : width,
                "depth"             : DEPTH,
                "model_size_mb"     : round(mem_mb, 2),
                "single_node_cap_mb": round(RAM_CAP_BYTES / 1024 ** 2, 1),
                "single_node_fits"  : s_fits,
                "pool_fits"         : p_fits,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"\n  CSV saved → {CSV_PATH}")
    return df


def plot(df: pd.DataFrame) -> None:
    """Bar chart: model size vs single-node cap, colour-coded by who fits."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x       = range(len(df))
    colours = []
    for _, row in df.iterrows():
        if row["single_node_fits"]:
            colours.append("#2ca02c")   # green — both fit
        elif row["pool_fits"]:
            colours.append("#ff7f0e")   # orange — pool only
        else:
            colours.append("#d62728")   # red — neither

    bars = ax.bar(x, df["model_size_mb"], color=colours, edgecolor="black", linewidth=0.5, label="Model size (MB)")
    ax.axhline(df["single_node_cap_mb"].iloc[0], color="black", linestyle="--", linewidth=1.5,
               label=f"Single-node RAM cap ({df['single_node_cap_mb'].iloc[0]:.0f} MB)")

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"w={w}" for w in df["width"]], rotation=15)
    ax.set_xlabel("Hidden width")
    ax.set_ylabel("Estimated model memory (MB)")
    ax.set_title("Capacity Test: Model Size vs Single-Node RAM Cap\n"
                 "(green=both fit, orange=pool only, red=neither)")

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ca02c", label="Single node fits"),
        Patch(facecolor="#ff7f0e", label="Pool fits (split across 2 nodes)"),
        Patch(facecolor="#d62728", label="Neither fits"),
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.5,
                   label=f"RAM cap {df['single_node_cap_mb'].iloc[0]:.0f} MB"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=120)
    plt.close(fig)
    print(f"  PNG saved → {PNG_PATH}")


if __name__ == "__main__":
    df = run()
    plot(df)
    print("\nCapacity test complete.")
