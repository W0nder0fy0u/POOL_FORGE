"""Run every PoolForge benchmark and write CSV + PNG + results.md to benchmarks/results/.

    python -m benchmarks.run_all            # everything (~10-15 min on a laptop)
    python -m benchmarks.run_all capacity latency   # only some
    python -m benchmarks.run_all --steps 15         # shorter runs

Benchmarks (from the project spec):
  a. capacity    grow a model until one RAM-capped node can't fit it; show the pool still can
  b. ram         RAM per node: single node vs 2-node pool
  c. speed       steps/sec: single node vs 2-node pool
  d. transport   relay vs direct TCP
  e. latency     injected one-way latency 0/20/50/100 ms and the slowdown it causes
  f. microbatch  effect of micro-batching (with and without latency)

Everything runs on ONE machine, so "network" means localhost plus simulated
latency; real Wi-Fi/Internet adds more. Each simulated laptop gets 4 threads.
"""

import argparse
import csv
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import psutil  # noqa: E402

from benchmarks.harness import THREADS_PER_NODE, Pool  # noqa: E402
from common.model_zoo import get_model  # noqa: E402
from common.specs import count_params, estimate_memory  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
ORANGE, BLUE, INK, MUTED, GRID = "#eb6834", "#2a78d6", "#1f1a17", "#8a7d73", "#efe3d8"

plt.rcParams.update({
    "figure.dpi": 120, "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.titleweight": "bold",
    "axes.titlesize": 11, "legend.frameon": False,
})


def write_csv(name, rows):
    path = OUT / f"{name}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path.relative_to(OUT.parent.parent)}")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def label_bars(ax, bars, fmt):
    for b in bars:
        ax.annotate(fmt(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()), ha="center", va="bottom",
                    fontsize=9, color=INK, xytext=(0, 2), textcoords="offset points")


def mlp_cfg(width, depth=8, steps=20, batch=64, **kw):
    return {"model": "large_mlp", "model_options": {"width": width, "depth": depth}, "dataset": "synthetic",
            "batch_size": batch, "epochs": 1, "steps_per_epoch": steps, "lr": 0.0005, "checkpoint_every": 1000, **kw}


# --------------------------------------------------------------------- a
def bench_capacity(pool, steps):
    """Grow large_mlp; 1 node capped at CAP GB vs 2 nodes capped at CAP GB each."""
    cap = 0.3
    widths = [1024, 1280, 1536, 1792, 2048, 2304]
    rows = []
    results = {}
    for n_nodes in (1, 2):
        for w in widths:
            # fresh agents for every size, so "idle RAM" is not polluted by the previous model
            pool.new_room()
            for i in range(n_nodes):
                pool.add_agent(f"capped{i}", ram_cap_gb=cap)
            pool.wait_nodes(n_nodes)
            cfg = mlp_cfg(w, steps=max(5, steps // 3))
            try:
                pool.plan(cfg)
                r = pool.run_job(cfg, warmup=1)
                results[(n_nodes, w)] = ("trained", r["avg_step_ms"], sum(v["peak_ram_mb"] - v["idle_ram_mb"] for v in r["per_node"].values()))
            except RuntimeError as e:
                results[(n_nodes, w)] = ("pool too small" if "pool too small" in str(e) else f"error: {e}", None, None)
            print(f"  capacity: {n_nodes} node(s), width {w}: {results[(n_nodes, w)][0]}")
    for w in widths:
        m = get_model("large_mlp", width=w, depth=8)
        est = sum(estimate_memory(m["specs"], tuple(m["input_shape"]), 64)) / 2**20
        s, p = results[(1, w)], results[(2, w)]
        rows.append({"width": w, "params_millions": round(sum(count_params(x) for x in m["specs"]) / 1e6, 2),
                     "estimated_mb": round(est), "cap_per_node_gb": cap,
                     "single_node": s[0], "pool_2_nodes": p[0],
                     "pool_step_ms": round(p[1], 1) if p[1] else "", "pool_measured_model_ram_mb": round(p[2]) if p[2] else ""})
    write_csv("a_capacity", rows)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    xs = [r["params_millions"] for r in rows]
    ax.plot(xs, [r["estimated_mb"] for r in rows], color=INK, lw=2, marker="o", ms=5, label="model needs (estimate)")
    ax.axhline(cap * 1024 * 0.9, color=BLUE, lw=2, ls="--")
    ax.axhline(2 * cap * 1024 * 0.9, color=ORANGE, lw=2, ls="--")
    ax.annotate("1 laptop usable RAM", (xs[0], cap * 1024 * 0.9), color=BLUE, xytext=(0, 4), textcoords="offset points")
    ax.annotate("2-laptop pool usable RAM", (xs[0], 2 * cap * 1024 * 0.9), color=ORANGE, xytext=(0, 4), textcoords="offset points")
    for r in rows:
        ok1, ok2 = r["single_node"] == "trained", r["pool_2_nodes"] == "trained"
        tag = "both" if ok1 else ("pool only" if ok2 else "neither")
        ax.annotate(tag, (r["params_millions"], r["estimated_mb"]), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color=MUTED)
    ax.set_xlabel("model parameters (millions)")
    ax.set_ylabel("memory (MB)")
    ax.set_title(f"a. Capacity: each laptop capped at {cap} GB")
    ax.legend(loc="lower right")
    save(fig, "a_capacity")
    return rows


# ------------------------------------------------------------------- b, c
def bench_single_vs_pool(pool, steps):
    width = 1536
    out = {}
    for n in (1, 2):
        pool.new_room()
        for i in range(n):
            pool.add_agent(f"node{i}")
        pool.wait_nodes(n)
        out[n] = pool.run_job(mlp_cfg(width, steps=steps))
        print(f"  single vs pool: {n} node(s): {out[n]['avg_step_ms']:.0f} ms/step")
    rows = []
    for n, r in out.items():
        for nid, v in r["per_node"].items():
            layers = next(p for p in r["plan"] if p["node_id"] == nid)
            rows.append({"setup": "single node" if n == 1 else "2-node pool", "node": nid,
                         "layers": f"{layers['start_layer']}-{layers['end_layer'] - 1}",
                         "peak_agent_ram_mb": round(v["peak_ram_mb"]), "idle_agent_ram_mb": round(v["idle_ram_mb"]),
                         "model_ram_mb": round(v["peak_ram_mb"] - v["idle_ram_mb"]),
                         "avg_step_ms": round(r["avg_step_ms"], 1), "steps_per_s": round(r["steps_per_s"], 2)})
    write_csv("b_ram_per_node", rows)
    write_csv("c_steps_per_sec", [{"setup": "single node" if n == 1 else "2-node pool (relay)", "nodes": n,
                                   "avg_step_ms": round(r["avg_step_ms"], 1), "steps_per_s": round(r["steps_per_s"], 2),
                                   "samples_per_s": round(64 * r["steps_per_s"], 1)} for n, r in out.items()])

    fig, ax = plt.subplots(figsize=(6, 3.6))
    labels = [f"{r['setup']}\n{r['node']} (layers {r['layers']})" for r in rows]
    bars = ax.bar(labels, [r["model_ram_mb"] for r in rows], color=[BLUE if r["setup"] == "single node" else ORANGE for r in rows], width=0.55)
    label_bars(ax, bars, lambda v: f"{v:.0f} MB")
    ax.set_ylabel("RAM used for model + training (MB)")
    ax.set_title(f"b. RAM per node, large_mlp width {width} (peak minus idle agent)")
    save(fig, "b_ram_per_node")

    fig, ax = plt.subplots(figsize=(5, 3.4))
    bars = ax.bar(["single node", "2-node pool"], [out[1]["steps_per_s"], out[2]["steps_per_s"]], color=[BLUE, ORANGE], width=0.5)
    label_bars(ax, bars, lambda v: f"{v:.2f}")
    ax.set_ylabel("steps per second (higher is better)")
    ax.set_title("c. Training speed: single vs pool")
    save(fig, "c_steps_per_sec")
    return rows, out


# --------------------------------------------------------------------- d
def bench_transport(pool, steps):
    rows = []
    for transport in ("relay", "tcp"):
        pool.new_room()
        for i in range(2):
            pool.add_agent(f"{transport}{i}", transport=transport)
        pool.wait_nodes(2)
        r = pool.run_job(mlp_cfg(1536, steps=steps))
        assert r["transport"] == transport, r["transport"]
        bps = sum(v["bytes_sent_per_step"] for v in r["per_node"].values())
        rows.append({"transport": transport, "avg_step_ms": round(r["avg_step_ms"], 1), "steps_per_s": round(r["steps_per_s"], 2),
                     "bytes_per_step": round(bps), "avg_send_ms_per_frame": round(sum(v["avg_send_ms"] for v in r["per_node"].values()) / 2, 2)})
        print(f"  transport {transport}: {r['avg_step_ms']:.0f} ms/step")
    write_csv("d_relay_vs_tcp", rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
    for ax, key, title in ((axes[0], "avg_step_ms", "step time (ms, lower is better)"),
                           (axes[1], "avg_send_ms_per_frame", "send time per frame (ms)")):
        bars = ax.bar([r["transport"] for r in rows], [r[key] for r in rows], color=[ORANGE, BLUE], width=0.5)
        label_bars(ax, bars, lambda v: f"{v:.1f}")
        ax.set_title(title, fontsize=10)
    fig.suptitle("d. Relay (via coordinator) vs direct TCP", fontweight="bold", fontsize=11)
    save(fig, "d_relay_vs_tcp")
    return rows


# --------------------------------------------------------------------- e
def bench_latency(pool, steps):
    rows = []
    base = None
    for lat in (0, 20, 50, 100):
        pool.new_room()
        for i in range(2):
            pool.add_agent(f"lat{i}", latency_ms=lat)
        pool.wait_nodes(2)
        r = pool.run_job(mlp_cfg(1024, steps=steps))
        base = base or r["avg_step_ms"]
        rows.append({"one_way_latency_ms": lat, "avg_step_ms": round(r["avg_step_ms"], 1), "steps_per_s": round(r["steps_per_s"], 2),
                     "slowdown_vs_0ms": round(r["avg_step_ms"] / base, 2), "expected_extra_ms(2 hops)": 2 * lat})
        print(f"  latency {lat} ms: {r['avg_step_ms']:.0f} ms/step")
    write_csv("e_latency", rows)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    xs = [r["one_way_latency_ms"] for r in rows]
    ax.plot(xs, [r["avg_step_ms"] for r in rows], color=ORANGE, lw=2, marker="o", ms=6, label="measured step time")
    ax.plot(xs, [rows[0]["avg_step_ms"] + 2 * x for x in xs], color=MUTED, lw=1.5, ls="--", label="0 ms time + 2 x latency")
    for r in rows:
        ax.annotate(f"x{r['slowdown_vs_0ms']}", (r["one_way_latency_ms"], r["avg_step_ms"]), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.set_xlabel("injected one-way network latency (ms)")
    ax.set_ylabel("step time (ms)")
    ax.set_title("e. Network latency slows every step")
    ax.legend()
    save(fig, "e_latency")
    return rows


# --------------------------------------------------------------------- f
def bench_microbatch(pool, steps):
    rows = []
    for lat in (0, 50):
        pool.new_room()
        for i in range(2):
            pool.add_agent(f"mb{i}", latency_ms=lat)
        pool.wait_nodes(2)
        for mb in (1, 2, 4, 8):
            r = pool.run_job(mlp_cfg(1536, steps=steps, microbatches=mb))
            rows.append({"one_way_latency_ms": lat, "microbatches": mb, "avg_step_ms": round(r["avg_step_ms"], 1),
                         "steps_per_s": round(r["steps_per_s"], 2)})
            print(f"  microbatch: latency {lat} ms, M={mb}: {r['avg_step_ms']:.0f} ms/step")
    write_csv("f_microbatch", rows)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for lat, color in ((0, BLUE), (50, ORANGE)):
        sub = [r for r in rows if r["one_way_latency_ms"] == lat]
        ax.plot([r["microbatches"] for r in sub], [r["avg_step_ms"] for r in sub], color=color, lw=2, marker="o", ms=6)
        ax.annotate(f"{lat} ms latency", (sub[-1]["microbatches"], sub[-1]["avg_step_ms"]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=color)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    ax.set_xlim(0.8, 14)
    ax.set_xlabel("micro-batches per step (batch 64)")
    ax.set_ylabel("step time (ms)")
    ax.set_title("f. Micro-batching, 2-node pool")
    save(fig, "f_microbatch")
    return rows


def table(rows):
    keys = list(rows[0].keys())
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("only", nargs="*", help="capacity ram speed transport latency microbatch (default: all)")
    ap.add_argument("--steps", type=int, default=25)
    args = ap.parse_args()
    want = set(args.only) or {"capacity", "speed", "transport", "latency", "microbatch"}
    if "ram" in want:
        want.add("speed")
    OUT.mkdir(exist_ok=True)
    sections = []
    t0 = time.time()
    with Pool() as pool:
        if "capacity" in want:
            print("a. capacity")
            sections.append(("a. Capacity: one capped laptop vs a 2-laptop pool", "a_capacity", table(bench_capacity(pool, args.steps))))
        if "speed" in want:
            print("b/c. single vs pool")
            rows, out = bench_single_vs_pool(pool, args.steps)
            sections.append(("b. RAM per node: single vs pooled", "b_ram_per_node", table(rows)))
            sections.append(("c. Steps per second: single vs pooled", "c_steps_per_sec",
                             table([{"setup": k, "avg_step_ms": round(v["avg_step_ms"], 1), "steps_per_s": round(v["steps_per_s"], 2)}
                                    for k, v in (("single node", out[1]), ("2-node pool", out[2]))])))
        if "transport" in want:
            print("d. relay vs tcp")
            sections.append(("d. Relay vs direct TCP", "d_relay_vs_tcp", table(bench_transport(pool, args.steps))))
        if "latency" in want:
            print("e. latency")
            sections.append(("e. Injected latency", "e_latency", table(bench_latency(pool, args.steps))))
        if "microbatch" in want:
            print("f. micro-batching")
            sections.append(("f. Micro-batching", "f_microbatch", table(bench_microbatch(pool, args.steps))))

    machine = f"{platform.system()} {platform.machine()}, {psutil.cpu_count()} CPU threads, {psutil.virtual_memory().total / 2**30:.0f} GB RAM"
    raw = OUT / "raw_tables.md"
    with open(raw, "w") as f:
        f.write(f"# Raw benchmark tables\n\nMachine: {machine}. {THREADS_PER_NODE} threads per simulated laptop. "
                f"Generated in {time.time() - t0:.0f}s by `python -m benchmarks.run_all`.\n\n")
        for title, img, tbl in sections:
            f.write(f"## {title}\n\n![{title}]({img}.png)\n\n{tbl}\n\n")
    print(f"done in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
