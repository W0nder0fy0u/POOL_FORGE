"""GPU / VRAM sharing: planner sizing, device choice, and real CUDA training when available."""

import pytest
import torch

from common.datasets import synthetic_batches
from common.model_zoo import get_model
from common.pipeline import even_split, train_single_process, train_split_locally
from coordinator.hub import NodeLive, node_capacity
from coordinator.scheduler import NodeCapacity, plan_split

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU")


def test_gpu_node_is_planned_by_vram_not_ram():
    gpu = NodeLive("gpu", "h", "o", stats={"device": "cuda", "vram_free_gb": 3.5, "ram_free_gb": 0.5})
    cpu = NodeLive("cpu", "h", "o", stats={"device": "cpu", "ram_free_gb": 6.0})
    mac = NodeLive("mac", "h", "o", stats={"device": "mps", "ram_free_gb": 4.0})
    assert node_capacity(gpu) == NodeCapacity("gpu", 3.5, "cuda")
    assert node_capacity(cpu) == NodeCapacity("cpu", 6.0, "cpu")
    assert node_capacity(mac) == NodeCapacity("mac", 4.0, "mps")  # Apple GPU shares RAM


def test_plan_records_device_and_says_vram_when_too_small():
    m = get_model("large_mlp", width=1024, depth=8)
    plan = plan_split(m["specs"], m["input_shape"], [NodeCapacity("g", 0.3, "cuda"), NodeCapacity("c", 0.1)], 32)
    assert [(p.node_id, p.device) for p in plan] == [("g", "cuda"), ("c", "cpu")]
    with pytest.raises(ValueError, match="VRAM"):
        plan_split(m["specs"], m["input_shape"], [NodeCapacity("g", 0.05, "cuda"), NodeCapacity("c", 0.05)], 32)


def test_pick_device_cpu_and_bad_choice():
    from agent.gpu import gpu_stats, pick_device

    assert pick_device("cpu") == "cpu" and gpu_stats("cpu") == {}
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            pick_device("cuda")


@needs_cuda
def test_gpu_stats_respect_vram_cap():
    from agent.gpu import gpu_stats

    s = gpu_stats("cuda", vram_cap_gb=0.25)
    assert s["gpu_name"] and s["vram_total_gb"] > 0 and s["vram_free_gb"] <= 0.25


@needs_cuda
@pytest.mark.parametrize("devices", [["cuda", "cuda"], ["cuda", "cpu"], ["cpu", "cuda"]])
async def test_gpu_and_mixed_pipelines_match_single_process(devices):
    """Same losses whether a stage runs on the GPU or the CPU (GPU maths differs slightly)."""
    m = get_model("mnist_cnn")
    single = train_single_process(m["specs"], synthetic_batches(m["input_shape"], 10, 16, seed=1), 6, lr=0.01)
    split, _ = await train_split_locally(
        m["specs"], synthetic_batches(m["input_shape"], 10, 16, seed=1), 6,
        even_split(len(m["specs"]), 2), microbatches=2, lr=0.01, devices=devices,
    )
    assert split == pytest.approx(single, rel=1e-3, abs=1e-4)


@needs_cuda
async def test_agent_on_gpu_trains_via_coordinator(coordinator, run_agent):
    room = await coordinator.create_room()
    code, h = room["room_code"], {"X-Admin-Token": room["admin_token"]}
    run_agent(code, coordinator.url, node_id="gpu", device="cuda", vram_cap_gb=1.0)
    run_agent(code, coordinator.url, node_id="cpu", ram_cap_gb=0.5)
    await coordinator.wait_nodes(code, 2)
    gpu_node = next(n for n in (await coordinator.room(code))["nodes"] if n["node_id"] == "gpu")
    assert gpu_node["device"] == "cuda" and gpu_node["vram_free_gb"] <= 1.0

    cfg = {"model": "mnist_mlp", "dataset": "synthetic", "batch_size": 32, "epochs": 1, "steps_per_epoch": 20}
    job = (await coordinator.http.post(f"/api/rooms/{code}/jobs", json=cfg, headers=h)).json()
    assert {p["node_id"]: p["device"] for p in job["plan"]} == {"gpu": "cuda", "cpu": "cpu"}
    done = await coordinator.wait_job(code, job["job_id"])
    assert done["status"] == "completed", done
    detail = (await coordinator.http.get(f"/api/rooms/{code}/jobs/{job['job_id']}")).json()
    assert any(m["node_id"] == "gpu" and m.get("gpu_mem_mb") for m in detail["node_metrics"])
