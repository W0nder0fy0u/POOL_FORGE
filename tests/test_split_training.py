"""Phase 2: model specs, partial building, memory estimates and the key
correctness proof: split (pipeline) training gives the same losses as
ordinary single-process training."""

import pytest
import torch

from common.datasets import synthetic_batches
from common.model_zoo import catalog, get_model
from common.models import build_layers
from common.pipeline import PipelineStage, even_split, train_single_process, train_split_locally
from common.specs import count_params, estimate_memory, infer_shapes, validate_specs
from common.transports import LocalHub, LocalTransport


def test_shape_inference_matches_real_layers():
    for name in ("tiny", "mnist_mlp", "mnist_cnn"):
        m = get_model(name)
        model = build_layers(m["specs"], 0, len(m["specs"]))
        x = torch.zeros(2, *m["input_shape"])
        shapes = infer_shapes(m["specs"], tuple(m["input_shape"]))
        for layer, shape in zip(model, shapes):
            x = layer(x)
            assert tuple(x.shape[1:]) == shape, name


def test_param_count_matches_torch():
    for name in ("tiny", "mnist_cnn"):
        specs = get_model(name)["specs"]
        model = build_layers(specs, 0, len(specs))
        assert sum(count_params(s) for s in specs) == sum(p.numel() for p in model.parameters())


def test_build_layers_builds_only_the_range():
    specs = get_model("large_mlp", width=64, depth=4)["specs"]
    part = build_layers(specs, 3, 6)
    assert len(part) == 3
    full = build_layers(specs, 0, len(specs))
    # Same seed per layer -> identical weights whether built alone or in the full model.
    assert torch.equal(part[0].weight, full[3].weight)


def test_memory_formula():
    specs = [{"type": "linear", "in": 10, "out": 5}]
    # params = 55 -> 55*4*4 = 880 bytes, activation = 5 * batch 8 * 4 = 160 bytes
    assert estimate_memory(specs, (10,), batch_size=8) == [1040]


def test_invalid_specs_rejected():
    with pytest.raises(ValueError):
        validate_specs([{"type": "linear", "in": 3, "out": 4}], (5,))  # shape mismatch
    with pytest.raises(ValueError):
        validate_specs([{"type": "os.system"}], (5,))
    with pytest.raises(ValueError):
        get_model("large_mlp", width=10**6)


def test_catalog_lists_models():
    assert {m["name"] for m in catalog()} >= {"tiny", "mnist_cnn", "large_mlp"}


@pytest.mark.parametrize("model,stages,microbatches,optimizer", [
    ("tiny", 2, 1, "sgd"),
    ("tiny", 3, 1, "adam"),
    ("tiny", 2, 4, "adam"),
    ("mnist_cnn", 2, 2, "adam"),
    ("mnist_mlp", 4, 1, "sgd"),
])
async def test_split_loss_equals_single_process(model, stages, microbatches, optimizer):
    """THE correctness test: same seed + same data -> same losses (within float tolerance)."""
    m = get_model(model)
    steps, bs = 8, 16
    single = train_single_process(
        m["specs"], synthetic_batches(m["input_shape"], m["num_classes"], bs, seed=1), steps, lr=0.01, optimizer=optimizer
    )
    split, _ = await train_split_locally(
        m["specs"], synthetic_batches(m["input_shape"], m["num_classes"], bs, seed=1), steps,
        even_split(len(m["specs"]), stages), microbatches=microbatches, lr=0.01, optimizer=optimizer,
    )
    assert split == pytest.approx(single, rel=1e-4, abs=1e-5)


async def test_split_training_actually_learns():
    m = get_model("tiny")
    losses, _ = await train_split_locally(
        m["specs"], synthetic_batches(m["input_shape"], 2, 32, seed=3), 80, even_split(5, 2), lr=0.01
    )
    assert sum(losses[-10:]) / 10 < 0.8 * sum(losses[:10]) / 10


async def test_checkpoint_round_trip(tmp_path):
    specs = get_model("tiny")["specs"]
    hub = LocalHub()
    t = LocalTransport("only", hub)
    await t.start()
    st = PipelineStage(specs, 0, 5, 0, 1, t, None, None, "job_ck", checkpoint_dir=str(tmp_path))
    batches = synthetic_batches((16,), 2, 8)
    for step in range(3):
        x, y = next(batches)
        await st.train_step(step, x, y)
    st.save_checkpoint(3)
    saved = [p.clone() for p in st.layers.parameters()]

    st2 = PipelineStage(specs, 0, 5, 0, 1, t, None, None, "job_ck", seed=99, checkpoint_dir=str(tmp_path))
    assert st2.load_checkpoint(3)
    assert all(torch.equal(a, b) for a, b in zip(saved, st2.layers.parameters()))
    assert not st2.load_checkpoint(42)  # missing -> False


# ---------------------------------------------------------------- scheduler
from coordinator.scheduler import NodeCapacity, PoolTooSmall, plan_split  # noqa: E402


def test_scheduler_contiguous_and_proportional():
    m = get_model("large_mlp", width=1024, depth=8)
    nodes = [NodeCapacity("small", 0.1), NodeCapacity("big", 0.3)]
    plan = plan_split(m["specs"], m["input_shape"], nodes, batch_size=32)
    assert [p.node_id for p in plan] == ["big", "small"]  # biggest first
    assert plan[0].start_layer == 0 and plan[-1].end_layer == len(m["specs"])
    assert plan[0].end_layer == plan[1].start_layer  # contiguous
    assert plan[0].estimated_mb > plan[1].estimated_mb  # proportional to RAM
    assert all(p.estimated_mb <= p.budget_mb for p in plan)


def test_scheduler_pool_too_small_and_pool_fits():
    m = get_model("large_mlp", width=1024, depth=8)  # ~8.4M params -> ~130 MB
    one = [NodeCapacity("a", 0.1)]
    with pytest.raises(PoolTooSmall, match="pool too small"):
        plan_split(m["specs"], m["input_shape"], one, 32)
    two = [NodeCapacity("a", 0.1), NodeCapacity("b", 0.1)]
    assert len(plan_split(m["specs"], m["input_shape"], two, 32)) == 2


def test_scheduler_never_more_stages_than_layers():
    m = get_model("tiny")
    nodes = [NodeCapacity(f"n{i}", 1.0) for i in range(8)]
    assert len(plan_split(m["specs"], m["input_shape"], nodes, 8)) == 5
