"""Phase 2 Tests: Model specs, model zoo, memory estimation, and Split Training Equivalence.

Acceptance criteria:
- build_layers instantiates only assigned ranges.
- Memory estimation accurately calculates weights + Adam states + activations.
- Distributed split-training loss matches single-process loss step-for-step (same seed).
- N-stage split training (3 stages: first, middle, last).
- Microbatching works correctly.
- Stage checkpoint save and restore.
"""

import asyncio
import copy
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from common.models import build_layers, estimate_pipeline_memory
from common.model_zoo import get_tiny_test_model, get_mnist_mlp, get_mnist_cnn, get_large_mlp
from common.datasets import get_dataloader
from common.transports import DirectTCPTransport
from common.pipeline import PipelineStage


def test_build_layers_partial_range():
    """Verify build_layers creates only the specified layer slice."""
    specs = get_tiny_test_model()  # 5 layers: Linear, ReLU, Linear, ReLU, Linear
    assert len(specs) == 5

    # Build only layers [0, 2) -> Linear(16, 32), ReLU
    stage0 = build_layers(specs, 0, 2)
    assert len(stage0) == 2
    assert isinstance(stage0[0], nn.Linear)
    assert isinstance(stage0[1], nn.ReLU)
    assert stage0[0].in_features == 16
    assert stage0[0].out_features == 32

    # Build layers [2, 5) -> Linear(32, 16), ReLU, Linear(16, 2)
    stage1 = build_layers(specs, 2, 5)
    assert len(stage1) == 3
    assert isinstance(stage1[0], nn.Linear)
    assert isinstance(stage1[1], nn.ReLU)
    assert isinstance(stage1[2], nn.Linear)
    assert stage1[2].out_features == 2


def test_memory_estimation():
    """Verify memory estimation formula follows specs: params*16 bytes + activations."""
    specs = [
        {"type": "linear", "in": 100, "out": 200, "bias": True},  # 100*200 + 200 = 20,200 params
    ]
    # params = 20,200
    # param_bytes = 20,200 * 4 * 4 = 323,200 bytes
    # approx activations = 1024 * 32 * 4 = 131,072 bytes
    # total = 454,272 bytes
    est = estimate_pipeline_memory(specs, batch_size=32)
    assert len(est) == 1
    assert est[0] == 323200 + (1024 * 32 * 4)


def test_large_mlp_generation():
    """Verify large MLP generator creates requested depth and width."""
    large_specs = get_large_mlp(in_features=512, hidden_width=1024, depth=4, num_classes=5)
    # 1 flatten + (depth * 2) = 1 + 8 = 9 layers
    assert len(large_specs) >= 8
    # Ensure parameter count is large
    est_bytes = sum(estimate_pipeline_memory(large_specs, batch_size=16))
    assert est_bytes > 10 * 1024 * 1024  # > 10 MB footprint


@pytest.mark.asyncio
async def test_split_training_loss_equivalence_2_stages():
    """CRITICAL TEST: Verify 2-stage split training produces the exact same loss as single-process training."""
    torch.manual_seed(1337)
    specs = get_tiny_test_model()

    # 1. Single process training
    torch.manual_seed(42)
    full_model = build_layers(specs, 0, len(specs))
    # Clone initial state dict before any gradient steps
    initial_state = copy.deepcopy(full_model.state_dict())

    single_optim = torch.optim.SGD(full_model.parameters(), lr=0.05)
    loss_fn = nn.CrossEntropyLoss()

    loader = get_dataloader(dataset_name="synthetic", batch_size=8, num_samples=32, input_shape=(16,), seed=999)
    batches = list(loader)

    single_losses = []
    for x_batch, y_batch in batches:
        single_optim.zero_grad()
        out = full_model(x_batch)
        loss = loss_fn(out, y_batch)
        loss.backward()
        single_optim.step()
        single_losses.append(loss.item())

    t0 = DirectTCPTransport(node_id="stage-0", host="127.0.0.1", port=0)
    t1 = DirectTCPTransport(node_id="stage-1", host="127.0.0.1", port=0)
    await t0.start()
    await t1.start()

    t0.register_peer("stage-1", "127.0.0.1", t1.listen_port)
    t1.register_peer("stage-0", "127.0.0.1", t0.listen_port)

    try:
        stage0 = PipelineStage(
            stage_index=0,
            num_stages=2,
            layer_specs=specs,
            start_layer=0,
            end_layer=2,
            transport=t0,
            next_node_id="stage-1",
            learning_rate=0.05,
            optimizer_type="sgd",
        )

        stage1 = PipelineStage(
            stage_index=1,
            num_stages=2,
            layer_specs=specs,
            start_layer=2,
            end_layer=5,
            transport=t1,
            prev_node_id="stage-0",
            learning_rate=0.05,
            optimizer_type="sgd",
        )

        # Copy the EXACT same initial weights from initial_state to stage0 and stage1
        with torch.no_grad():
            stage0.layers[0].weight.copy_(initial_state["0.weight"])
            stage0.layers[0].bias.copy_(initial_state["0.bias"])
            stage1.layers[0].weight.copy_(initial_state["2.weight"])
            stage1.layers[0].bias.copy_(initial_state["2.bias"])
            stage1.layers[2].weight.copy_(initial_state["4.weight"])
            stage1.layers[2].bias.copy_(initial_state["4.bias"])

        split_losses = []

        async def run_stage_0():
            for step_idx, (x_b, y_b) in enumerate(batches):
                stage0.zero_grad()
                await stage0.forward_step(x=x_b, labels=y_b, step=step_idx, microbatch=0)
                await stage0.backward_step(step=step_idx, microbatch=0)
                stage0.step_optimizer()

        async def run_stage_1():
            for step_idx, (x_b, y_b) in enumerate(batches):
                stage1.zero_grad()
                loss_val = await stage1.forward_step(step=step_idx, microbatch=0)
                split_losses.append(loss_val)
                await stage1.backward_step(step=step_idx, microbatch=0)
                stage1.step_optimizer()

        await asyncio.gather(run_stage_0(), run_stage_1())

        # Verify equivalence
        assert len(single_losses) == len(split_losses)
        for idx, (l_single, l_split) in enumerate(zip(single_losses, split_losses)):
            diff = abs(l_single - l_split)
            assert diff < 1e-5, f"Step {idx} loss mismatch: single={l_single}, split={l_split}, diff={diff}"

    finally:
        await t0.stop()
        await t1.stop()


@pytest.mark.asyncio
async def test_split_training_3_stages_with_middle_node():
    """Verify 3-stage split training (First -> Middle -> Last) trains and reduces loss."""
    specs = get_tiny_test_model()
    # Stage 0: [0, 2)
    # Stage 1: [2, 4)
    # Stage 2: [4, 5)

    t0 = DirectTCPTransport("s0", port=0)
    t1 = DirectTCPTransport("s1", port=0)
    t2 = DirectTCPTransport("s2", port=0)
    await t0.start()
    await t1.start()
    await t2.start()

    t0.register_peer("s1", "127.0.0.1", t1.listen_port)
    t1.register_peer("s0", "127.0.0.1", t0.listen_port)
    t1.register_peer("s2", "127.0.0.1", t2.listen_port)
    t2.register_peer("s1", "127.0.0.1", t1.listen_port)

    try:
        s0 = PipelineStage(0, 3, specs, 0, 2, t0, next_node_id="s1", learning_rate=0.05)
        s1 = PipelineStage(1, 3, specs, 2, 4, t1, prev_node_id="s0", next_node_id="s2", learning_rate=0.05)
        s2 = PipelineStage(2, 3, specs, 4, 5, t2, prev_node_id="s1", learning_rate=0.05)

        loader = get_dataloader("synthetic", batch_size=8, num_samples=32, input_shape=(16,))
        batches = list(loader)

        losses = []

        async def run_s0():
            for i, (xb, yb) in enumerate(batches):
                s0.zero_grad()
                await s0.forward_step(x=xb, labels=yb, step=i)
                await s0.backward_step(step=i)
                s0.step_optimizer()

        async def run_s1():
            for i in range(len(batches)):
                s1.zero_grad()
                await s1.forward_step(step=i)
                await s1.backward_step(step=i)
                s1.step_optimizer()

        async def run_s2():
            for i in range(len(batches)):
                s2.zero_grad()
                loss = await s2.forward_step(step=i)
                losses.append(loss)
                await s2.backward_step(step=i)
                s2.step_optimizer()

        await asyncio.gather(run_s0(), run_s1(), run_s2())
        assert len(losses) == len(batches)
        # Loss should decrease from initial step
        assert losses[-1] < losses[0]

    finally:
        await t0.stop()
        await t1.stop()
        await t2.stop()


@pytest.mark.asyncio
async def test_stage_checkpoint_save_and_restore(tmp_path):
    """Verify that stage weights and optimizer states checkpoint and restore properly."""
    specs = get_tiny_test_model()
    t = DirectTCPTransport("node-ckpt", port=0)
    await t.start()
    try:
        stage = PipelineStage(
            stage_index=0,
            num_stages=1,
            layer_specs=specs,
            start_layer=0,
            end_layer=2,
            transport=t,
            checkpoint_dir=str(tmp_path),
        )

        # Modify weight
        with torch.no_grad():
            stage.layers[0].weight.fill_(1.234)

        ckpt_file = stage.save_checkpoint(step=10)
        assert Path(ckpt_file).exists()

        # Reset weight
        with torch.no_grad():
            stage.layers[0].weight.fill_(0.0)

        # Restore from checkpoint
        restored_step = stage.load_checkpoint(ckpt_file)
        assert restored_step == 10
        assert torch.allclose(stage.layers[0].weight, torch.tensor(1.234))
    finally:
        await t.stop()


@pytest.mark.asyncio
async def test_microbatching_training_2_stages():
    """Verify distributed split training with microbatching (microbatches=2) accumulates gradients and converges."""
    torch.manual_seed(42)
    specs = get_tiny_test_model()
    t0 = DirectTCPTransport(node_id="mb-s0", port=0)
    t1 = DirectTCPTransport(node_id="mb-s1", port=0)
    await t0.start()
    await t1.start()

    t0.register_peer("mb-s1", "127.0.0.1", t1.listen_port)
    t1.register_peer("mb-s0", "127.0.0.1", t0.listen_port)

    try:
        stage0 = PipelineStage(0, 2, specs, 0, 2, t0, next_node_id="mb-s1", learning_rate=0.05)
        stage1 = PipelineStage(1, 2, specs, 2, 5, t1, prev_node_id="mb-s0", learning_rate=0.05)

        # Batch of 16, split into 2 microbatches of 8 over 8 steps
        loader = get_dataloader("synthetic", batch_size=16, num_samples=128, input_shape=(16,), seed=42)
        batches = list(loader)

        num_microbatches = 2
        losses = []

        async def run_stage_0():
            for step_idx, (x_b, y_b) in enumerate(batches):
                stage0.zero_grad()
                m_inputs = torch.chunk(x_b, num_microbatches)
                m_labels = torch.chunk(y_b, num_microbatches)

                # Forward microbatches
                for m_idx in range(num_microbatches):
                    await stage0.forward_step(
                        x=m_inputs[m_idx],
                        labels=m_labels[m_idx],
                        step=step_idx,
                        microbatch=m_idx,
                    )

                # Backward microbatches
                for m_idx in range(num_microbatches):
                    await stage0.backward_step(step=step_idx, microbatch=m_idx)

                stage0.step_optimizer()

        async def run_stage_1():
            for step_idx in range(len(batches)):
                stage1.zero_grad()
                step_loss = 0.0

                # Forward microbatches
                for m_idx in range(num_microbatches):
                    l_val = await stage1.forward_step(step=step_idx, microbatch=m_idx)
                    step_loss += l_val / num_microbatches

                losses.append(step_loss)

                # Backward microbatches with scaled loss
                for m_idx in range(num_microbatches):
                    await stage1.backward_step(
                        step=step_idx,
                        microbatch=m_idx,
                        loss_scale=1.0 / num_microbatches,
                    )

                stage1.step_optimizer()

        await asyncio.gather(run_stage_0(), run_stage_1())
        assert len(losses) == len(batches)
        assert losses[-1] < losses[0]

    finally:
        await t0.stop()
        await t1.stop()
