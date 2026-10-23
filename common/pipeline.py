"""PoolForge Distributed Pipeline Parallelism Execution Engine.

Implements the forward/backward autograd splitting trick across N nodes:
- Stage 0 (first): computes out = stage(x), detaches, forwards out + labels to Stage 1.
- Stage 1..(N-2) (middle): receives activation, sets requires_grad_(), forwards out to next.
- Stage (N-1) (last): receives activation, computes loss, calls loss.backward(), sends grad backward.
- Middle stages receive grad, call out.backward(grad), and propagate grad backward.
- Each stage updates its own parameters with its own local optimizer.
- Supports micro-batching and stage checkpointing.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from common.models import build_layers
from common.protocol import serialize_tensor, deserialize_tensor
from common.transports import Transport


class PipelineStage:
    """Manages the execution of a contiguous layer slice on a worker node."""

    def __init__(
        self,
        stage_index: int,
        num_stages: int,
        layer_specs: List[Dict[str, Any]],
        start_layer: int,
        end_layer: int,
        transport: Transport,
        prev_node_id: Optional[str] = None,
        next_node_id: Optional[str] = None,
        learning_rate: float = 0.01,
        optimizer_type: str = "sgd",
        job_id: str = "job_default",
        checkpoint_dir: str = "./checkpoints",
    ):
        self.stage_index = stage_index
        self.num_stages = num_stages
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.transport = transport
        self.prev_node_id = prev_node_id
        self.next_node_id = next_node_id
        self.job_id = job_id
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Determine stage role
        self.is_first = (stage_index == 0)
        self.is_last = (stage_index == num_stages - 1)

        # Build ONLY assigned layers
        self.layers: nn.Sequential = build_layers(layer_specs, start_layer, end_layer)

        # Local optimizer
        params = list(self.layers.parameters())
        if optimizer_type.lower() == "adam":
            self.optimizer = torch.optim.Adam(params, lr=learning_rate) if params else None
        else:
            self.optimizer = torch.optim.SGD(params, lr=learning_rate) if params else None

        self.loss_fn = nn.CrossEntropyLoss() if self.is_last else None

        # Activation and gradient caches for autograd linking across microbatches
        # Key: microbatch_index -> cache
        self._cached_inputs: Dict[int, torch.Tensor] = {}
        self._cached_outputs: Dict[int, torch.Tensor] = {}
        self._cached_labels: Dict[int, torch.Tensor] = {}

    def zero_grad(self) -> None:
        """Clear gradients on local parameters."""
        if self.optimizer:
            self.optimizer.zero_grad()

    def step_optimizer(self) -> None:
        """Perform optimizer parameter update on local layers."""
        if self.optimizer:
            self.optimizer.step()

    async def forward_step(
        self,
        x: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        step: int = 0,
        microbatch: int = 0,
        timeout: float = 10.0,
    ) -> Optional[float]:
        """Execute forward pass for one microbatch.

        Returns loss float on the last stage; None on non-last stages.
        """
        if self.is_first:
            if x is None:
                raise ValueError("Stage 0 requires input tensor x")

            out = self.layers(x)
            self._cached_outputs[microbatch] = out

            if not self.is_last:
                # Forward activation to next node
                act_payload = serialize_tensor(
                    out,
                    kind="activation",
                    job_id=self.job_id,
                    step=step,
                    microbatch=microbatch,
                )
                await self.transport.send_tensor(self.next_node_id, act_payload)

                # Send labels along so only the last stage needs them
                if labels is not None:
                    label_payload = serialize_tensor(
                        labels,
                        kind="labels",
                        job_id=self.job_id,
                        step=step,
                        microbatch=microbatch,
                    )
                    await self.transport.send_tensor(self.next_node_id, label_payload)
            else:
                # Single-node pipeline case
                if labels is None:
                    raise ValueError("Single stage pipeline requires labels")
                loss = self.loss_fn(out, labels)
                return loss.item()

            return None

        else:
            # Middle or last stage: receive activation from previous node
            sender, act_bytes = await self.transport.receive_tensor(timeout=timeout)
            act_tensor, act_hdr = deserialize_tensor(act_bytes)

            # If middle stage forwarding labels to next, or last stage receiving labels
            received_labels = None
            if self.is_last or self.next_node_id:
                sender, lbl_bytes = await self.transport.receive_tensor(timeout=timeout)
                lbl_tensor, _ = deserialize_tensor(lbl_bytes)
                received_labels = lbl_tensor

            # Autograd hook: requires_grad_() allows gradients to flow back into this tensor
            inp = act_tensor.detach().clone().requires_grad_()
            self._cached_inputs[microbatch] = inp

            out = self.layers(inp)
            self._cached_outputs[microbatch] = out

            if self.is_last:
                self._cached_labels[microbatch] = received_labels
                loss = self.loss_fn(out, received_labels)
                return loss.item()
            else:
                # Middle stage: forward activation and labels to next node
                act_payload = serialize_tensor(
                    out,
                    kind="activation",
                    job_id=self.job_id,
                    step=step,
                    microbatch=microbatch,
                )
                await self.transport.send_tensor(self.next_node_id, act_payload)

                if received_labels is not None:
                    lbl_payload = serialize_tensor(
                        received_labels,
                        kind="labels",
                        job_id=self.job_id,
                        step=step,
                        microbatch=microbatch,
                    )
                    await self.transport.send_tensor(self.next_node_id, lbl_payload)
                return None

    async def backward_step(
        self,
        step: int = 0,
        microbatch: int = 0,
        loss_scale: float = 1.0,
        timeout: float = 10.0,
    ) -> None:
        """Execute backward pass for one microbatch.

        Propagates gradients sequentially from last stage back to first stage.
        """
        if self.is_last:
            out = self._cached_outputs.pop(microbatch)
            labels = self._cached_labels.pop(microbatch)
            inp = self._cached_inputs.pop(microbatch, None)

            # Scaled loss for microbatch gradient accumulation
            loss = self.loss_fn(out, labels) * loss_scale
            loss.backward()

            if not self.is_first and inp is not None and inp.grad is not None:
                # Send activation gradients back to previous stage
                grad_payload = serialize_tensor(
                    inp.grad,
                    kind="gradient",
                    job_id=self.job_id,
                    step=step,
                    microbatch=microbatch,
                )
                await self.transport.send_tensor(self.prev_node_id, grad_payload)

        else:
            # Middle or first stage: receive incoming gradient from downstream stage
            sender, grad_bytes = await self.transport.receive_tensor(timeout=timeout)
            grad_tensor, _ = deserialize_tensor(grad_bytes)

            out = self._cached_outputs.pop(microbatch)
            out.backward(grad_tensor)

            if not self.is_first:
                # Middle stage: send upstream gradient to previous stage
                inp = self._cached_inputs.pop(microbatch)
                if inp.grad is not None:
                    grad_payload = serialize_tensor(
                        inp.grad,
                        kind="gradient",
                        job_id=self.job_id,
                        step=step,
                        microbatch=microbatch,
                    )
                    await self.transport.send_tensor(self.prev_node_id, grad_payload)

    def save_checkpoint(self, step: int) -> str:
        """Save local stage parameters and optimizer state to disk."""
        filename = f"{self.job_id}_stage_{self.stage_index}_step_{step}.pt"
        filepath = self.checkpoint_dir / filename
        state = {
            "step": step,
            "stage_index": self.stage_index,
            "num_stages": self.num_stages,
            "start_layer": self.start_layer,
            "end_layer": self.end_layer,
            "model_state": self.layers.state_dict(),
            "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
        }
        torch.save(state, str(filepath))
        return str(filepath)

    def load_checkpoint(self, filepath: str) -> int:
        """Load stage parameters from a checkpoint file and return the saved step."""
        state = torch.load(filepath, map_location="cpu")
        self.layers.load_state_dict(state["model_state"])
        if self.optimizer and state.get("optimizer_state"):
            self.optimizer.load_state_dict(state["optimizer_state"])
        return state.get("step", 0)
