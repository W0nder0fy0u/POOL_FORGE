"""Pipeline-parallel training engine (the core of PoolForge).

The model's layers are cut into contiguous stages; stage i runs on node i.
One training step with M micro-batches works like this (GPipe schedule):

  FORWARD, for each micro-batch m:
    first stage : out = layers(x_m)          -> send out.detach() + labels to next
    middle stage: inp = recv().requires_grad_(); out = layers(inp) -> send to next
    last stage  : inp = recv().requires_grad_(); loss = CE(layers(inp), labels)

  BACKWARD, for each micro-batch m:
    last stage  : loss.backward()             -> send inp.grad to previous
    middle stage: g = recv(); out.backward(g) -> send inp.grad to previous
    first stage : g = recv(); out.backward(g)

  Finally every stage calls optimizer.step() on its OWN parameters.

The "autograd trick": `detach()` cuts the graph at the network boundary and
`requires_grad_()` on the received tensor starts a new graph, so after
backward() the receiver's `inp.grad` is exactly dLoss/d(activation), which is
what the previous stage needs to continue backprop with out.backward(grad).

With M > 1, stage 0 can start micro-batch 1 while stage 1 still works on
micro-batch 0, which hides some network latency (when stages run on
different machines/processes).
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

from common.models import build_layers, make_optimizer
from common.protocol import deserialize_tensor, serialize_tensor
from common.transports import LocalHub, LocalTransport, Transport

logger = logging.getLogger("poolforge.pipeline")


class Mailbox:
    """Receives frames and hands them out by (kind, step, microbatch).

    Frames may arrive in any order (e.g. labels before activations, or a
    gradient for micro-batch 1 before micro-batch 0). Frames are stored until
    someone asks for exactly that key. Frames of other jobs are ignored.
    """

    def __init__(self, transport: Transport, job_id: str, decrypt_key: Optional[bytes] = None):
        self.transport = transport
        self.job_id = job_id
        self.decrypt_key = decrypt_key
        self._stash: Dict[Tuple[str, int, int], torch.Tensor] = {}

    async def get(self, kind: str, step: int, mb: int) -> torch.Tensor:
        key = (kind, step, mb)
        while key not in self._stash:
            _sender, frame = await self.transport.recv()
            try:
                tensor, hdr = deserialize_tensor(frame, self.decrypt_key)
            except Exception as exc:  # corrupt / wrong key: log and ignore the frame
                logger.warning("Dropping bad frame: %s", exc)
                continue
            if hdr["job_id"] != self.job_id:
                continue  # leftover from an older job
            self._stash[(hdr["kind"], hdr["step"], hdr["microbatch"])] = tensor
        return self._stash.pop(key)

    def forget_before(self, step: int) -> None:
        """Drop stashed frames of old steps (after a resume from checkpoint)."""
        for k in [k for k in self._stash if k[1] < step]:
            del self._stash[k]


class PipelineStage:
    """One node's slice of the model plus its optimizer and network links."""

    def __init__(
        self,
        specs: List[Dict[str, Any]],
        start_layer: int,
        end_layer: int,
        stage_index: int,
        num_stages: int,
        transport: Transport,
        prev_node: Optional[str],
        next_node: Optional[str],
        job_id: str,
        lr: float = 1e-3,
        optimizer: str = "adam",
        seed: int = 0,
        encrypt_key: Optional[bytes] = None,
        checkpoint_dir: str = "./checkpoints",
        attempt: int = 0,
    ):
        self.stage_index = stage_index
        self.num_stages = num_stages
        self.is_first = stage_index == 0
        self.is_last = stage_index == num_stages - 1
        self.start_layer, self.end_layer = start_layer, end_layer
        self.transport = transport
        self.prev_node, self.next_node = prev_node, next_node
        self.job_id = job_id
        # Frames are tagged "<job>#<attempt>". After a node drop the job restarts
        # with attempt+1, so frames still in flight from the old attempt are ignored.
        self.frame_tag = f"{job_id}#{attempt}"
        self.key = encrypt_key
        self.mailbox = Mailbox(transport, self.frame_tag, encrypt_key)
        self.checkpoint_dir = Path(checkpoint_dir) / job_id

        # Build ONLY this stage's layers. The full model never exists on one node.
        self.layers: nn.Sequential = build_layers(specs, start_layer, end_layer, seed=seed)
        self.optimizer = make_optimizer(self.layers.parameters(), optimizer, lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def num_params(self) -> int:
        return sum(p.numel() for p in self.layers.parameters())

    async def _send(self, target: str, tensor: torch.Tensor, kind: str, step: int, mb: int) -> None:
        frame = serialize_tensor(tensor, kind=kind, job_id=self.frame_tag, step=step, microbatch=mb, encrypt_key=self.key)
        await self.transport.send(target, frame)

    async def train_step(
        self,
        step: int,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        microbatches: int = 1,
    ) -> Optional[float]:
        """Run one full training step. Returns the loss on the last stage, else None.

        Heavy maths runs via asyncio.to_thread so the event loop stays free to
        answer heartbeats and receive frames while a big layer is computing.
        """
        run = asyncio.to_thread
        self.layers.train()
        if self.optimizer:
            self.optimizer.zero_grad()

        xs = torch.chunk(x, microbatches) if self.is_first else [None] * microbatches
        ys = torch.chunk(y, microbatches) if self.is_first else [None] * microbatches
        if self.is_first and len(xs) != microbatches:
            raise ValueError("batch_size must be >= microbatches")

        saved: List[Tuple[Optional[torch.Tensor], torch.Tensor]] = []  # (input, output-or-loss)
        total_loss = 0.0

        # ---------------- forward pass for every micro-batch ----------------
        for m in range(microbatches):
            if self.is_first:
                inp, labels = None, ys[m]
                out = await run(self.layers, xs[m])
            else:
                inp = (await self.mailbox.get("activation", step, m)).requires_grad_()
                labels = await self.mailbox.get("labels", step, m)
                out = await run(self.layers, inp)

            if self.is_last:
                # Divide by M so the summed gradients equal those of the full batch.
                loss = self.loss_fn(out, labels) / microbatches
                total_loss += loss.item()
                saved.append((inp, loss))
            else:
                await self._send(self.next_node, out.detach(), "activation", step, m)
                await self._send(self.next_node, labels, "labels", step, m)
                saved.append((inp, out))

        # ---------------- backward pass for every micro-batch ---------------
        for m in range(microbatches):
            inp, out = saved[m]
            if self.is_last:
                await run(out.backward)  # `out` is the loss here
            else:
                grad = await self.mailbox.get("gradient", step, m)
                if out.requires_grad:  # False only for a first stage with no weights
                    await run(out.backward, grad)
            if not self.is_first:
                await self._send(self.prev_node, inp.grad, "gradient", step, m)

        if self.optimizer:
            await run(self.optimizer.step)
        return total_loss if self.is_last else None

    # ------------------------------ checkpoints ------------------------------
    def checkpoint_path(self, step: int) -> Path:
        return self.checkpoint_dir / f"stage{self.stage_index}_step{step}.pt"

    def save_checkpoint(self, step: int, keep: int = 3) -> Path:
        """Save weights + optimizer state after `step` completed steps."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_path(step)
        torch.save(
            {
                "step": step,
                "layers": [self.start_layer, self.end_layer],
                "model": self.layers.state_dict(),
                "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            },
            path,
        )
        # keep only the newest few files
        old = sorted(self.checkpoint_dir.glob(f"stage{self.stage_index}_step*.pt"), key=lambda p: p.stat().st_mtime)
        for p in old[:-keep]:
            p.unlink(missing_ok=True)
        return path

    def load_checkpoint(self, step: int) -> bool:
        """Load the checkpoint for `step` if it exists and matches our layers."""
        path = self.checkpoint_path(step)
        if not path.exists():
            return False
        # weights_only=True: only tensors/numbers are loaded, never arbitrary objects.
        state = torch.load(path, map_location="cpu", weights_only=True)
        if list(state["layers"]) != [self.start_layer, self.end_layer]:
            return False
        self.layers.load_state_dict(state["model"])
        if self.optimizer and state["optimizer"]:
            self.optimizer.load_state_dict(state["optimizer"])
        return True


# ---------------------------------------------------------------------------
# Reference implementations used by tests and benchmarks
# ---------------------------------------------------------------------------
def train_single_process(
    specs: List[Dict[str, Any]],
    batches: Iterator[Tuple[torch.Tensor, torch.Tensor]],
    steps: int,
    lr: float = 1e-3,
    optimizer: str = "adam",
    seed: int = 0,
) -> List[float]:
    """Ordinary training of the whole model in one process (the baseline)."""
    model = build_layers(specs, 0, len(specs), seed=seed)
    opt = make_optimizer(model.parameters(), optimizer, lr)
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    for _ in range(steps):
        x, y = next(batches)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def even_split(num_layers: int, num_stages: int) -> List[Tuple[int, int]]:
    """Cut num_layers into num_stages contiguous, roughly equal ranges."""
    bounds = [round(i * num_layers / num_stages) for i in range(num_stages + 1)]
    return [(bounds[i], bounds[i + 1]) for i in range(num_stages)]


async def train_split_locally(
    specs: List[Dict[str, Any]],
    batches: Iterator[Tuple[torch.Tensor, torch.Tensor]],
    steps: int,
    ranges: List[Tuple[int, int]],
    microbatches: int = 1,
    lr: float = 1e-3,
    optimizer: str = "adam",
    seed: int = 0,
    latency_ms: float = 0.0,
) -> Tuple[List[float], float]:
    """Run an N-stage pipeline inside one process over LocalTransport.

    Returns (losses, seconds). Used by the loss-equivalence test.
    """
    hub = LocalHub()
    n = len(ranges)
    ids = [f"stage{i}" for i in range(n)]
    stages = []
    for i, (s, e) in enumerate(ranges):
        t = LocalTransport(ids[i], hub, simulate_latency_ms=latency_ms)
        await t.start()
        stages.append(
            PipelineStage(
                specs, s, e, i, n, t,
                prev_node=ids[i - 1] if i > 0 else None,
                next_node=ids[i + 1] if i < n - 1 else None,
                job_id="local", lr=lr, optimizer=optimizer, seed=seed,
            )
        )
    losses = []
    t0 = time.perf_counter()
    for step in range(steps):
        x, y = next(batches)
        results = await asyncio.gather(
            *[st.train_step(step, x if st.is_first else None, y if st.is_first else None, microbatches) for st in stages]
        )
        losses.append(results[-1])
    return losses, time.perf_counter() - t0
