"""Split planner: which node runs which layers.

Rules (from the project spec):
  * every node gets ONE contiguous range of layers (pipeline order),
  * ranges are sized in proportion to each node's free memory,
  * every range must fit its node, otherwise we fail with "pool too small".

"Free memory" is free VRAM for a node training on an NVIDIA/AMD GPU (cuda),
and free RAM otherwise (CPU, or Apple GPUs which share RAM with the CPU).

Algorithm (simple and explainable):
  1. Estimate memory per layer with common.specs.estimate_memory.
  2. Order nodes by free memory, biggest first (fewer layers on weak laptops).
  3. Walk the layers; node i keeps taking layers while it stays under its
     proportional share of the model (and under its RAM budget), always
     leaving at least one layer for each remaining node.
  4. Check every stage fits; if not, explain exactly which node is too small.

Nodes that would only get zero-parameter layers are not wasted on tiny
models: we never use more stages than the model has layers.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence

from common.specs import estimate_memory

GB = 1024 ** 3
MB = 1024 ** 2
# Keep 10% of each node's free memory spare for PyTorch itself and the OS / driver.
USABLE_FRACTION = 0.9


@dataclass
class NodeCapacity:
    node_id: str
    mem_free_gb: float  # free VRAM if device == "cuda", else free RAM
    device: str = "cpu"

    @property
    def memory(self) -> str:
        return "VRAM" if self.device == "cuda" else "RAM"


@dataclass
class StagePlan:
    node_id: str
    stage_index: int
    start_layer: int
    end_layer: int
    estimated_mb: float
    budget_mb: float
    device: str = "cpu"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PoolTooSmall(ValueError):
    """Raised when the model cannot fit into the pool's free RAM."""


def plan_split(
    specs: List[Dict[str, Any]],
    input_shape: Sequence[int],
    nodes: List[NodeCapacity],
    batch_size: int,
) -> List[StagePlan]:
    if not nodes:
        raise PoolTooSmall("pool too small: no nodes are connected")
    layer_mem = estimate_memory(specs, tuple(input_shape), batch_size)
    total = sum(layer_mem)

    # Biggest node first; never more stages than layers.
    nodes = sorted(nodes, key=lambda n: n.mem_free_gb, reverse=True)[: len(specs)]
    budgets = [n.mem_free_gb * GB * USABLE_FRACTION for n in nodes]
    if total > sum(budgets):
        raise PoolTooSmall(
            f"pool too small: model needs ~{total / MB:.0f} MB but the pool has only "
            f"{sum(budgets) / MB:.0f} MB usable ({len(nodes)} node(s); VRAM on GPU nodes, RAM on the rest). "
            "Add a laptop or shrink the model."
        )

    try:
        return _greedy(specs, nodes, layer_mem, budgets, proportional=True)
    except PoolTooSmall:
        # Proportional shares didn't work out (e.g. one huge layer at the end).
        # Second try: let each node take as many layers as its RAM allows.
        return _greedy(specs, nodes, layer_mem, budgets, proportional=False)


def _greedy(specs, nodes, layer_mem, budgets, proportional: bool) -> List[StagePlan]:
    total = sum(layer_mem)
    plans: List[StagePlan] = []
    start = 0
    for i, node in enumerate(nodes):
        remaining_nodes = len(nodes) - i - 1
        if remaining_nodes == 0:
            end = len(specs)  # last node takes everything left
        else:
            share = total * budgets[i] / sum(budgets) if proportional else float("inf")
            end, used = start, 0
            max_end = len(specs) - remaining_nodes  # leave >= 1 layer per later node
            while end < max_end:
                nxt = used + layer_mem[end]
                # Stop if the next layer would overflow the budget, or if we are
                # already at our proportional share (but always take >= 1 layer).
                if end > start and (nxt > budgets[i] or used >= share):
                    break
                used = nxt
                end += 1
        stage_mem = sum(layer_mem[start:end])
        if stage_mem > budgets[i]:
            raise PoolTooSmall(
                f"pool too small: node '{node.node_id}' has {budgets[i] / MB:.0f} MB {node.memory} usable but its "
                f"layers [{start}, {end}) need ~{stage_mem / MB:.0f} MB"
            )
        plans.append(StagePlan(node.node_id, i, start, end, round(stage_mem / MB, 2), round(budgets[i] / MB, 2), node.device))
        start = end
    return plans
