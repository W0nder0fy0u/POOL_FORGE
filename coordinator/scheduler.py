"""PoolForge Pipeline Parallelism Scheduler.

Assigns CONTIGUOUS layer ranges proportional to each worker node's reported free RAM cap.
Validates memory boundaries and fails early with "pool too small" if the model cannot fit.
"""

from typing import Any, Dict, List
from common.models import estimate_pipeline_memory
from common.schemas import NodeTelemetry, SplitPlanStage


def schedule_pipeline_split(
    layer_specs: List[Dict[str, Any]],
    nodes: List[NodeTelemetry],
    batch_size: int = 16,
) -> List[SplitPlanStage]:
    """Partition model layer specifications across available nodes.

    Args:
        layer_specs: Complete ordered list of model layers.
        nodes: Available healthy worker nodes in the pool.
        batch_size: Training batch size used for activation sizing.

    Returns:
        List of SplitPlanStage allocations with contiguous layer ranges [start, end).

    Raises:
        ValueError: If pool has 0 nodes, or if total pool RAM is insufficient ("pool too small").
    """
    if not nodes:
        raise ValueError("Cannot schedule: pool has 0 available worker nodes")

    num_layers = len(layer_specs)
    if num_layers == 0:
        raise ValueError("Cannot schedule: model has 0 layers")

    # Limit active stages to min(len(nodes), num_layers)
    active_nodes = nodes[:num_layers]
    num_stages = len(active_nodes)

    # 1. Estimate memory footprint per layer in bytes
    layer_mem_bytes = estimate_pipeline_memory(layer_specs, batch_size=batch_size)
    total_model_mem_bytes = sum(layer_mem_bytes)
    total_model_mem_gb = total_model_mem_bytes / (1024 ** 3)

    # 2. Check total pool capacity
    node_capacities_bytes = [int(n.ram_free_gb * (1024 ** 3)) for n in active_nodes]
    total_pool_bytes = sum(node_capacities_bytes)

    if total_pool_bytes < total_model_mem_bytes:
        raise ValueError(
            f"pool too small: model requires {total_model_mem_gb:.3f} GB, "
            f"but pool only has {total_pool_bytes / (1024 ** 3):.3f} GB free"
        )

    # 3. Allocate contiguous slices proportional to node RAM capacities
    # Use greedy prefix-sum partitioning to match target fractions
    allocations: List[SplitPlanStage] = []
    current_start = 0

    for stage_idx in range(num_stages):
        is_last_stage = (stage_idx == num_stages - 1)
        node = active_nodes[stage_idx]
        node_cap_bytes = node_capacities_bytes[stage_idx]

        if is_last_stage:
            # Last node takes all remaining layers
            assigned_end = num_layers
        else:
            # Target memory for this node based on its fraction of total remaining capacity
            remaining_pool_bytes = sum(node_capacities_bytes[stage_idx:])
            fraction = node_cap_bytes / max(remaining_pool_bytes, 1)

            remaining_layers = num_layers - current_start
            remaining_stages = num_stages - stage_idx

            # Each stage must receive at least 1 layer
            min_layers_needed_for_future = remaining_stages - 1

            # Accumulate layers until fraction of remaining memory is met
            accumulated_mem = 0
            candidate_end = current_start + 1

            target_mem_this_stage = fraction * sum(layer_mem_bytes[current_start:])

            for idx in range(current_start, num_layers - min_layers_needed_for_future):
                accumulated_mem += layer_mem_bytes[idx]
                if accumulated_mem >= target_mem_this_stage:
                    candidate_end = idx + 1
                    break
                candidate_end = idx + 1

            # Ensure we do not starve future stages
            max_allowed_end = num_layers - min_layers_needed_for_future
            assigned_end = min(candidate_end, max_allowed_end)
            assigned_end = max(assigned_end, current_start + 1)

        # Validate memory fit for this stage
        stage_mem = sum(layer_mem_bytes[current_start:assigned_end])
        if stage_mem > node_cap_bytes:
            raise ValueError(
                f"pool too small: node '{node.node_id}' capped at {node.ram_free_gb:.3f} GB, "
                f"cannot fit assigned partition [{current_start}, {assigned_end}) requiring {stage_mem / (1024 ** 3):.3f} GB"
            )

        allocations.append(
            SplitPlanStage(
                node_id=node.node_id,
                stage_index=stage_idx,
                start_layer=current_start,
                end_layer=assigned_end,
                num_layers=assigned_end - current_start,
                estimated_ram_mb=round(stage_mem / (1024 * 1024), 2),
                node_ram_cap_gb=node.ram_free_gb,
            )
        )
        current_start = assigned_end

    return allocations
