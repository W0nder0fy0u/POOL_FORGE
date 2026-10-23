"""Build real PyTorch layers from JSON layer specs.

The key function is build_layers(specs, start, end): it creates ONLY the
layers in [start, end). A node in the pool therefore never allocates memory
for layers that belong to other nodes.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


def build_single_layer(spec: Dict[str, Any]) -> nn.Module:
    """Create one nn.Module from its spec dict."""
    t = str(spec.get("type", "")).lower()
    bias = bool(spec.get("bias", True))
    if t == "linear":
        return nn.Linear(spec["in"], spec["out"], bias=bias)
    if t == "conv2d":
        return nn.Conv2d(
            spec["in_channels"],
            spec["out_channels"],
            kernel_size=spec["kernel_size"],
            stride=spec.get("stride", 1),
            padding=spec.get("padding", 0),
            bias=bias,
        )
    if t == "maxpool2d":
        k = spec["kernel_size"]
        return nn.MaxPool2d(k, stride=spec.get("stride", k), padding=spec.get("padding", 0))
    if t == "flatten":
        return nn.Flatten(start_dim=spec.get("start_dim", 1))
    if t == "relu":
        return nn.ReLU()
    if t == "gelu":
        return nn.GELU()
    if t == "sigmoid":
        return nn.Sigmoid()
    if t == "tanh":
        return nn.Tanh()
    if t == "dropout":
        return nn.Dropout(p=spec.get("p", 0.5))
    if t == "batchnorm1d":
        return nn.BatchNorm1d(spec["num_features"])
    if t == "batchnorm2d":
        return nn.BatchNorm2d(spec["num_features"])
    raise ValueError(f"Unsupported layer type: {t!r}")


def build_layers(
    specs: List[Dict[str, Any]], start: int, end: int, seed: Optional[int] = 0
) -> nn.Sequential:
    """Build only layers [start, end).

    If `seed` is given, layer i is initialised with torch seed (seed + i). So
    layer 5 gets exactly the same starting weights whether it is built alone on
    laptop B or as part of the full model on one machine. That is what makes
    the split-vs-single loss test possible.
    """
    if not (0 <= start < end <= len(specs)):
        raise ValueError(f"Invalid layer range [{start}, {end}) for {len(specs)} layers")
    modules = []
    for i in range(start, end):
        if seed is not None:
            torch.manual_seed(seed + i)
        modules.append(build_single_layer(specs[i]))
    return nn.Sequential(*modules)


def make_optimizer(params, name: str, lr: float) -> Optional[torch.optim.Optimizer]:
    """Each stage owns an optimizer for its own parameters only."""
    params = list(params)
    if not params:  # e.g. a stage made only of ReLU/Flatten layers
        return None
    if name.lower() == "adam":
        return torch.optim.Adam(params, lr=lr)
    return torch.optim.SGD(params, lr=lr, momentum=0.9)
