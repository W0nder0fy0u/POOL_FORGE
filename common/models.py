"""PoolForge Model Specifications and Dynamic Layer Builder.

Enables JSON-serializable model architecture definitions and partial instantiation:
a node builds ONLY the subset of layers assigned to it, never instantiating the full model.
"""

from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn


def build_single_layer(spec: Dict[str, Any]) -> nn.Module:
    """Instantiate a single PyTorch layer from its JSON-serializable specification."""
    ltype = spec.get("type", "").lower()

    if ltype == "linear":
        in_features = spec["in"]
        out_features = spec["out"]
        bias = spec.get("bias", True)
        layer = nn.Linear(in_features, out_features, bias=bias)
        # Optional deterministic initialization if provided in spec
        if "init_weight" in spec:
            layer.weight.data.copy_(torch.tensor(spec["init_weight"]))
        if bias and "init_bias" in spec:
            layer.bias.data.copy_(torch.tensor(spec["init_bias"]))
        return layer

    elif ltype == "conv2d":
        in_channels = spec["in_channels"]
        out_channels = spec["out_channels"]
        kernel_size = spec["kernel_size"]
        stride = spec.get("stride", 1)
        padding = spec.get("padding", 0)
        bias = spec.get("bias", True)
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    elif ltype == "relu":
        return nn.ReLU()

    elif ltype == "gelu":
        return nn.GELU()

    elif ltype == "sigmoid":
        return nn.Sigmoid()

    elif ltype == "tanh":
        return nn.Tanh()

    elif ltype == "flatten":
        start_dim = spec.get("start_dim", 1)
        return nn.Flatten(start_dim=start_dim)

    elif ltype == "maxpool2d":
        kernel_size = spec["kernel_size"]
        stride = spec.get("stride", kernel_size)
        padding = spec.get("padding", 0)
        return nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    elif ltype == "dropout":
        p = spec.get("p", 0.5)
        return nn.Dropout(p=p)

    elif ltype == "batchnorm1d":
        num_features = spec["num_features"]
        return nn.BatchNorm1d(num_features=num_features)

    elif ltype == "batchnorm2d":
        num_features = spec["num_features"]
        return nn.BatchNorm2d(num_features=num_features)

    else:
        raise ValueError(f"Unsupported layer type: {ltype}")


def build_layers(specs: List[Dict[str, Any]], start: int, end: int) -> nn.Sequential:
    """Build only the contiguous slice [start, end) of layers specified.

    Args:
        specs: Complete list of layer specifications for the model.
        start: Inclusive start layer index.
        end: Exclusive end layer index.

    Returns:
        nn.Sequential module containing only the requested sub-range of layers.
    """
    if start < 0 or end > len(specs) or start >= end:
        raise ValueError(f"Invalid layer range [{start}, {end}) for {len(specs)} specs")

    sub_specs = specs[start:end]
    modules = [build_single_layer(s) for s in sub_specs]
    return nn.Sequential(*modules)


def count_layer_parameters(spec: Dict[str, Any]) -> int:
    """Calculate the number of learnable parameters for a layer spec without instantiating it."""
    ltype = spec.get("type", "").lower()
    bias = spec.get("bias", True)

    if ltype == "linear":
        in_f = spec["in"]
        out_f = spec["out"]
        return (in_f * out_f) + (out_f if bias else 0)

    elif ltype == "conv2d":
        in_c = spec["in_channels"]
        out_c = spec["out_channels"]
        k = spec["kernel_size"]
        if isinstance(k, (list, tuple)):
            k_w, k_h = k[0], k[1]
        else:
            k_w, k_h = k, k
        params = in_c * out_c * k_w * k_h
        if bias:
            params += out_c
        return params

    elif ltype in ("batchnorm1d", "batchnorm2d"):
        num_f = spec["num_features"]
        # Weight (gamma) and bias (beta)
        return 2 * num_f

    return 0


def estimate_layer_memory_bytes(
    spec: Dict[str, Any],
    batch_size: int,
    approx_activation_elements: int = 1024,
) -> int:
    """Estimate memory requirement for a single layer in bytes.

    Formula from specification:
        Memory per layer = params * 4 bytes * 4 (weights + grads + Adam states)
                           + activation size * batch * 4 bytes
    """
    params = count_layer_parameters(spec)
    # 4 bytes per float32 * 4 (weights + grads + Adam 1st moment + Adam 2nd moment)
    param_memory = params * 4 * 4

    # Activation memory in bytes
    activation_memory = approx_activation_elements * batch_size * 4

    return param_memory + activation_memory


def estimate_pipeline_memory(
    specs: List[Dict[str, Any]],
    batch_size: int = 32,
) -> List[int]:
    """Return a list of estimated memory footprints (in bytes) for each layer."""
    return [estimate_layer_memory_bytes(s, batch_size=batch_size) for s in specs]
