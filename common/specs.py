"""Model layer specs: shape inference, parameter counts and memory estimates.

A model is an ordered list of JSON-friendly dicts, e.g.

    [{"type": "flatten"}, {"type": "linear", "in": 784, "out": 256}, {"type": "relu"}, ...]

This file is pure Python (no torch) so the coordinator can plan a split
without installing PyTorch. common/models.py turns specs into real layers.
"""

import math
from typing import Any, Dict, List, Sequence, Tuple

Shape = Tuple[int, ...]  # shape of ONE sample (batch dimension left out)

# Layers that keep the shape unchanged.
_SAME_SHAPE = {"relu", "gelu", "sigmoid", "tanh", "dropout", "batchnorm1d", "batchnorm2d"}
SUPPORTED_TYPES = _SAME_SHAPE | {"linear", "conv2d", "maxpool2d", "flatten"}

BYTES_PER_FLOAT = 4
# weights + gradients + Adam 1st moment + Adam 2nd moment = 4 copies of every parameter
PARAM_COPIES = 4


def _pair(v: Any) -> Tuple[int, int]:
    return (int(v[0]), int(v[1])) if isinstance(v, (list, tuple)) else (int(v), int(v))


def output_shape(spec: Dict[str, Any], in_shape: Shape) -> Shape:
    """Shape of one sample after passing through this layer (raises if it doesn't fit)."""
    t = str(spec.get("type", "")).lower()
    if t in _SAME_SHAPE:
        return in_shape
    if t == "linear":
        if not in_shape or in_shape[-1] != spec["in"]:
            raise ValueError(f"linear expects last dim {spec['in']}, got shape {in_shape}")
        return in_shape[:-1] + (int(spec["out"]),)
    if t == "flatten":
        start = int(spec.get("start_dim", 1)) - 1  # -1 because batch dim is not in in_shape
        return in_shape[:start] + (math.prod(in_shape[start:]),)
    if t in ("conv2d", "maxpool2d"):
        if len(in_shape) != 3:
            raise ValueError(f"{t} expects (C, H, W), got {in_shape}")
        c, h, w = in_shape
        kh, kw = _pair(spec["kernel_size"])
        sh, sw = _pair(spec.get("stride", 1 if t == "conv2d" else spec["kernel_size"]))
        ph, pw = _pair(spec.get("padding", 0))
        oh = (h + 2 * ph - kh) // sh + 1
        ow = (w + 2 * pw - kw) // sw + 1
        if t == "conv2d":
            if c != spec["in_channels"]:
                raise ValueError(f"conv2d expects {spec['in_channels']} channels, got {c}")
            c = int(spec["out_channels"])
        return (c, oh, ow)
    raise ValueError(f"Unsupported layer type: {t!r}")


def infer_shapes(specs: Sequence[Dict[str, Any]], input_shape: Shape) -> List[Shape]:
    """Output shape after every layer. Also validates that the layers connect."""
    shapes, cur = [], tuple(input_shape)
    for i, spec in enumerate(specs):
        try:
            cur = output_shape(spec, cur)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Layer {i} ({spec.get('type')}): {exc}") from exc
        shapes.append(cur)
    return shapes


def count_params(spec: Dict[str, Any]) -> int:
    """Number of trainable parameters in one layer (without building it)."""
    t = str(spec.get("type", "")).lower()
    bias = bool(spec.get("bias", True))
    if t == "linear":
        return spec["in"] * spec["out"] + (spec["out"] if bias else 0)
    if t == "conv2d":
        kh, kw = _pair(spec["kernel_size"])
        return spec["in_channels"] * spec["out_channels"] * kh * kw + (spec["out_channels"] if bias else 0)
    if t in ("batchnorm1d", "batchnorm2d"):
        return 2 * spec["num_features"]
    return 0


def layer_memory_bytes(spec: Dict[str, Any], out_shape: Shape, batch_size: int) -> int:
    """Memory estimate for one layer (the formula from the project spec):

        params * 4 bytes * 4 (weights + grads + 2 Adam states)
      + activation size * batch * 4 bytes
    """
    return count_params(spec) * BYTES_PER_FLOAT * PARAM_COPIES + math.prod(out_shape) * batch_size * BYTES_PER_FLOAT


def estimate_memory(specs: Sequence[Dict[str, Any]], input_shape: Shape, batch_size: int) -> List[int]:
    """Estimated bytes for every layer of the model."""
    shapes = infer_shapes(specs, input_shape)
    return [layer_memory_bytes(s, sh, batch_size) for s, sh in zip(specs, shapes)]


def validate_specs(specs: Any, input_shape: Shape) -> None:
    """Reject anything that isn't a well-formed, connected list of supported layers."""
    if not isinstance(specs, list) or not specs:
        raise ValueError("Model must be a non-empty list of layer specs")
    if len(specs) > 1000:
        raise ValueError("Too many layers")
    for i, s in enumerate(specs):
        if not isinstance(s, dict) or str(s.get("type", "")).lower() not in SUPPORTED_TYPES:
            raise ValueError(f"Layer {i}: unsupported spec {s!r}")
    infer_shapes(specs, input_shape)
