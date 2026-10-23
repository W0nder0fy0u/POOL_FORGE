"""Model zoo: ready-made models as layer-spec lists (pure Python, no torch).

    tiny       - 5 layers, 16 inputs, 2 classes. Fast; used by tests.
    mnist_mlp  - small MLP for MNIST.
    mnist_cnn  - small CNN for MNIST.
    large_mlp  - deliberately big MLP on MNIST; width/depth configurable so
                 it can be made too big for one RAM-capped node.
"""

from typing import Any, Dict, List


def tiny() -> List[Dict[str, Any]]:
    return [
        {"type": "linear", "in": 16, "out": 32},
        {"type": "relu"},
        {"type": "linear", "in": 32, "out": 16},
        {"type": "relu"},
        {"type": "linear", "in": 16, "out": 2},
    ]


def mnist_mlp() -> List[Dict[str, Any]]:
    return [
        {"type": "flatten"},
        {"type": "linear", "in": 784, "out": 256},
        {"type": "relu"},
        {"type": "linear", "in": 256, "out": 128},
        {"type": "relu"},
        {"type": "linear", "in": 128, "out": 10},
    ]


def mnist_cnn() -> List[Dict[str, Any]]:
    return [
        {"type": "conv2d", "in_channels": 1, "out_channels": 16, "kernel_size": 3, "padding": 1},
        {"type": "relu"},
        {"type": "maxpool2d", "kernel_size": 2},
        {"type": "conv2d", "in_channels": 16, "out_channels": 32, "kernel_size": 3, "padding": 1},
        {"type": "relu"},
        {"type": "maxpool2d", "kernel_size": 2},
        {"type": "flatten"},
        {"type": "linear", "in": 32 * 7 * 7, "out": 128},
        {"type": "relu"},
        {"type": "linear", "in": 128, "out": 10},
    ]


def large_mlp(width: int = 2048, depth: int = 8) -> List[Dict[str, Any]]:
    """MNIST MLP with `depth` hidden layers of `width` neurons.

    Rough size: depth * width^2 parameters. width=2048, depth=8 is ~30M
    parameters, i.e. ~0.5 GB with Adam (16 bytes per parameter).
    """
    layers: List[Dict[str, Any]] = [{"type": "flatten"}, {"type": "linear", "in": 784, "out": width}, {"type": "relu"}]
    for _ in range(depth - 1):
        layers += [{"type": "linear", "in": width, "out": width}, {"type": "relu"}]
    layers.append({"type": "linear", "in": width, "out": 10})
    return layers


# name -> (builder, input shape of one sample, number of classes, default dataset, description)
MODELS = {
    "tiny": (tiny, (16,), 2, "synthetic", "Tiny 5-layer MLP (tests, instant)"),
    "mnist_mlp": (mnist_mlp, (1, 28, 28), 10, "mnist", "MNIST MLP, 235K params"),
    "mnist_cnn": (mnist_cnn, (1, 28, 28), 10, "mnist", "MNIST CNN, 206K params"),
    "large_mlp": (large_mlp, (1, 28, 28), 10, "mnist", "Big MLP, configurable width/depth"),
}


def get_model(name: str, **options: Any) -> Dict[str, Any]:
    """Return {"name", "specs", "input_shape", "num_classes", "dataset"} for a model."""
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODELS)}")
    builder, input_shape, num_classes, dataset, _ = MODELS[name]
    if name == "large_mlp":
        width = int(options.get("width", 2048))
        depth = int(options.get("depth", 8))
        if not (8 <= width <= 16384 and 1 <= depth <= 64):
            raise ValueError("large_mlp needs 8 <= width <= 16384 and 1 <= depth <= 64")
        specs = builder(width=width, depth=depth)
    else:
        specs = builder()
    return {
        "name": name,
        "specs": specs,
        "input_shape": list(input_shape),
        "num_classes": num_classes,
        "dataset": dataset,
    }


def catalog() -> List[Dict[str, Any]]:
    """Model list for the dashboard's model picker."""
    return [
        {"name": n, "description": m[4], "input_shape": list(m[1]), "num_classes": m[2], "dataset": m[3]}
        for n, m in MODELS.items()
    ]
