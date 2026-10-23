"""PoolForge Model Zoo.

Predefined model architectures for testing, demonstration, and capacity benchmarks:
1. "tiny_test": Fast 4-layer network for deterministic unit testing.
2. "mnist_cnn": Classic Convolutional Neural Network for digit classification.
3. "mnist_mlp": Multi-layer perceptron for MNIST.
4. "large_mlp": Parameter-heavy model with configurable width/depth to intentionally
   exceed a single node's RAM cap and prove pool scaling.
"""

from typing import Any, Dict, List


def get_tiny_test_model() -> List[Dict[str, Any]]:
    """Return spec for a tiny model for fast unit testing."""
    return [
        {"type": "linear", "in": 16, "out": 32, "bias": True},
        {"type": "relu"},
        {"type": "linear", "in": 32, "out": 16, "bias": True},
        {"type": "relu"},
        {"type": "linear", "in": 16, "out": 2, "bias": True},
    ]


def get_mnist_mlp() -> List[Dict[str, Any]]:
    """Return spec for an MLP suitable for MNIST classification."""
    return [
        {"type": "flatten", "start_dim": 1},
        {"type": "linear", "in": 784, "out": 256, "bias": True},
        {"type": "relu"},
        {"type": "linear", "in": 256, "out": 128, "bias": True},
        {"type": "relu"},
        {"type": "linear", "in": 128, "out": 10, "bias": True},
    ]


def get_mnist_cnn() -> List[Dict[str, Any]]:
    """Return spec for a standard CNN suitable for MNIST classification."""
    return [
        {"type": "conv2d", "in_channels": 1, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1},
        {"type": "relu"},
        {"type": "maxpool2d", "kernel_size": 2, "stride": 2},
        {"type": "conv2d", "in_channels": 16, "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1},
        {"type": "relu"},
        {"type": "maxpool2d", "kernel_size": 2, "stride": 2},
        {"type": "flatten", "start_dim": 1},
        {"type": "linear", "in": 32 * 7 * 7, "out": 128, "bias": True},
        {"type": "relu"},
        {"type": "linear", "in": 128, "out": 10, "bias": True},
    ]


def get_large_mlp(
    in_features: int = 1024,
    hidden_width: int = 4096,
    depth: int = 6,
    num_classes: int = 10,
) -> List[Dict[str, Any]]:
    """Generate a deliberately large MLP to exceed single-machine memory caps.

    Args:
        in_features: Input vector dimensionality.
        hidden_width: Number of neurons per hidden layer (e.g. 4096).
        depth: Number of hidden layers.
        num_classes: Output classification logits.
    """
    layers: List[Dict[str, Any]] = [
        {"type": "flatten", "start_dim": 1},
        {"type": "linear", "in": in_features, "out": hidden_width, "bias": True},
        {"type": "relu"},
    ]

    for _ in range(depth - 1):
        layers.append({"type": "linear", "in": hidden_width, "out": hidden_width, "bias": True})
        layers.append({"type": "relu"})

    layers.append({"type": "linear", "in": hidden_width, "out": num_classes, "bias": True})
    return layers


MODEL_CATALOG = {
    "tiny_test": get_tiny_test_model,
    "mnist_mlp": get_mnist_mlp,
    "mnist_cnn": get_mnist_cnn,
    "large_mlp": get_large_mlp,
}


def get_model_specs(model_name: str, **kwargs) -> List[Dict[str, Any]]:
    """Retrieve layer specifications for a model by name from the catalog."""
    factory = MODEL_CATALOG.get(model_name)
    if not factory:
        raise KeyError(f"Model '{model_name}' not found. Available models: {list(MODEL_CATALOG.keys())}")
    return factory(**kwargs)
