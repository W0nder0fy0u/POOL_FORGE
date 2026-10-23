"""Training data. Each agent creates its data locally (nothing big is sent).

Only the FIRST stage actually loads data; it sends the labels along with the
activations so the last stage can compute the loss.

  "synthetic" - random inputs labelled by a fixed random "teacher" network,
                so the task is learnable and the loss really goes down.
  "mnist"     - real MNIST via torchvision (downloaded to ./data). If
                torchvision is missing or there is no internet, we fall back
                to synthetic data of the same shape and log a warning.
"""

import logging
from typing import Iterator, Sequence, Tuple

import torch

logger = logging.getLogger("poolforge.data")

Batch = Tuple[torch.Tensor, torch.Tensor]


def synthetic_batches(
    input_shape: Sequence[int], num_classes: int, batch_size: int, seed: int = 0
) -> Iterator[Batch]:
    """Endless stream of (x, y). Same seed -> same batches (needed for tests)."""
    gen = torch.Generator().manual_seed(seed)
    dim = 1
    for d in input_shape:
        dim *= d
    teacher = torch.randn(dim, num_classes, generator=gen)  # fixed labelling rule
    while True:
        x = torch.randn((batch_size, *input_shape), generator=gen)
        y = (x.reshape(batch_size, -1) @ teacher).argmax(dim=1)
        yield x, y


def mnist_batches(batch_size: int, seed: int = 0, data_dir: str = "./data") -> Iterator[Batch]:
    """Endless shuffled MNIST batches (re-shuffled every epoch)."""
    from torchvision import datasets, transforms  # optional dependency

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    ds = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
    # Convert once to big tensors: much faster than a DataLoader for tiny images.
    xs = ds.data.unsqueeze(1).float().div(255.0).sub(0.1307).div(0.3081)
    ys = ds.targets.clone()
    gen = torch.Generator().manual_seed(seed)
    while True:
        order = torch.randperm(len(ys), generator=gen)
        for i in range(0, len(order) - batch_size + 1, batch_size):
            idx = order[i:i + batch_size]
            yield xs[idx], ys[idx]


def get_batches(
    dataset: str, input_shape: Sequence[int], num_classes: int, batch_size: int, seed: int = 0
) -> Tuple[Iterator[Batch], str]:
    """Return (batch iterator, name of the dataset actually used)."""
    if dataset == "mnist":
        if list(input_shape) != [1, 28, 28]:
            raise ValueError("MNIST models need input shape (1, 28, 28)")
        try:
            it = mnist_batches(batch_size, seed)
            first = next(it)  # triggers the download now, so errors happen here

            def chain():
                yield first
                yield from it

            return chain(), "mnist"
        except Exception as exc:  # no torchvision / offline / download failed
            logger.warning("MNIST unavailable (%s); using synthetic data of the same shape", exc)
            return synthetic_batches(input_shape, num_classes, batch_size, seed), "synthetic (mnist fallback)"
    if dataset == "synthetic":
        return synthetic_batches(input_shape, num_classes, batch_size, seed), "synthetic"
    raise ValueError(f"Unknown dataset '{dataset}'")
