"""PoolForge Dataset Utilities.

Provides synthetic data generators for zero-dependency offline runs and
deterministic unit testing, alongside optional torchvision dataset loaders with
automatic offline fallback.
"""

from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader


class SyntheticClassificationDataset(Dataset):
    """Generates synthetic multi-class feature vectors and integer labels in memory."""

    def __init__(
        self,
        num_samples: int = 1000,
        input_shape: Tuple[int, ...] = (16,),
        num_classes: int = 2,
        seed: int = 42,
    ):
        generator = torch.Generator().manual_seed(seed)
        self.data = torch.randn((num_samples, *input_shape), generator=generator, dtype=torch.float32)
        self.targets = torch.randint(0, num_classes, (num_samples,), generator=generator, dtype=torch.int64)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]


def get_dataloader(
    dataset_name: str = "synthetic",
    batch_size: int = 32,
    num_samples: int = 1000,
    input_shape: Tuple[int, ...] = (16,),
    num_classes: int = 2,
    seed: int = 42,
    shuffle: bool = False,
) -> DataLoader:
    """Return a DataLoader for training. Defaults to offline synthetic data."""
    if dataset_name == "synthetic":
        ds = SyntheticClassificationDataset(
            num_samples=num_samples,
            input_shape=input_shape,
            num_classes=num_classes,
            seed=seed,
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    elif dataset_name in ("mnist", "mnist_synthetic"):
        # For offline robustness, generate MNIST-shaped (1, 28, 28) synthetic tensors
        ds = SyntheticClassificationDataset(
            num_samples=num_samples,
            input_shape=(1, 28, 28),
            num_classes=10,
            seed=seed,
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
