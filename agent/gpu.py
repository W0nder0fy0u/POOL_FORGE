"""Which device this laptop trains on, and how much GPU memory it can share.

    cuda : NVIDIA GPU (also AMD on Linux with the ROCm build of PyTorch)
    mps  : Apple Silicon GPU (shares memory with the CPU, so we plan with RAM)
    cpu  : no usable GPU

`--device auto` (the default) picks cuda, then mps, then cpu.
"""

import logging
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger("poolforge.gpu")
GB = 1024 ** 3


def pick_device(preference: str = "auto") -> str:
    cuda_ok = torch.cuda.is_available()
    mps_ok = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    if preference == "auto":
        return "cuda" if cuda_ok else "mps" if mps_ok else "cpu"
    if preference == "cuda" and not cuda_ok:
        raise RuntimeError("--device cuda: PyTorch cannot see a CUDA GPU (is the CUDA build of torch installed?)")
    if preference == "mps" and not mps_ok:
        raise RuntimeError("--device mps: Apple GPU (MPS) is not available on this machine")
    return preference


def gpu_stats(device: str, vram_cap_gb: Optional[float] = None) -> Dict[str, Any]:
    """GPU name and VRAM numbers for the heartbeat (empty for cpu)."""
    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        free_gb = free / GB
        if vram_cap_gb is not None:
            free_gb = min(free_gb, vram_cap_gb)  # share only part of the GPU
        return {"gpu_name": torch.cuda.get_device_name()[:100], "vram_total_gb": round(total / GB, 2),
                "vram_free_gb": round(free_gb, 3), "vram_cap_gb": vram_cap_gb}
    if device == "mps":
        return {"gpu_name": "Apple GPU (MPS)"}
    return {}


def peak_gpu_mb(device: str) -> Optional[float]:
    """Peak VRAM PyTorch allocated in this process (None on cpu/mps)."""
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return None
