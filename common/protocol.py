"""PoolForge Binary Tensor Wire Protocol.

Provides secure, pickle-free serialization and framing for PyTorch tensors
over network transports (Relay and DirectTCP).

Wire Frame Layout:
+-------------------+--------------------+------------------------+
| 4-byte uint32     | N-byte JSON Header | M-byte Raw Tensor Data |
| Big-Endian Length | {kind, dtype, ...} | (IEEE 754 float/int)   |
+-------------------+--------------------+------------------------+
"""

import json
import struct
from typing import Any, Dict, Optional, Tuple
import torch

# Standard header length is a 4-byte unsigned big-endian integer
HEADER_LEN_STRUCT = struct.Struct(">I")
UINT32_STRUCT = HEADER_LEN_STRUCT


# Supported torch dtypes mapping for deterministic conversion
DTYPE_STR_MAP = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.int16: "int16",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.bool: "bool",
}

STR_DTYPE_MAP = {v: k for k, v in DTYPE_STR_MAP.items()}


def serialize_tensor(
    tensor: torch.Tensor,
    kind: str = "activation",
    job_id: str = "",
    step: int = 0,
    microbatch: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Serialize a PyTorch tensor into the PoolForge binary wire format.

    Args:
        tensor: The torch.Tensor to serialize.
        kind: Message kind: "activation", "gradient", "labels", or "checkpoint".
        job_id: Identifier of the running training job.
        step: Current training step.
        microbatch: Micro-batch index within the current step.
        extra: Optional arbitrary JSON-serializable metadata.

    Returns:
        Framed bytes ready for transmission over WebSocket or TCP.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

    # Ensure tensor is contiguous and on CPU for byte extraction
    t_cpu = tensor.detach().cpu().contiguous()

    # Extract raw bytes directly via uint8 view without pickle or intermediate copies
    raw_bytes = t_cpu.view(torch.uint8).numpy().tobytes()

    dtype_str = DTYPE_STR_MAP.get(t_cpu.dtype)
    if dtype_str is None:
        raise ValueError(f"Unsupported tensor dtype for wire protocol: {t_cpu.dtype}")

    header_dict = {
        "kind": kind,
        "dtype": dtype_str,
        "shape": list(t_cpu.shape),
        "job_id": job_id,
        "step": step,
        "microbatch": microbatch,
        "num_bytes": len(raw_bytes),
    }

    if extra:
        header_dict["extra"] = extra

    header_bytes = json.dumps(header_dict, separators=(",", ":")).encode("utf-8")
    header_len = len(header_bytes)

    # Wire format: [4-byte header length][JSON header][raw tensor bytes]
    return HEADER_LEN_STRUCT.pack(header_len) + header_bytes + raw_bytes


def deserialize_tensor(data: bytes) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Deserialize binary wire format into a PyTorch tensor and its metadata header.

    Args:
        data: Framed bytes received from the network.

    Returns:
        (tensor, header_dict) where tensor has the original shape, dtype, and contents,
        and is detached with its own memory copy (ready for autograd).
    """
    if len(data) < HEADER_LEN_STRUCT.size:
        raise ValueError(f"Packet too short to contain header length: {len(data)} bytes")

    (header_len,) = HEADER_LEN_STRUCT.unpack_from(data, 0)
    header_start = HEADER_LEN_STRUCT.size
    header_end = header_start + header_len

    if len(data) < header_end:
        raise ValueError(
            f"Packet truncated: claims header length {header_len}, but total payload is {len(data)} bytes"
        )

    header_str = data[header_start:header_end].decode("utf-8")
    header = json.loads(header_str)

    dtype_str = header.get("dtype")
    shape = header.get("shape")
    num_bytes = header.get("num_bytes")

    if dtype_str not in STR_DTYPE_MAP:
        raise ValueError(f"Unknown dtype in tensor header: {dtype_str}")

    torch_dtype = STR_DTYPE_MAP[dtype_str]
    raw_bytes = data[header_end:]

    if num_bytes is not None and len(raw_bytes) != num_bytes:
        raise ValueError(
            f"Corrupt tensor payload: expected {num_bytes} bytes from header, got {len(raw_bytes)}"
        )

    # Make buffer writable to ensure autograd safety and suppress PyTorch non-writable buffer warnings
    writable_bytes = bytearray(raw_bytes)
    tensor = torch.frombuffer(writable_bytes, dtype=torch_dtype).reshape(shape).clone()

    return tensor, header
