"""PoolForge binary tensor wire format (NO pickle, ever).

One tensor frame looks like this:

    +----------------+----------------------+---------------------+
    | 4 bytes        | N bytes              | M bytes             |
    | header length  | JSON header (UTF-8)  | raw tensor bytes    |
    | (big-endian)   | {kind, dtype, ...}   | (or AES ciphertext) |
    +----------------+----------------------+---------------------+

The receiver reads the JSON header first, checks every field, and only then
rebuilds the tensor with torch.frombuffer(...).reshape(shape). Because the
payload is just numbers plus a JSON description, a malicious peer cannot make
us execute code (which is exactly what pickle would allow).

Optional encryption (Phase 8): only the raw bytes are encrypted; the header
stays readable and carries "encrypted": true.
"""

import json
import math
import struct
from typing import Any, Dict, Optional, Tuple

import torch

# 4-byte unsigned big-endian integer used for every length prefix in PoolForge.
UINT32 = struct.Struct(">I")
UINT32_STRUCT = UINT32  # backwards-compatible alias

# Hard limits so a bad peer can't make us allocate absurd amounts of memory.
MAX_HEADER_BYTES = 64 * 1024
MAX_PAYLOAD_BYTES = 512 * 1024 * 1024  # 512 MB per tensor

# Allowed message kinds on the data plane.
KINDS = ("activation", "gradient", "labels")

# dtype <-> string table (only plain numeric types are allowed on the wire).
DTYPE_TO_STR = {
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
STR_TO_DTYPE = {v: k for k, v in DTYPE_TO_STR.items()}


def serialize_tensor(
    tensor: torch.Tensor,
    kind: str = "activation",
    job_id: str = "",
    step: int = 0,
    microbatch: int = 0,
    encrypt_key: Optional[bytes] = None,
) -> bytes:
    """Turn a tensor into one framed message: [len][JSON header][raw bytes]."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor).__name__}")
    if kind not in KINDS:
        raise ValueError(f"Unknown frame kind '{kind}', expected one of {KINDS}")

    t = tensor.detach().cpu().contiguous()
    dtype_str = DTYPE_TO_STR.get(t.dtype)
    if dtype_str is None:
        raise ValueError(f"Unsupported dtype for the wire: {t.dtype}")

    # View the tensor memory as bytes (no pickle, no Python objects).
    raw = t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b""

    encrypted = False
    if encrypt_key is not None:
        from common.crypto import encrypt_payload

        raw = encrypt_payload(raw, encrypt_key)
        encrypted = True

    header = {
        "kind": kind,
        "dtype": dtype_str,
        "shape": list(t.shape),
        "job_id": job_id,
        "step": int(step),
        "microbatch": int(microbatch),
        "num_bytes": len(raw),
        "encrypted": encrypted,
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return UINT32.pack(len(header_bytes)) + header_bytes + raw


def read_header(data: bytes) -> Tuple[Dict[str, Any], int]:
    """Parse and validate only the JSON header. Returns (header, payload_offset)."""
    if len(data) < UINT32.size:
        raise ValueError(f"Frame too short: {len(data)} bytes")
    (header_len,) = UINT32.unpack_from(data, 0)
    if header_len > MAX_HEADER_BYTES:
        raise ValueError(f"Header too large: {header_len} bytes")
    end = UINT32.size + header_len
    if len(data) < end:
        raise ValueError("Frame truncated inside the header")

    try:
        header = json.loads(bytes(data[UINT32.size:end]).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Header is not valid JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError("Header must be a JSON object")

    # Validate every field we rely on.
    if header.get("kind") not in KINDS:
        raise ValueError(f"Bad kind: {header.get('kind')!r}")
    if header.get("dtype") not in STR_TO_DTYPE:
        raise ValueError(f"Bad dtype: {header.get('dtype')!r}")
    shape = header.get("shape")
    if not isinstance(shape, list) or not all(isinstance(d, int) and d >= 0 for d in shape):
        raise ValueError(f"Bad shape: {shape!r}")
    for field in ("step", "microbatch", "num_bytes"):
        if not isinstance(header.get(field), int) or header[field] < 0:
            raise ValueError(f"Bad {field}: {header.get(field)!r}")
    if not isinstance(header.get("job_id"), str):
        raise ValueError("Bad job_id")
    if header["num_bytes"] > MAX_PAYLOAD_BYTES:
        raise ValueError(f"Payload too large: {header['num_bytes']} bytes")
    return header, end


def deserialize_tensor(
    data: bytes, decrypt_key: Optional[bytes] = None
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Rebuild (tensor, header) from a frame produced by serialize_tensor."""
    header, offset = read_header(data)
    raw = data[offset:]
    if len(raw) != header["num_bytes"]:
        raise ValueError(f"Payload is {len(raw)} bytes, header says {header['num_bytes']}")

    if header.get("encrypted"):
        if decrypt_key is None:
            raise ValueError("Frame is encrypted but no key was given (use --encrypt-passphrase)")
        from common.crypto import decrypt_payload

        raw = decrypt_payload(bytes(raw), decrypt_key)

    dtype = STR_TO_DTYPE[header["dtype"]]
    shape = header["shape"]
    expected = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if len(raw) != expected:
        raise ValueError(f"Shape {shape} of {header['dtype']} needs {expected} bytes, got {len(raw)}")

    if expected == 0:
        return torch.empty(shape, dtype=dtype), header
    # bytearray gives frombuffer a writable buffer; clone() gives the tensor its own memory.
    tensor = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape).clone()
    return tensor, header
