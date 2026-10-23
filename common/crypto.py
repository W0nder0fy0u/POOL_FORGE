"""Optional AES-256-GCM encryption of tensor payloads (Phase 8).

How the key is shared: every agent in the room is started with the SAME
`--encrypt-passphrase`. Each agent derives the key locally with PBKDF2, using
the room code as salt. The coordinator never sees the passphrase, so even
though tensors pass through the relay it only ever sees ciphertext.

Needs the `cryptography` package (pip install cryptography).
"""

import os

KEY_LEN = 32      # 256-bit AES key
NONCE_LEN = 12    # 96-bit GCM nonce (the standard size)
TAG_LEN = 16      # GCM authentication tag
PBKDF2_ITERATIONS = 200_000


def derive_room_key(passphrase: str, room_code: str = "") -> bytes:
    """Derive a 32-byte AES key from a passphrase (salted with the room code)."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=KEY_LEN,
        salt=b"poolforge-v1:" + room_code.upper().encode("utf-8"),
        iterations=PBKDF2_ITERATIONS,
    )
    return kdf.derive(passphrase.encode("utf-8"))


def encrypt_payload(data: bytes, key: bytes) -> bytes:
    """Encrypt bytes. Output = [12-byte random nonce][ciphertext + 16-byte tag]."""
    if len(key) != KEY_LEN:
        raise ValueError(f"AES key must be {KEY_LEN} bytes, got {len(key)}")
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = os.urandom(NONCE_LEN)  # fresh nonce per message: never reuse with the same key
    return nonce + AESGCM(key).encrypt(nonce, data, None)


def decrypt_payload(data: bytes, key: bytes) -> bytes:
    """Decrypt bytes from encrypt_payload. Raises InvalidTag if tampered or wrong key."""
    if len(key) != KEY_LEN:
        raise ValueError(f"AES key must be {KEY_LEN} bytes, got {len(key)}")
    if len(data) < NONCE_LEN + TAG_LEN:
        raise ValueError(f"Encrypted payload too short: {len(data)} bytes")
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    return AESGCM(key).decrypt(data[:NONCE_LEN], data[NONCE_LEN:], None)
