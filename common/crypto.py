"""Optional AES-GCM encryption for tensor payloads.

Uses a per-room key derived from the room secret using PBKDF2-HMAC-SHA256.
Enable with --encrypt flag on the agent CLI.

Only imported when encryption is requested — the cryptography library must be installed:
    pip install cryptography

Key derivation is deterministic: all nodes in the same room derive the same key
from the room secret, so no out-of-band key exchange is needed.
"""

import os
from typing import Optional

# Salt is fixed so that all nodes independently derive the same key from the same secret.
# This is intentional — the "secret" itself is the entropy source.
_PBKDF2_SALT = b"poolforge-v1-aes-gcm-salt"
_PBKDF2_ITERATIONS = 200_000
_AES_GCM_KEY_LEN = 32    # 256-bit AES key
_AES_GCM_NONCE_LEN = 12  # 96-bit nonce (GCM standard)


def derive_room_key(room_secret: str) -> bytes:
    """Derive a 256-bit AES key from the room secret via PBKDF2-HMAC-SHA256.

    All nodes in the same room derive the same key independently (no key exchange).
    The room secret should be at least 16 characters for adequate security.

    Args:
        room_secret: The shared secret string for the room (e.g. the admin_token or
                     a user-supplied passphrase).

    Returns:
        32-byte AES-256 key.
    """
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.backends import default_backend

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=_AES_GCM_KEY_LEN,
        salt=_PBKDF2_SALT,
        iterations=_PBKDF2_ITERATIONS,
        backend=default_backend(),
    )
    return kdf.derive(room_secret.encode("utf-8"))


def encrypt_payload(data: bytes, key: bytes) -> bytes:
    """Encrypt bytes using AES-256-GCM.

    A fresh 12-byte random nonce is prepended to every ciphertext so the output
    is self-contained and safe to transmit over the wire.

    Wire layout:
        [ 12-byte nonce ] [ variable-length ciphertext + 16-byte GCM tag ]

    Args:
        data: Plaintext bytes to encrypt (typically the raw tensor bytes portion).
        key: 32-byte AES key, as returned by ``derive_room_key``.

    Returns:
        Nonce-prepended ciphertext bytes (length = len(data) + 12 + 16).
    """
    if len(key) != _AES_GCM_KEY_LEN:
        raise ValueError(f"AES key must be {_AES_GCM_KEY_LEN} bytes, got {len(key)}")

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = os.urandom(_AES_GCM_NONCE_LEN)
    aesgcm = AESGCM(key)
    # AESGCM.encrypt returns ciphertext + 16-byte authentication tag concatenated
    ciphertext = aesgcm.encrypt(nonce, data, associated_data=None)
    return nonce + ciphertext


def decrypt_payload(data: bytes, key: bytes) -> bytes:
    """Decrypt AES-256-GCM ciphertext produced by ``encrypt_payload``.

    Args:
        data: Nonce-prepended ciphertext bytes (as returned by ``encrypt_payload``).
        key: 32-byte AES key, as returned by ``derive_room_key``.

    Returns:
        Decrypted plaintext bytes.

    Raises:
        ValueError: If the data is too short to contain nonce + tag.
        cryptography.exceptions.InvalidTag: If the ciphertext has been tampered with.
    """
    min_len = _AES_GCM_NONCE_LEN + 16  # nonce + GCM auth tag minimum
    if len(data) < min_len:
        raise ValueError(
            f"Encrypted payload too short: {len(data)} bytes (minimum {min_len})"
        )
    if len(key) != _AES_GCM_KEY_LEN:
        raise ValueError(f"AES key must be {_AES_GCM_KEY_LEN} bytes, got {len(key)}")

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = data[:_AES_GCM_NONCE_LEN]
    ciphertext = data[_AES_GCM_NONCE_LEN:]
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, associated_data=None)
