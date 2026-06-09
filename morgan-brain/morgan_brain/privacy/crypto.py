"""Field-level encryption seam — AES-256-GCM envelope with Argon2id key derivation.

All heavy dependencies (``cryptography``, ``argon2-cffi``) are **optional**.
The module imports cleanly without them; errors are raised only when the functions
are *called*, with a clear message pointing to the relevant extra.

Functions::

    new_dek() -> bytes
        Generate a fresh 256-bit random Data Encryption Key.

    seal(plaintext: bytes, dek: bytes) -> bytes
        Encrypt *plaintext* with AES-256-GCM using *dek*.
        Returns ciphertext: 12-byte nonce ‖ GCM ciphertext ‖ 16-byte tag.

    open_sealed(ciphertext: bytes, dek: bytes) -> bytes
        Decrypt ciphertext produced by :func:`seal`.

    derive_kek(passphrase: str, salt: bytes) -> bytes
        Derive a 256-bit Key Encryption Key from *passphrase* + *salt* using
        Argon2id (time_cost=2, memory_cost=65536, parallelism=1, hash_len=32).
        The same (passphrase, salt) pair always produces the same key.

Usage pattern (wiring into stores)::

    salt = os.urandom(16)
    kek = derive_kek(settings.passphrase, salt)
    dek = new_dek()
    encrypted_dek = seal(dek, kek)   # stored alongside the record
    sealed_payload = seal(plaintext_bytes, dek)

    # … later …
    dek = open_sealed(encrypted_dek, kek)
    plaintext = open_sealed(sealed_payload, dek)
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# DEK generation — stdlib only, always works
# ---------------------------------------------------------------------------


def new_dek() -> bytes:
    """Return a fresh 256-bit (32-byte) random Data Encryption Key."""
    return os.urandom(32)


# ---------------------------------------------------------------------------
# Seal / open — requires `cryptography` package
# ---------------------------------------------------------------------------


def seal(plaintext: bytes, dek: bytes) -> bytes:
    """Encrypt *plaintext* with AES-256-GCM using *dek*.

    Returns a self-contained blob: ``nonce(12) ‖ ciphertext ‖ tag(16)``.

    Raises
    ------
    ImportError
        If the ``cryptography`` package is not installed.  Install
        ``morgan-brain[privacy]`` to enable encryption.
    ValueError
        If *dek* is not exactly 32 bytes.
    """
    if len(dek) != 32:
        raise ValueError(f"DEK must be 32 bytes; got {len(dek)}")

    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # type: ignore[import-not-found]  # optional dep
    except ImportError as exc:
        raise ImportError(
            "seal() requires the 'cryptography' package.  "
            "Install it with: pip install 'morgan-brain[privacy]'"
        ) from exc

    nonce = os.urandom(12)
    aesgcm = AESGCM(dek)
    ct = aesgcm.encrypt(nonce, plaintext, None)  # ct includes 16-byte GCM tag appended
    sealed: bytes = nonce + ct
    return sealed


def open_sealed(ciphertext: bytes, dek: bytes) -> bytes:
    """Decrypt *ciphertext* produced by :func:`seal`.

    Raises
    ------
    ImportError
        If the ``cryptography`` package is not installed.
    ValueError
        If *dek* is not exactly 32 bytes or *ciphertext* is too short.
    cryptography.exceptions.InvalidTag
        If authentication fails (wrong key or tampered ciphertext).
    """
    if len(dek) != 32:
        raise ValueError(f"DEK must be 32 bytes; got {len(dek)}")
    if len(ciphertext) < 12 + 16:
        raise ValueError(f"Ciphertext too short: {len(ciphertext)} bytes")

    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # type: ignore[import-not-found]  # optional dep
    except ImportError as exc:
        raise ImportError(
            "open_sealed() requires the 'cryptography' package.  "
            "Install it with: pip install 'morgan-brain[privacy]'"
        ) from exc

    nonce = ciphertext[:12]
    ct = ciphertext[12:]
    aesgcm = AESGCM(dek)
    plaintext: bytes = aesgcm.decrypt(nonce, ct, None)
    return plaintext


# ---------------------------------------------------------------------------
# KEK derivation — requires `argon2-cffi` package
# ---------------------------------------------------------------------------


def derive_kek(passphrase: str, salt: bytes) -> bytes:
    """Derive a 256-bit Key Encryption Key using Argon2id.

    Parameters
    ----------
    passphrase:
        The owner's passphrase (unicode string; encoded to UTF-8 internally).
    salt:
        A per-instance random salt (at least 16 bytes recommended).

    Returns
    -------
    bytes
        32-byte derived key.

    Raises
    ------
    ImportError
        If the ``argon2-cffi`` package is not installed.  Install
        ``morgan-brain[privacy]`` to enable passphrase-based key derivation.
    """
    try:
        from argon2.low_level import (  # type: ignore[import-not-found]  # optional dep
            Type,
            hash_secret_raw,
        )
    except ImportError as exc:
        raise ImportError(
            "derive_kek() requires the 'argon2-cffi' package.  "
            "Install it with: pip install 'morgan-brain[privacy]'"
        ) from exc

    return hash_secret_raw(  # type: ignore[no-any-return]
        secret=passphrase.encode("utf-8"),
        salt=salt,
        time_cost=2,
        memory_cost=65536,
        parallelism=1,
        hash_len=32,
        type=Type.ID,
    )
