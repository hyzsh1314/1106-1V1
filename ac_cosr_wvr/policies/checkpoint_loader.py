"""Hash enforcement for checkpoint loading; architecture belongs in manifests."""

from __future__ import annotations

import os

from ..schemas import sha256_file


def require_checkpoint_hash(path, expected_sha256):
    if not expected_sha256:
        raise KeyError("checkpoint hash is required and may not be bypassed")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual.lower() != expected_sha256.lower():
        raise ValueError("checkpoint SHA256 mismatch: expected {0}, got {1}".format(expected_sha256, actual))
    return actual


def load_checkpoint(manifest, loader):
    """Verify first, then let an injected loader select its own architecture."""
    require_checkpoint_hash(manifest.checkpoint_path, manifest.checkpoint_sha256)
    return loader(manifest)
