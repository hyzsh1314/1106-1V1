"""Canonical provenance primitives shared by PBGR research components."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def require_sha256(value: str, name: str) -> str:
    """Return one normalized SHA256 digest or fail closed."""

    if not isinstance(value, str):
        raise ValueError("{} must be a SHA256 string".format(name))
    digest = value.strip().lower()
    if len(digest) != 64:
        raise ValueError("{} must be a 64-character SHA256 digest".format(name))
    # ``int(text, 16)`` accepts several Unicode decimal-digit classes (for
    # example full-width or Arabic-Indic zero).  Provenance manifests require
    # the interoperable ASCII representation used by ``hashlib.hexdigest``.
    if any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(
            "{} must be an ASCII hexadecimal SHA256 digest".format(name)
        )
    return digest


def strict_nonnegative_int(value: Any, name: str, positive: bool = False) -> int:
    """Reject booleans and fractional values instead of silently truncating."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError("{} must be an integer".format(name))
    result = int(value)
    lower_bound = 1 if positive else 0
    if result < lower_bound:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError("{} must be {}".format(name, qualifier))
    return result


def canonical_sha256(payload: Any) -> str:
    """Hash one finite, JSON-serializable payload using canonical encoding."""

    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("canonical provenance payload is invalid") from error
    return hashlib.sha256(encoded).hexdigest()
