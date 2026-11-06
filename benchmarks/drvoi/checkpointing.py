"""Atomic V7 checkpoints with reproducible RNG capture."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def capture_rng_state() -> dict[str, object]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # ``map_location='cuda'`` also moves the saved CPU generator tensor.
    # Both RNG restoration APIs require CPU ByteTensors.
    torch.set_rng_state(
        state["torch"].detach().to(device="cpu", dtype=torch.uint8))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([
            value.detach().to(device="cpu", dtype=torch.uint8)
            for value in state["cuda"]
        ])


def atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    # Windows requires a writable file descriptor for fsync.
    with temporary.open("r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
