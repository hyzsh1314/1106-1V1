import random

import numpy as np
import torch

from benchmarks.drvoi.checkpointing import (
    atomic_torch_save,
    capture_rng_state,
    restore_rng_state,
)


def test_atomic_torch_save_replaces_target_without_leaving_temporary(tmp_path):
    path = tmp_path / "checkpoint.pt"
    atomic_torch_save({"value": torch.tensor([1.0])}, path)
    atomic_torch_save({"value": torch.tensor([2.0])}, path)
    payload = torch.load(path, weights_only=False)
    assert payload["value"].item() == 2.0
    assert not (tmp_path / "checkpoint.pt.tmp").exists()


def test_rng_capture_restore_reproduces_next_draws():
    random.seed(4)
    np.random.seed(4)
    torch.manual_seed(4)
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(3))
    restore_rng_state(state)
    actual = (random.random(), np.random.rand(), torch.rand(3))
    assert expected[0] == actual[0]
    assert expected[1] == actual[1]
    assert torch.equal(expected[2], actual[2])
