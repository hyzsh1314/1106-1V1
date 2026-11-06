from types import SimpleNamespace

import torch

from benchmarks.drvoi_v7_1.snapshot import (
    ENV_TENSORS,
    capture_wvr_snapshot,
    restore_wvr_snapshot,
)


def fake_environment():
    env = SimpleNamespace()
    env.device = torch.device("cpu")
    env.generator = torch.Generator().manual_seed(9)
    for index, name in enumerate(ENV_TENSORS):
        setattr(env, name, torch.full((2, 3), float(index)))
    env.alive = torch.ones(2, 3, dtype=torch.bool)
    env.remaining_missiles = torch.ones(2, 3, dtype=torch.long)
    env.last_launch_step = torch.zeros(2, 3, dtype=torch.long)
    env.episode_step = torch.zeros(2, 3, dtype=torch.long)
    env.episode_id = torch.zeros(2, 3, dtype=torch.long)
    return env


def test_snapshot_restores_environment_rng_lowlevel_and_extra_state():
    env = fake_environment()
    hidden = torch.randn(6, 1, 8)
    snapshot = capture_wvr_snapshot(
        env, hidden, policy_hidden=torch.randn(2, 4))
    expected_random = torch.rand(4, generator=env.generator)
    env.aircraft_state.add_(100.0)
    env.episode_id.add_(5)
    restored_hidden, extra = restore_wvr_snapshot(env, snapshot)
    actual_random = torch.rand(4, generator=env.generator)
    assert torch.equal(expected_random, actual_random)
    assert torch.equal(env.aircraft_state, snapshot.environment["aircraft_state"])
    assert torch.equal(env.episode_id, snapshot.environment["episode_id"])
    assert torch.equal(restored_hidden, hidden)
    assert torch.equal(extra["policy_hidden"], snapshot.extra["policy_hidden"])
