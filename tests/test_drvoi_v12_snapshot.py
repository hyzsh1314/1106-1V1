import torch

from benchmarks.drvoi_v12 import (
    capture_execution_snapshot,
    repeat_execution_snapshot,
    restore_execution_snapshot,
    snapshot_scope,
)
from envs.torch6dof import TorchWVRConfig, TorchWVRVecEnv


def make_env():
    return TorchWVRVecEnv(
        TorchWVRConfig(num_envs=2, reset_pool_size=16),
        device="cpu", seed=85, compile_dynamics=False)


def test_snapshot_restores_environment_and_execution_tensors_exactly():
    env = make_env()
    execution = {
        "hidden": torch.arange(12, dtype=torch.float32).reshape(2, 2, 3),
        "option": torch.tensor([1, 2]),
    }
    snapshot = capture_execution_snapshot(env, execution)
    original_observation = env.observations().clone()
    env.aircraft_state.add_(10.0)
    env.remaining_missiles.zero_()
    execution["hidden"].zero_()
    restored = restore_execution_snapshot(env, snapshot)
    torch.testing.assert_close(env.observations(), original_observation)
    torch.testing.assert_close(
        restored["hidden"], snapshot.execution_tensors["hidden"])
    assert restored["option"].tolist() == [1, 2]
    restored["hidden"].zero_()
    assert snapshot.execution_tensors["hidden"].count_nonzero() > 0


def test_snapshot_scope_restores_entry_state_after_branch():
    env = make_env()
    before = capture_execution_snapshot(env)
    with snapshot_scope(env):
        env.aircraft_state.mul_(0.0)
        env.episode_step.add_(7)
    after = capture_execution_snapshot(env)
    for name in before.env_tensors:
        torch.testing.assert_close(
            before.env_tensors[name], after.env_tensors[name])


def test_snapshot_restores_global_torch_rng_for_paired_sampling():
    env = make_env()
    torch.manual_seed(20261107)
    snapshot = capture_execution_snapshot(env)
    first = torch.rand(8)
    restore_execution_snapshot(env, snapshot)
    second = torch.rand(8)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_snapshot_repeat_preserves_environment_and_aircraft_pair_order():
    env = make_env()
    lowlevel = torch.arange(4 * 128, dtype=torch.float32).reshape(4, 1, 128)
    snapshot = capture_execution_snapshot(env, {
        "hidden": torch.tensor([[1.0], [2.0]]),
        "lowlevel_hidden": lowlevel,
    })
    repeated = repeat_execution_snapshot(snapshot, 3)
    assert repeated.num_envs == 6
    assert repeated.execution_tensors["hidden"].flatten().tolist() == [
        1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    paired = repeated.execution_tensors["lowlevel_hidden"].reshape(6, 2, 128)
    for offset in (0, 2, 4):
        torch.testing.assert_close(paired[offset:offset + 2], lowlevel.reshape(2, 2, 128))
