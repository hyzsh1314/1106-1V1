import torch

from envs.torch6dof import TorchF16BatchDynamics, TorchWVRMissileBatch
from envs.torch6dof.tables import lookup_1d, lookup_2d


def test_lookup_tables_interpolate_and_clamp():
    grid = torch.tensor([0.0, 1.0, 2.0])
    values = torch.tensor([0.0, 10.0, 30.0])
    actual = lookup_1d(torch.tensor([-1.0, 0.5, 3.0]), grid, values)
    torch.testing.assert_close(actual, torch.tensor([0.0, 5.0, 30.0]))
    matrix = torch.tensor([[0.0, 10.0], [20.0, 30.0]])
    actual_2d = lookup_2d(
        torch.tensor([0.5]), torch.tensor([0.5]),
        torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]), matrix)
    torch.testing.assert_close(actual_2d, torch.tensor([15.0]))


def test_f100_table_is_finite_and_responds_to_throttle():
    dynamics = TorchF16BatchDynamics()
    throttle = torch.tensor([0.0, 0.5, 1.0])
    thrust = dynamics._f100_thrust(
        throttle, torch.full_like(throttle, 0.7),
        torch.full_like(throttle, 6000.0))
    assert torch.isfinite(thrust).all()
    assert torch.all(thrust[1:] > thrust[:-1])


def test_batched_missile_capture_absorbing_and_dead_target():
    missile = TorchWVRMissileBatch()
    position = torch.zeros(3, 3)
    velocity = torch.tensor([[300.0, 0.0, 0.0]]).repeat(3, 1)
    state = missile.launch(position, velocity)
    targets = torch.tensor([
        [20.0, 0.0, 0.0],
        [5000.0, 0.0, 0.0],
        [5000.0, 0.0, 0.0],
    ])
    target_velocity = torch.zeros_like(targets)
    successor = missile(
        state, targets, target_velocity,
        target_alive=torch.tensor([True, True, False]))
    assert successor[0, 12].item() == missile.HIT
    assert successor[1, 12].item() == missile.FLYING
    assert successor[2, 12].item() == missile.MISS
    repeated = missile(successor, targets, target_velocity)
    torch.testing.assert_close(repeated[0], successor[0])
    torch.testing.assert_close(repeated[2], successor[2])


def test_batched_missile_large_batch_is_finite_and_independent():
    missile = TorchWVRMissileBatch()
    position = torch.zeros(512, 3)
    position[:, 1] = torch.linspace(-1000.0, 1000.0, 512)
    velocity = torch.zeros_like(position)
    velocity[:, 0] = 300.0
    target_position = position.clone()
    target_position[:, 0] = 8000.0
    target_velocity = torch.zeros_like(position)
    state = missile.launch(position, velocity)
    for _ in range(120):
        state = missile(state, target_position, target_velocity)
    assert torch.isfinite(state).all()
    assert torch.all(state[:, 12] == missile.FLYING)
    assert torch.unique(state[:, 1]).numel() == 512
