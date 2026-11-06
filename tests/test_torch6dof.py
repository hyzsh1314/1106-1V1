import pytest
import torch

from envs.torch6dof import TorchF16BatchDynamics


def test_torch6dof_shapes_finite_and_differentiable():
    model = TorchF16BatchDynamics()
    position = torch.tensor([[0.0, 0.0, -6000.0]]).repeat(8, 1)
    heading = torch.linspace(0.0, 1.0, 8)
    speed = torch.full((8,), 250.0)
    state = model.level_state(position, heading, speed).requires_grad_(True)
    controls = torch.tensor([[0.1, -0.05, 0.02, 0.75]]).repeat(8, 1)
    successor = model(state, controls, substeps=12)
    assert successor.shape == (8, 20)
    assert torch.isfinite(successor).all()
    successor[..., :13].square().mean().backward()
    assert state.grad is not None
    assert torch.isfinite(state.grad).all()


def test_torch6dof_fixed_rate_and_quaternion_norm():
    model = TorchF16BatchDynamics(dt=1.0 / 60.0)
    assert model.dt == pytest.approx(1.0 / 60.0)
    state = model.level_state(
        torch.tensor([[0.0, 0.0, -6000.0]]),
        torch.tensor([0.0]), torch.tensor([250.0]))
    control = torch.tensor([[0.0, 0.0, 0.0, 0.7]])
    successor = model(state, control, substeps=120)
    norm = torch.linalg.vector_norm(successor[..., 6:10], dim=-1)
    torch.testing.assert_close(norm, torch.ones_like(norm))
    assert torch.isfinite(successor).all()
