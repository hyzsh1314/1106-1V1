import torch

from benchmarks.collect_drvoi_v7_3_return_horizon_data import (
    add_discounted_reward,
)


def test_discounted_return_keeps_terminal_branch_absorbing():
    total = torch.tensor([1.0, 1.0])
    reward = torch.tensor([2.0, 3.0])
    active = torch.tensor([True, False])
    result = add_discounted_reward(total, reward, active, elapsed=2, gamma=0.5)
    assert torch.allclose(result, torch.tensor([1.5, 1.0]))
