import copy

import torch

from benchmarks.drvoi_v9 import (
    linear_anneal,
    option_segment_targets,
    polyak_update,
    progress_violation_events,
)


def test_option_segment_targets_bootstrap_and_terminal_mask():
    rewards = torch.tensor([
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0],
    ])
    done = torch.tensor([
        [False, False],
        [False, True],
        [False, False],
        [False, False],
    ])
    bootstrap = torch.full((5, 2), 10.0)
    targets = option_segment_targets(
        rewards, done, bootstrap, horizon=2, gamma=0.5)
    assert torch.allclose(targets[0], torch.tensor([4.5, 2.0]))
    assert torch.allclose(targets[2], torch.tensor([7.5, 7.5]))
    assert torch.allclose(targets[3], torch.tensor([9.0, 9.0]))


def test_polyak_update_and_linear_anneal():
    source = torch.nn.Linear(3, 2)
    target = copy.deepcopy(source)
    with torch.no_grad():
        source.weight.add_(2.0)
    before = target.weight.clone()
    polyak_update(source, target, tau=0.25)
    assert torch.allclose(target.weight, before + 0.5)
    assert linear_anneal(2.0, 0.0, 1, 5) == 2.0
    assert linear_anneal(2.0, 0.0, 3, 5) == 1.0
    assert linear_anneal(2.0, 0.0, 5, 5) == 0.0


def test_progress_cost_is_a_window_event_not_persistent_penalty():
    steps = torch.tensor([0, 59, 60, 61, 119, 120, 121])
    cost = progress_violation_events(steps, window=60)
    assert cost.tolist() == [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
