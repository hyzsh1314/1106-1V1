from types import SimpleNamespace

import torch

from benchmarks.validate_drvoi_wvr_response_v7 import (
    flatten_response_data,
    stratified_action_permutation,
    train_paired_ensembles,
)


def synthetic_episode(episode_id=1, opponent=0, split="train", length=12):
    observation = torch.zeros(length, 39)
    observation[:, 9] = torch.arange(length) / 10.0
    observation[:, 13] = torch.arange(length) / 20.0
    done = torch.zeros(length, dtype=torch.bool)
    done[-1] = True
    return {
        "observation": observation,
        "belief": torch.randn(length, 8),
        "action": torch.stack((
            torch.arange(length) % 3,
            torch.arange(length) % 5,
            torch.arange(length) % 3,
            torch.zeros(length, dtype=torch.long),
        ), -1),
        "reward": torch.zeros(length),
        "done": done,
        "episode_id": episode_id,
        "opponent": opponent,
        "split": split,
    }


def test_flatten_response_uses_delta_without_crossing_terminal():
    flat = flatten_response_data([synthetic_episode()], "train", 3, "rel")
    assert flat.target.shape == (9, 12)
    assert torch.allclose(flat.target[:, 0], torch.full((9,), 0.3))
    assert torch.unique(flat.episode).tolist() == [1]


def test_stratified_permutation_does_not_cross_opponent_or_phase():
    action = torch.arange(12).float().unsqueeze(-1)
    opponent = torch.tensor([0] * 6 + [1] * 6)
    phase = torch.tensor([0] * 3 + [1] * 3 + [0] * 3 + [1] * 3)
    permuted = stratified_action_permutation(action, opponent, phase, 7)
    for opponent_id in (0, 1):
        for phase_id in (0, 1):
            selected = (opponent == opponent_id) & (phase == phase_id)
            assert set(permuted[selected, 0].tolist()) == set(action[selected, 0].tolist())


def test_paired_models_start_identically_when_action_weights_are_zero():
    episodes = []
    for index in range(6):
        episodes.append(synthetic_episode(
            index, index % 3, "train" if index < 3 else "validation"))
    train = flatten_response_data(episodes, "train", 1, "opp")
    validation = flatten_response_data(episodes, "validation", 1, "opp")
    args = SimpleNamespace(
        hidden_size=16, seed=5, epochs=1, patience=1, batch_size=64)
    _, _, report = train_paired_ensembles(
        train, validation, "direct", 3, args)
    assert report["initial_function_max_difference"] == 0.0
    assert report["raw_gradient_norm_max"] < 1000.0
