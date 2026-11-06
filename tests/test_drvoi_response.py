import torch

from benchmarks.drvoi.response import (
    ResponseEnsemble,
    mixture_nll,
    prequential_learning_progress,
)
from benchmarks.validate_drvoi_wvr_response_p1 import (
    phase_bins,
    stratified_episode_split,
    zero_action_input_weights,
)


def test_response_ensemble_shapes_and_finite_nll():
    model = ResponseEnsemble(belief_dim=39, members=5, hidden_size=32)
    belief = torch.randn(7, 39)
    action = torch.zeros(7, 13)
    action[:, [0, 3, 8, 11]] = 1.0
    means, variances, side = model(belief, action)
    assert means.shape == variances.shape == (5, 7, 12)
    assert side.shape == (5, 7, 3)
    assert torch.all(variances >= 1.0e-4)
    loss = mixture_nll(
        model, belief, action, torch.randn(7, 12), torch.arange(7) % 3)
    assert loss.shape == (7,) and torch.isfinite(loss).all()


def test_candidate_sampling_is_antithetic_and_reproducible():
    model = ResponseEnsemble(belief_dim=39, members=2, hidden_size=16)
    belief = torch.randn(3, 39)
    action = torch.zeros(3, 13)
    current = torch.randn(3, 12)
    first = model.sample_candidates(
        belief, action, current, samples_per_member=4,
        generator=torch.Generator().manual_seed(17))
    second = model.sample_candidates(
        belief, action, current, samples_per_member=4,
        generator=torch.Generator().manual_seed(17))
    for key in first:
        assert torch.equal(first[key], second[key])
    assert torch.allclose(first["weights"].sum(-1), torch.ones(3))


def test_prequential_lp_does_not_mutate_models():
    current = ResponseEnsemble(belief_dim=39, members=2, hidden_size=16)
    delayed = current.delayed_snapshot()
    before = [parameter.clone() for parameter in current.parameters()]
    result = prequential_learning_progress(
        current, delayed, torch.randn(4, 39), torch.zeros(4, 13),
        torch.randn(4, 12), torch.zeros(4, dtype=torch.long))
    assert torch.all(result["lp"] == 0.0)
    assert all(torch.equal(left, right) for left, right in zip(before, current.parameters()))


def test_response_validation_split_has_no_episode_leakage():
    episode = torch.arange(30).repeat_interleave(3)
    opponent = (episode % 3).clone()
    splits, split_episodes = stratified_episode_split(episode, opponent, seed=9)
    assert all(index.numel() > 0 for index in splits.values())
    names = list(split_episodes)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            assert not torch.isin(
                split_episodes[left], split_episodes[right]).any()
    assert sum(value.numel() for value in splits.values()) == episode.numel()


def test_phase_bins_are_computed_within_each_episode():
    episode = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1])
    step = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2])
    assert torch.equal(
        phase_bins(episode, step),
        torch.tensor([0, 0, 1, 1, 2, 2, 0, 1, 2]))


def test_zero_action_initialization_makes_paired_models_identical():
    conditioned = ResponseEnsemble(belief_dim=8, members=2, hidden_size=16)
    zero_action_input_weights(conditioned)
    no_action = ResponseEnsemble(belief_dim=8, members=2, hidden_size=16)
    no_action.load_state_dict(conditioned.state_dict())
    belief = torch.randn(11, 8)
    action = torch.zeros(11, 13)
    action[:, [0, 3, 8, 11]] = 1.0
    conditioned_output = conditioned(belief, action)
    no_action_output = no_action(belief, torch.zeros_like(action))
    assert all(torch.equal(left, right) for left, right in zip(
        conditioned_output, no_action_output))
