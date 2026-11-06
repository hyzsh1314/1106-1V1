import numpy as np
import pytest
import torch

from benchmarks.drvoi import (
    CrossFittedQTransitionAssembler,
    HOTTab,
    action_ids_to_multi_one_hot,
    cross_fitted_voi,
    ensemble_mi,
    extract_response_state,
)


def test_action_encoding_enumerates_all_90_unique_actions():
    encoded = action_ids_to_multi_one_hot(torch.arange(90))
    assert encoded.shape == (90, 13)
    assert torch.unique(encoded, dim=0).shape[0] == 90
    assert torch.all(encoded.sum(-1) == 4)


def test_response_extraction_uses_only_v4_legal_observation():
    obs = torch.zeros(2, 39)
    obs[:, 11] = torch.tensor([0.2, -0.3])
    obs[:, 12] = torch.tensor([-0.4, 0.5])
    obs[:, 14] = torch.tensor([-1.0, 1.0])
    continuous, side = extract_response_state(obs)
    assert continuous.shape == (2, 12)
    assert side.tolist() == [0, 2]
    with pytest.raises(ValueError, match="39-D"):
        extract_response_state(torch.zeros(2, 21))


def test_cross_fitted_voi_is_zero_when_one_action_is_common_optimum():
    base = torch.tensor([[[3.0, 1.0], [3.0, 2.0], [3.0, -1.0]]])
    result = cross_fitted_voi(base, base.clone(), torch.ones(1, 3))
    assert torch.allclose(result["signed_voi"], torch.zeros(1))


def test_cross_fitted_voi_matches_independent_manual_example():
    q = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    result = cross_fitted_voi(q, q.clone(), torch.tensor([[0.5, 0.5]]))
    assert torch.allclose(result["signed_voi"], torch.tensor([1.0]))
    assert torch.allclose(result["positive_voi"], torch.tensor([1.0]))


def test_ensemble_mi_null_and_nonnegative():
    means = torch.zeros(5, 4, 3)
    variances = torch.ones_like(means)
    side = torch.full((5, 4, 3), 1.0 / 3.0)
    null = ensemble_mi(means, variances, side)
    assert torch.allclose(null["total"], torch.zeros(4), atol=1.0e-6)
    means[0] = 2.0
    nonnull = ensemble_mi(means, variances, side)
    assert torch.all(nonnull["continuous"] > 0)
    assert torch.all(nonnull["total"] >= 0)


def test_hot_tab_controls_have_expected_oracle_voi():
    for mode in ("N0", "N1"):
        _, voi = HOTTab(mode).oracle_q_voi()
        assert np.allclose(voi, 0.0, atol=1.0e-12)
    _, voi = HOTTab("N2").oracle_q_voi()
    assert voi[0] > voi[1] > voi[2] - 1.0e-12
    assert np.isclose(voi[2], 0.0, atol=1.0e-12)


def test_hot_tab_oracle_matches_brute_force_episode_value():
    env = HOTTab("N2")
    q, voi = env.oracle_q_voi()
    ctx = env.context()
    for action in range(3):
        likelihood = env.response_likelihood(float(ctx.reliabilities[action]))
        informed = 0.0
        for latent in range(3):
            for response in range(3):
                probability = ctx.prior[latent] * likelihood[latent, response]
                informed += probability * q[action, response].max()
        common = (ctx.prior[:, None] * ctx.reward_matrix).sum(0).max()
        assert np.isclose(voi[action], informed - common)


def test_q_transition_assembler_preserves_episode_fold_and_alignment():
    assembler = CrossFittedQTransitionAssembler()
    first = assembler.observe_context(
        0, 7, torch.tensor([1.0]), torch.tensor(2), torch.tensor(0.1), False)
    assert first is None
    second = assembler.observe_context(
        0, 7, torch.tensor([2.0]), torch.tensor(3), torch.tensor(0.7), False)
    assert second is not None
    assert second.episode_id == 7 and second.fold == 1
    assert torch.equal(second.context, torch.tensor([1.0]))
    assert torch.equal(second.next_action, torch.tensor(2))
    assert torch.equal(second.next_reward_ext, torch.tensor(0.7))
    assert torch.equal(second.next_context, torch.tensor([2.0]))
    terminal = assembler.observe_context(
        0, 7, torch.tensor([3.0]), torch.tensor(4), torch.tensor(1.0), True)
    assert terminal.done and terminal.next_context is None
