import torch

from benchmarks.drvoi.core import action_ids_to_multi_one_hot
from benchmarks.drvoi.rhppo import (
    RecurrentMultiDiscreteActorCritic,
    dual_channel_gae,
)


def test_joint_log_probability_equals_sum_of_four_heads():
    torch.manual_seed(3)
    model = RecurrentMultiDiscreteActorCritic(hidden_size=32)
    obs = torch.randn(5, 39)
    previous = action_ids_to_multi_one_hot(torch.arange(5))
    result = model.act(obs, previous, model.initial_hidden(5))
    distributions = model.distributions(result["belief"], obs)
    manual = sum(
        distribution.log_prob(result["action"][:, index])
        for index, distribution in enumerate(distributions))
    assert torch.allclose(result["log_prob"], manual)


def test_sequence_reset_matches_fresh_hidden_state():
    torch.manual_seed(4)
    model = RecurrentMultiDiscreteActorCritic(hidden_size=32)
    observations = torch.randn(3, 2, 39)
    previous = action_ids_to_multi_one_hot(torch.arange(6)).reshape(3, 2, 13)
    resets = torch.zeros(3, 2, dtype=torch.bool)
    resets[1, 0] = True
    beliefs, _ = model.sequence_beliefs(
        observations, previous, model.initial_hidden(2), resets)
    fresh, _ = model.belief_step(
        observations[1, :1], previous[1, :1], model.initial_hidden(1))
    assert torch.allclose(beliefs[1, 0], fresh[0], atol=1.0e-6)


def test_dual_gae_terminal_mask_and_beta_not_applied_inside():
    result = dual_channel_gae(
        reward_ext=torch.tensor([[1.0], [2.0]]),
        reward_int=torch.tensor([[0.5], [1.0]]),
        value_ext=torch.tensor([[0.2], [0.3]]),
        value_int=torch.tensor([[0.1], [0.2]]),
        next_value_ext=torch.tensor([[0.3], [99.0]]),
        next_value_int=torch.tensor([[0.2], [99.0]]),
        done=torch.tensor([[False], [True]]), gamma=0.9, gae_lambda=0.0)
    assert torch.allclose(result["advantage_ext"], torch.tensor([[1.07], [1.7]]))
    assert torch.allclose(result["advantage_int"], torch.tensor([[0.58], [0.8]]))
