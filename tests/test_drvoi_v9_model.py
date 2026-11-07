import torch

from benchmarks.drvoi.rhppo import RecurrentMultiDiscreteActorCritic
from benchmarks.drvoi_v9 import AdaptiveOptionRecurrentActorCritic


def test_v9_initialization_exactly_recovers_b0_actor():
    torch.manual_seed(5)
    base = RecurrentMultiDiscreteActorCritic(obs_dim=39, hidden_size=32)
    model = AdaptiveOptionRecurrentActorCritic(
        obs_dim=39, hidden_size=32, response_feature_dim=24,
        num_options=7, option_embedding_size=8)
    model.load_from_rhppo(base.state_dict())
    observation = torch.randn(11, 39)
    observation[:, 0] = 1.0
    observation[:, 23] = 1.0
    previous = torch.randn(11, 13)
    hidden = torch.randn(11, 32)
    base_belief, base_hidden = base.belief_step(observation, previous, hidden)
    belief, next_hidden = model.belief_step(observation, previous, hidden)
    assert torch.allclose(belief, base_belief)
    assert torch.allclose(next_hidden, base_hidden)
    base_distributions = base.distributions(base_belief, observation)
    for option_id in range(7):
        options = torch.full((11,), option_id)
        distributions = model.conditioned_distributions(
            belief, options, observation)
        for source, conditioned in zip(base_distributions, distributions):
            assert torch.allclose(source.probs, conditioned.probs, atol=1e-7)


def test_option_policy_respects_legal_mask_and_prior():
    model = AdaptiveOptionRecurrentActorCritic(
        obs_dim=39, hidden_size=16, response_feature_dim=6,
        num_options=3, option_embedding_size=4)
    belief = torch.zeros(2, 16)
    response = torch.zeros(2, 6)
    legal = torch.tensor([[True, False, True], [False, True, True]])
    prior = torch.tensor([[0.0, 0.0, 5.0], [0.0, 5.0, 0.0]])
    logits = model.option_logits(
        belief, response, legal, prior_logits=prior,
        prior_coefficient=10.0)
    assert logits.argmax(-1).tolist() == [2, 1]
    assert logits[0, 1] < -1.0e20
    assert logits[1, 0] < -1.0e20


def test_option_policy_starts_as_exact_gated_prior():
    model = AdaptiveOptionRecurrentActorCritic(
        obs_dim=39, hidden_size=16, response_feature_dim=6,
        num_options=3, option_embedding_size=4)
    belief = torch.randn(2, 16)
    response = torch.randn(2, 6)
    legal = torch.ones(2, 3, dtype=torch.bool)
    prior = torch.tensor([[0.0, 1.0, -1.0], [2.0, -2.0, 0.0]])
    logits = model.option_logits(
        belief, response, legal, prior_logits=prior,
        prior_coefficient=0.5)
    assert torch.allclose(logits, 0.5 * prior)


def test_twin_online_option_q_has_expected_shape():
    model = AdaptiveOptionRecurrentActorCritic(
        obs_dim=39, hidden_size=16, response_feature_dim=6,
        num_options=3, option_embedding_size=4)
    q_a, q_b = model.option_q_values(
        torch.zeros(4, 16), torch.zeros(4, 6),
        torch.ones(4, 3, dtype=torch.bool))
    assert q_a.shape == q_b.shape == (4, 3)
    assert not torch.equal(q_a, q_b)
