import torch

from benchmarks.drvoi_v10 import (
    LEAD_INTERCEPT_OPTION,
    HystereticReachabilityArbitrator,
    EpisodeLatchedReachabilityArbitrator,
    SustainedReachabilityArbitrator,
    ReachabilityAnchoredOptionActorCritic,
    anchored_action_logits,
    nonclosing_intercept_guard,
    reachability_option_legal_mask,
    reachability_shielded_actions,
)
from benchmarks.drvoi_v7_1.macro_options import macro_option_actions


def test_nonclosing_guard_uses_range_and_closure():
    obs = torch.zeros(4, 39)
    obs[:, 13] = torch.tensor([0.7, 0.5, 0.7, 0.7])
    obs[:, 28] = torch.tensor([0.0, 0.0, 0.1, 0.05])
    assert nonclosing_intercept_guard(obs).tolist() == [True, False, False, True]


def test_guard_leaves_exactly_lead_intercept_legal():
    obs = torch.zeros(2, 39)
    obs[:, 13] = torch.tensor([0.8, 0.4])
    obs[:, 28] = 0.0
    obs[:, 23] = 1.0
    legal = reachability_option_legal_mask(obs)
    assert legal[0].sum() == 1
    assert bool(legal[0, LEAD_INTERCEPT_OPTION])
    assert legal[1].all()


def test_anchor_bias_targets_macro_action():
    obs = torch.zeros(3, 39)
    obs[:, 14] = torch.tensor([-1.0, 0.0, 1.0])
    obs[:, 24] = torch.tensor([-0.2, 0.0, 0.2])
    option = torch.tensor([0, 1, 4])
    logits = [torch.zeros(3, size) for size in (3, 5, 3, 2)]
    anchored = anchored_action_logits(logits, obs, option, strength=3.0)
    selected = torch.stack([value.argmax(-1) for value in anchored], -1)
    expected = macro_option_actions(obs, option, elapsed=1)
    assert torch.equal(selected, expected)


def test_lead_intercept_anchor_fires_when_legal():
    obs = torch.zeros(2, 39)
    obs[:, 23] = torch.tensor([0.0, 1.0])
    option = torch.full((2,), LEAD_INTERCEPT_OPTION)
    logits = [torch.zeros(2, size) for size in (3, 5, 3, 2)]
    anchored = anchored_action_logits(logits, obs, option, strength=8.0)
    assert anchored[-1].argmax(-1).tolist() == [0, 1]


def test_reachability_shield_only_replaces_guarded_rows():
    obs = torch.zeros(2, 39)
    obs[:, 13] = torch.tensor([0.8, 0.4])
    obs[:, 28] = 0.0
    obs[:, 23] = 1.0
    learned = torch.tensor([[2, 0, 2, 0], [2, 0, 2, 0]])
    action, guard = reachability_shielded_actions(obs, learned)
    assert guard.tolist() == [True, False]
    assert action[0, 3] == 1
    assert torch.equal(action[1], learned[1])


def test_hysteretic_arbitrator_latches_until_recovery():
    arbitrator = HystereticReachabilityArbitrator()
    learned = torch.tensor([[1, 0, 2, 0]])
    obs = torch.zeros(1, 39)
    obs[:, 13] = 0.8
    obs[:, 28] = 0.0
    first = arbitrator(obs, learned, torch.ones(1, dtype=torch.bool))
    assert not torch.equal(first, learned)
    obs[:, 28] = 0.08  # no longer enters, but not yet recovered
    latched = arbitrator(obs, learned, torch.zeros(1, dtype=torch.bool))
    assert not torch.equal(latched, learned)
    obs[:, 13] = 0.5
    obs[:, 28] = 0.2
    arbitrator(obs, learned, torch.zeros(1, dtype=torch.bool))
    released = arbitrator(obs, learned, torch.zeros(1, dtype=torch.bool))
    assert torch.equal(released, learned)


def test_episode_latched_arbitrator_releases_only_on_reset():
    arbitrator = EpisodeLatchedReachabilityArbitrator()
    learned = torch.tensor([[1, 0, 2, 0]])
    obs = torch.zeros(1, 39)
    obs[:, 13] = 0.8
    obs[:, 28] = 0.0
    first = arbitrator(obs, learned, torch.ones(1, dtype=torch.bool))
    assert not torch.equal(first, learned)
    obs[:, 13] = 0.3
    obs[:, 28] = 0.5
    latched = arbitrator(obs, learned, torch.zeros(1, dtype=torch.bool))
    assert not torch.equal(latched, learned)
    released = arbitrator(obs, learned, torch.ones(1, dtype=torch.bool))
    assert torch.equal(released, learned)


def test_sustained_arbitrator_filters_transient_violation():
    arbitrator = SustainedReachabilityArbitrator(enter_steps=3)
    learned = torch.tensor([[1, 0, 2, 0]])
    obs = torch.zeros(1, 39)
    obs[:, 13] = 0.8
    obs[:, 28] = 0.0
    for step in range(2):
        action = arbitrator(
            obs, learned, torch.tensor([step == 0]))
        assert torch.equal(action, learned)
    obs[:, 28] = 0.2
    assert torch.equal(
        arbitrator(obs, learned, torch.zeros(1, dtype=torch.bool)), learned)
    obs[:, 28] = 0.0
    for _ in range(2):
        assert torch.equal(
            arbitrator(obs, learned, torch.zeros(1, dtype=torch.bool)),
            learned)
    assert not torch.equal(
        arbitrator(obs, learned, torch.zeros(1, dtype=torch.bool)), learned)


def test_anchored_model_persists_strength_in_state_dict():
    model = ReachabilityAnchoredOptionActorCritic(
        obs_dim=39, hidden_size=16, response_feature_dim=4,
        num_options=7, option_embedding_size=3, anchor_strength=2.5)
    state = model.state_dict()
    assert torch.isclose(state["anchor_strength"], torch.tensor(2.5))
    clone = ReachabilityAnchoredOptionActorCritic(
        obs_dim=39, hidden_size=16, response_feature_dim=4,
        num_options=7, option_embedding_size=3, anchor_strength=0.0)
    clone.load_state_dict(state)
    assert torch.isclose(clone.anchor_strength, torch.tensor(2.5))
