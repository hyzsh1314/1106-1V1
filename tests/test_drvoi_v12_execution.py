import torch

from benchmarks.drvoi_v12 import (
    EXECUTION_FEATURE_DIM,
    ROUTER_FEATURE_DIM,
    V9ExecutionAdapter,
    build_execution_features,
    build_router_features,
    lead_intercept_expert_actions,
    opponent_hypothesis_actions,
)


def test_all_opponent_hypotheses_are_distinct_and_fire_only_when_legal():
    observation = torch.zeros(5, 2, 39)
    observation[..., 14] = 1.0
    observation[:, 0, 23] = 1.0
    kinds = torch.arange(5)
    episode_step = torch.tensor([0, 0, 0, 0, 31])
    action = opponent_hypothesis_actions(observation, kinds, episode_step)
    assert action.shape == (5, 2, 4)
    assert action[:, 0, 1].tolist() == [2, 4, 0, 4, 4]
    assert action[:, 0, 3].tolist() == [1, 1, 1, 1, 1]
    assert action[:, 1, 3].tolist() == [0, 0, 0, 0, 0]


def test_lead_intercept_expert_fires_exactly_on_legal_mask():
    observation = torch.zeros(3, 39)
    observation[:, 23] = torch.tensor([0.0, 1.0, 0.7])
    action = lead_intercept_expert_actions(observation)
    assert action[:, 3].tolist() == [0, 1, 1]


def test_v9_post_step_applies_terminal_masks_to_shadow_state():
    adapter = object.__new__(V9ExecutionAdapter)
    done = torch.tensor([False, True])
    state = {
        "pending_previous_action": torch.ones(2, 13),
        "pending_option_decision": torch.tensor([False, True]),
        "option_age": torch.tensor([4, 8]),
        "active_option": torch.tensor([3, 5]),
        "expert_age": torch.tensor([7, 9]),
        "pending_lowlevel_hidden": torch.ones(4, 1, 128),
    }
    updated = adapter.post_step(None, state, done, {})
    assert updated["option_age"].tolist() == [5, 0]
    assert updated["active_option"].tolist() == [3, 0]
    assert updated["previous_action"][0].count_nonzero() == 13
    assert updated["previous_action"][1].count_nonzero() == 0
    assert updated["lowlevel_hidden"][:2].count_nonzero() == 256
    assert updated["lowlevel_hidden"][2:].count_nonzero() == 0
    assert updated["expert_age"].tolist() == [8, 0]
    assert torch.equal(updated["reset"], done)


def test_execution_features_have_frozen_auditable_dimension():
    batch = 3
    observation = torch.zeros(batch, 39)
    state = {
        "hidden": torch.zeros(batch, 256),
        "previous_action": torch.zeros(batch, 13),
        "active_option": torch.tensor([0, 1, 6]),
        "option_age": torch.tensor([0, 6, 150]),
        "lowlevel_hidden": torch.zeros(batch * 2, 1, 128),
        "selected_expert": torch.tensor([0, 1, 0]),
        "expert_age": torch.tensor([0, 12, 150]),
    }
    features = build_execution_features(observation, state)
    assert features.shape == (batch, EXECUTION_FEATURE_DIM)
    assert EXECUTION_FEATURE_DIM == 327
    assert torch.isfinite(features).all()
    reachability = {
        key: torch.zeros(batch, 2)
        for key in ("lcb", "score_mean", "probability", "time")
    }
    router = build_router_features(features, reachability)
    assert router.shape == (batch, ROUTER_FEATURE_DIM)
    assert ROUTER_FEATURE_DIM == 335
