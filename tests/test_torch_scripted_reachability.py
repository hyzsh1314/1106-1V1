import torch

from benchmarks.evaluate_torch_wvr_fixed_opponents import (
    scripted_learner_actions,
)
from benchmarks.pretrain_torch_wvr_bc import (
    behavior_cloning_loss,
    teacher_opponent_actions,
)
from benchmarks.train_ppo_1v1_shoot_smoke import MultiDiscreteActorCritic


def test_scripted_reachability_controllers_use_legal_fire_mask():
    obs = torch.zeros(2, 1, 39)
    obs[0, 0, 23] = 1.0
    obs[:, 0, 14] = torch.tensor([1.0, -1.0])
    pure = scripted_learner_actions(obs, "pure-pursuit")
    assert pure[:, 0, 1].tolist() == [4, 0]
    assert pure[:, 0, 2].tolist() == [0, 0]
    assert pure[:, 0, 3].tolist() == [1, 0]


def test_los_intercept_quantizes_signed_steering():
    obs = torch.zeros(5, 1, 39)
    obs[:, 0, 24] = torch.tensor([-0.2, -0.03, 0.0, 0.03, 0.2])
    action = scripted_learner_actions(obs, "los-intercept")
    assert action[:, 0, 1].tolist() == [0, 1, 2, 3, 4]


def test_behavior_cloning_loss_is_finite_with_no_legal_fire_samples():
    model = MultiDiscreteActorCritic(
        obs_dim=39, action_dims=[3, 5, 3, 2], hidden_size=16,
        fire_mask_obs_index=23, altitude_safety_obs_index=0,
        altitude_descend_mask_threshold=0.8,
        altitude_force_climb_threshold=0.6)
    obs = torch.zeros(8, 39)
    obs[:, 0] = 1.2
    actions = scripted_learner_actions(
        obs.reshape(8, 1, 39), "los-intercept").reshape(8, 4)
    loss, head_accuracy, exact_accuracy, legal_count = (
        behavior_cloning_loss(model, obs, actions))
    assert torch.isfinite(loss)
    assert head_accuracy.shape == (4,)
    assert exact_accuracy.ndim == 0
    assert legal_count == 0


def test_mixed_teacher_opponents_preserve_static_batch_shape():
    obs = torch.zeros(6, 2, 39)
    obs[..., 14] = 1.0
    actions = teacher_opponent_actions(obs, "mixed")
    assert actions.shape == (6, 2, 4)
    assert actions[:, 0, 1].tolist() == [2, 4, 0, 2, 4, 0]
