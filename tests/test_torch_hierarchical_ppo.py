import pytest
import torch

from benchmarks.train_ppo_1v1_shoot_smoke import MultiDiscreteActorCritic
from benchmarks.train_torch_wvr_hierarchical_ppo import (
    apply_fire_controller,
    build_gate_observation,
    gate_teacher_loss,
    gate_anchor_kl_loss,
    normalize_advantages_by_group,
    quality_gated_fire,
    scheduled_teacher_coefficient,
    select_skill_actions,
    semi_markov_gae,
    semi_markov_td_target,
)


def test_select_skill_actions_uses_complete_option_action():
    base = torch.tensor([[0, 1, 2, 0], [0, 1, 2, 0]])
    intercept = torch.tensor([[2, 3, 0, 1], [2, 3, 0, 1]])
    selected = select_skill_actions(
        torch.tensor([0, 1]), base, intercept)
    assert torch.equal(selected[0], base[0])
    assert torch.equal(selected[1], intercept[1])


def test_select_skill_actions_rejects_misaligned_batch():
    with pytest.raises(ValueError, match="one choice per environment"):
        select_skill_actions(
            torch.tensor([0]), torch.zeros(2, 4, dtype=torch.long),
            torch.ones(2, 4, dtype=torch.long))


def test_semi_markov_td_target_bootstraps_only_unfinished_options():
    target = semi_markov_td_target(
        reward_sum=torch.tensor([1.0, 2.0]),
        continuation_discount=torch.tensor([0.9, 0.8]),
        next_value=torch.tensor([3.0, 4.0]),
        done=torch.tensor([False, True]))
    assert torch.allclose(target, torch.tensor([3.7, 2.0]))


def test_semi_markov_gae_respects_duration_and_episode_end():
    advantages, returns = semi_markov_gae(
        reward_sum=torch.tensor([1.0, 2.0]),
        continuation_discount=torch.tensor([0.9, 0.9]),
        duration=torch.tensor([1, 1]),
        next_value=torch.tensor([0.25, 0.0]),
        old_value=torch.tensor([0.5, 0.25]),
        done=torch.tensor([False, True]),
        env_ids=torch.tensor([0, 0]),
        gae_lambda=0.5)
    assert torch.allclose(advantages, torch.tensor([1.5125, 1.75]))
    assert torch.allclose(returns, torch.tensor([2.0125, 2.0]))


def test_quality_gated_fire_requires_stable_close_geometry():
    obs = torch.zeros(4, 39)
    obs[:, 23] = 1.0
    obs[:, 13] = torch.tensor([0.8, 1.2, 0.8, 0.8])
    obs[:, 11] = torch.tensor([0.2, 0.2, 0.5, 0.2])
    obs[:, 26] = torch.tensor([0.05, 0.05, 0.05, 0.2])
    obs[:, 28] = 0.2
    assert quality_gated_fire(obs).tolist() == [1, 0, 0, 0]


def test_shared_fire_controller_preserves_maneuver_commands():
    obs = torch.zeros(2, 39)
    obs[:, 23] = 1.0
    obs[:, 13] = 0.8
    obs[:, 11] = 0.2
    obs[:, 28] = 0.2
    selected = torch.tensor([[2, 4, 0, 0], [0, 1, 2, 1]])
    base = torch.tensor([[1, 2, 1, 1], [1, 2, 1, 0]])
    result = apply_fire_controller(
        selected, obs, base, "quality-gated",
        max_range_norm=1.0, max_ao_rad=0.35,
        max_los_rate=0.12, min_closure_norm=-0.1)
    assert torch.equal(result[:, :3], selected[:, :3])
    assert result[:, 3].tolist() == [1, 1]


def test_option_aware_fire_preserves_base_and_gates_intercept():
    obs = torch.zeros(2, 39)
    obs[:, 23] = 1.0
    obs[:, 13] = torch.tensor([1.2, 1.2])
    obs[:, 11] = 0.2
    obs[:, 28] = 0.2
    selected = torch.tensor([[1, 2, 1, 0], [1, 2, 1, 1]])
    base = torch.tensor([[1, 2, 1, 1], [1, 2, 1, 1]])
    result = apply_fire_controller(
        selected, obs, base, "option-aware", options=torch.tensor([0, 1]),
        max_range_norm=1.0, max_ao_rad=0.35,
        max_los_rate=0.12, min_closure_norm=-0.1)
    assert result[:, 3].tolist() == [1, 0]


def test_gate_observation_adds_masked_delta_and_previous_option():
    current = torch.tensor([[2.0, 4.0], [3.0, 5.0]])
    previous = torch.tensor([[1.0, 1.0], [9.0, 9.0]])
    result = build_gate_observation(
        current, previous, torch.tensor([1, -1]),
        torch.tensor([True, False]))
    assert result.shape == (2, 7)
    assert torch.equal(result[0, 2:4], torch.tensor([1.0, 3.0]))
    assert torch.equal(result[0, 4:6], torch.tensor([0.0, 1.0]))
    assert torch.equal(result[1, 2:], torch.zeros(5))


def test_groupwise_advantage_normalization_balances_opponent_scales():
    normalized = normalize_advantages_by_group(
        torch.tensor([1.0, 3.0, 10.0, 14.0]),
        torch.tensor([0, 0, 1, 1]))
    assert torch.allclose(normalized, torch.tensor([-1.0, 1.0, -1.0, 1.0]))


def test_gate_teacher_loss_ignores_boundaries_without_history():
    model = MultiDiscreteActorCritic(
        obs_dim=7, action_dims=[2], hidden_size=8)
    observations = torch.zeros(3, 7)
    observations[1:, -1] = 1.0
    loss = gate_teacher_loss(
        model, observations, torch.tensor([1, 0, 1]))
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_teacher_schedule_leaves_terminal_pure_ppo_phase():
    assert scheduled_teacher_coefficient(1.0, 0.0, 0.0, 0.5) == 1.0
    assert scheduled_teacher_coefficient(1.0, 0.0, 0.25, 0.5) == 0.5
    assert scheduled_teacher_coefficient(1.0, 0.0, 0.5, 0.5) == 0.0
    assert scheduled_teacher_coefficient(1.0, 0.0, 1.0, 0.5) == 0.0


def test_teacher_schedule_rejects_invalid_cutoff():
    with pytest.raises(ValueError, match="cutoff fraction"):
        scheduled_teacher_coefficient(1.0, 0.0, 0.1, 0.0)


def test_gate_anchor_kl_is_zero_for_identical_policy_and_positive_after_change():
    model = MultiDiscreteActorCritic(obs_dim=7, action_dims=[2], hidden_size=8)
    anchor = MultiDiscreteActorCritic(obs_dim=7, action_dims=[2], hidden_size=8)
    anchor.load_state_dict(model.state_dict())
    observations = torch.randn(16, 7)
    assert torch.allclose(
        gate_anchor_kl_loss(model, anchor, observations), torch.tensor(0.0),
        atol=1.0e-7)
    with torch.no_grad():
        next(model.parameters()).add_(0.2)
    assert gate_anchor_kl_loss(model, anchor, observations) > 0.0
