import torch

from benchmarks.benchmark_torch6dof_ppo import lowlevel_controls
from benchmarks.train_torch_wvr_ppo import (
    curriculum_gate_metrics,
    gated_opponent_kind,
    reward_variant_config,
    los_teacher_auxiliary_loss,
    staged_opponent_kind,
)


class ConstantLowLevel(torch.nn.Module):
    def forward(self, inputs, rnn_states):
        action = torch.full((inputs.shape[0], 4), 20.0,
                            device=inputs.device)
        return action, rnn_states


def test_torch_speed_adapter_orders_commands_and_high_speed_trim():
    policy = ConstantLowLevel()
    obs = torch.zeros(3, 1, 39)
    obs[:, 0, 8] = torch.tensor([250.0, 250.0, 300.0]) / 340.0
    action = torch.tensor([
        [[1, 2, 0, 0]],  # accelerate at 250 m/s
        [[1, 2, 2, 0]],  # decelerate at 250 m/s
        [[1, 2, 1, 0]],  # neutral at 300 m/s
    ])
    rnn = torch.zeros(3, 1, 128)
    legacy, _ = lowlevel_controls(policy, rnn, obs, action)
    adapted, _ = lowlevel_controls(
        policy, rnn, obs, action, torch_speed_adapter=True)
    assert adapted[0, 3] > adapted[1, 3]
    assert adapted[2, 3] < legacy[2, 3] - 0.4
    torch.testing.assert_close(adapted[:, :3], legacy[:, :3])


def test_metric_gate_requires_competence_fire_quality_and_repetition_input():
    failed = curriculum_gate_metrics(
        0, [0, 24, 0], [0, 13, 0], [0, 8, 0],
        accepted_launches=6, fire_requests=10, rejected_requests=4,
        min_episodes_per_kind=24)
    assert not failed["passed"]
    passed = curriculum_gate_metrics(
        0, [0, 24, 0], [0, 13, 0], [0, 8, 0],
        accepted_launches=9, fire_requests=10, rejected_requests=1,
        min_episodes_per_kind=24)
    assert passed["passed"]


def test_timeout_gate_does_not_accept_stalling_policy():
    result = curriculum_gate_metrics(
        0, [0, 100, 0], [0, 60, 0], [0, 60, 0],
        accepted_launches=90, fire_requests=100, rejected_requests=0,
        min_episodes_per_kind=24)
    assert not result["passed"]
    assert result["win_rates_by_kind"] == [0.6]
    assert result["timeout_rates_by_kind"] == [0.6]


def test_gated_opponent_mixture_expands_without_changing_batch_shape():
    index = torch.arange(6)
    kinds0, label0 = gated_opponent_kind(index, 0)
    kinds1, label1 = gated_opponent_kind(index, 1)
    kinds2, label2 = gated_opponent_kind(index, 2)
    assert label0 == "pursuit" and kinds0.tolist() == [1] * 6
    assert label1 == "pursuit+hold" and kinds1.tolist() == [1, 0] * 3
    assert label2 == "pursuit+hold+evasive"
    assert kinds2.tolist() == [1, 0, 2, 1, 0, 2]


def test_reward_variants_change_only_the_reward_contract():
    name, competitive = reward_variant_config("gated", "competitive")
    assert name == "competitive"
    assert competitive == {
        "reward_mode": "competitive", "attack_quality_weight": 0.0}
    name, attack = reward_variant_config("gated", "attack-quality")
    assert name == "attack-quality"
    assert attack["reward_mode"] == "engagement"
    assert attack["timeout_penalty"] == 1.0
    assert attack["attack_quality_weight"] == 0.5


def test_auto_reward_uses_ablation_selected_competitive_baseline():
    for opponents in ("selfplay", "fixed-mix", "staged", "gated", "evasive"):
        name, config = reward_variant_config(opponents, "auto")
        assert name == "competitive"
        assert config == {
            "reward_mode": "competitive", "attack_quality_weight": 0.0}


def test_los_teacher_auxiliary_loss_is_finite():
    from benchmarks.train_ppo_1v1_shoot_smoke import MultiDiscreteActorCritic

    model = MultiDiscreteActorCritic(
        obs_dim=39, action_dims=[3, 5, 3, 2], hidden_size=16,
        fire_mask_obs_index=23, altitude_safety_obs_index=0,
        altitude_descend_mask_threshold=0.8,
        altitude_force_climb_threshold=0.6)
    observations = torch.zeros(12, 39)
    observations[:, 0] = 1.2
    loss = los_teacher_auxiliary_loss(model, observations)
    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_fixed_stage_schedule_uses_empirical_difficulty_order():
    index = torch.arange(6)
    first, first_label = staged_opponent_kind(index, 1, 9)
    second, second_label = staged_opponent_kind(index, 4, 9)
    third, third_label = staged_opponent_kind(index, 7, 9)
    assert first_label == "pursuit" and first.tolist() == [1] * 6
    assert second_label == "pursuit+hold"
    assert second.tolist() == [1, 0, 1, 0, 1, 0]
    assert third_label == "pursuit+hold+evasive"
    assert third.tolist() == [1, 0, 2, 1, 0, 2]
