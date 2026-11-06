import numpy as np
import pytest
import torch

from benchmarks.evaluate_ppo_1v1_acceptance import (
    scripted_evasive_fire,
    scripted_evasive_fire_break_skill,
    scripted_hold,
    scripted_standoff_fire,
)
from benchmarks.train_ppo_1v1_shoot_smoke import (
    DEFAULT_REWARD_VARIANT,
    make_environment,
)
from benchmarks.train_ppo_1v1_shoot_smoke import MultiDiscreteActorCritic
from benchmarks.train_ppo_population_1v1 import (
    HistoricalPolicyPopulation,
    anchor_fraction_for_update,
    build_opponent_schedule,
    resume_configs_compatible,
)


EXPECTED = {
    "R0_terminal": [],
    "R1_attack_pbrs": ["PBGRDiscountedPostureReward"],
    "R1_timeout025": ["PBGRDiscountedPostureReward"],
    "R2_threat_pbrs": ["ThreatConditionedPotentialReward"],
    "R3_constraints": [
        "ThreatConditionedPotentialReward",
        "AltitudeReward",
        "ShootPenaltyReward",
    ],
}


def test_direct_formal_scenario_defaults_to_selected_r1_contract():
    env = make_environment(8500)
    try:
        assert DEFAULT_REWARD_VARIANT == "R1_attack_pbrs"
        assert env.config.RewardVariant == DEFAULT_REWARD_VARIANT
        assert env.config.FormalRewardContract_version == (
            "WVR-1V1-V5-R1-attack-PBRS-timeout-zero")
        assert float(env.config.TeamOutcomeReward_scale) == 1.0
        assert float(env.config.TimeoutOutcomeReward_scale) == 0.0
        assert float(env.config.PostureReward_scale) == pytest.approx(0.1)
        assert float(env.config.AltitudeReward_scale) == 0.0
        assert float(env.config.ShootPenaltyReward_scale) == 0.0
        assert [type(item).__name__ for item in env.task.reward_functions] == [
            "PBGRDiscountedPostureReward"]
    finally:
        env.close()


def test_fixed_hold_opponent_never_requests_fire():
    observations = np.zeros((7, 24), dtype=np.float32)
    observations[:, 23] = 1.0
    actions = scripted_hold(observations)
    assert actions.shape == (7, 4)
    assert np.all(actions == np.asarray([1, 2, 1, 0]))


def test_strong_fixed_opponents_obey_action_and_fire_masks():
    observations = np.zeros((4, 24), dtype=np.float32)
    observations[:, 13] = np.asarray([0.7, 0.8, 1.1, 1.3])
    observations[:, 14] = np.asarray([-1.0, 1.0, -1.0, 1.0])
    observations[:, 19] = np.asarray([0.2, 0.3, 0.0, 0.0])
    observations[:, 20] = np.asarray([-1.0, 1.0, 0.0, 0.0])
    observations[:, 23] = np.asarray([1.0, 0.0, 1.0, 0.0])
    for policy in (scripted_evasive_fire, scripted_standoff_fire):
        actions = policy(observations)
        assert actions.shape == (4, 4)
        assert np.all(actions >= 0)
        assert np.all(actions < np.asarray([3, 5, 3, 2]))
        assert np.array_equal(actions[:, 3], observations[:, 23])
    evasive = scripted_evasive_fire(observations)
    assert evasive[0, 1] == 4
    assert evasive[1, 1] == 0
    standoff = scripted_standoff_fire(observations)
    assert standoff[0, 1] == 4
    assert standoff[1, 1] == 0


def test_population_schedule_reserves_balanced_scripted_anchors():
    population = HistoricalPolicyPopulation(capacity=4)
    model = MultiDiscreteActorCritic(
        obs_dim=24, action_dims=[3, 5, 3, 2], hidden_size=16,
        fire_mask_obs_index=23)
    population.add(model, source_update=0)
    historical, scripted = build_opponent_schedule(
        population, np.random.default_rng(85), count=96, max_unique=0,
        scripted_anchor_fraction=0.25,
        scripted_anchor_policies=(
            "pursuit_fire", "evasive_fire", "standoff_fire"))
    assert np.count_nonzero(historical >= 0) == 72
    assert np.count_nonzero(historical == -1) == 24
    assert np.count_nonzero(scripted == "") == 72
    assert {
        name: int(np.count_nonzero(scripted == name))
        for name in ("pursuit_fire", "evasive_fire", "standoff_fire")
    } == {
        "pursuit_fire": 8,
        "evasive_fire": 8,
        "standoff_fire": 8,
    }


def test_altitude_safety_mask_is_applied_inside_ppo_distribution():
    model = MultiDiscreteActorCritic(
        obs_dim=24, action_dims=[3, 5, 3, 2], hidden_size=16,
        fire_mask_obs_index=23, altitude_safety_obs_index=0,
        altitude_descend_mask_threshold=0.8,
        altitude_force_climb_threshold=0.65)
    observations = torch.zeros((3, 24), dtype=torch.float32)
    observations[:, 0] = torch.tensor([1.0, 0.75, 0.60])
    observations[:, 23] = 1.0
    with torch.no_grad():
        features = model._features(observations)
        altitude = model._distributions(observations, features)[0]
        samples = torch.stack([altitude.sample() for _ in range(256)])
        actions, log_prob, _value = model.act(observations)
        evaluated, _entropy, _value = model.evaluate_actions(
            observations, actions)
    assert torch.any(samples[:, 0] == 2)
    assert not torch.any(samples[:, 1] == 2)
    assert torch.all(samples[:, 2] == 0)
    assert torch.all(torch.isfinite(log_prob))
    assert torch.allclose(log_prob, evaluated)


def test_scripted_anchor_fraction_decays_deterministically():
    class Args:
        scripted_anchor_fraction = 0.25
        scripted_anchor_final_fraction = 0.0625
        scripted_anchor_decay_updates = 21

    assert anchor_fraction_for_update(Args, 1) == pytest.approx(0.25)
    assert anchor_fraction_for_update(Args, 11) == pytest.approx(0.15625)
    assert anchor_fraction_for_update(Args, 21) == pytest.approx(0.0625)
    assert anchor_fraction_for_update(Args, 99) == pytest.approx(0.0625)


def test_scripted_anchor_fraction_can_ramp_after_resume_boundary():
    class Args:
        scripted_anchor_fraction = 0.25
        scripted_anchor_final_fraction = 1.0
        scripted_anchor_decay_updates = 20
        scripted_anchor_ramp_start_update = 42

    assert anchor_fraction_for_update(Args, 41) == pytest.approx(0.25)
    assert anchor_fraction_for_update(Args, 42) == pytest.approx(0.25)
    assert anchor_fraction_for_update(Args, 51) == pytest.approx(
        0.25 + 9.0 / 19.0 * 0.75)
    assert anchor_fraction_for_update(Args, 61) == pytest.approx(1.0)


def test_resume_curriculum_override_preserves_immutable_contract():
    saved = {"seed": 85, "gamma": 0.99}
    curriculum = {
        "seed": 85, "gamma": 0.99,
        "opponent_curriculum": {"scripted_anchor_fraction": 1.0},
    }
    changed_ppo = {
        "seed": 85, "gamma": 0.95,
        "opponent_curriculum": {"scripted_anchor_fraction": 1.0},
    }
    assert not resume_configs_compatible(saved, curriculum)
    assert resume_configs_compatible(
        saved, curriculum, allow_curriculum_override=True)
    assert not resume_configs_compatible(
        saved, changed_ppo, allow_curriculum_override=True)


def test_resume_action_interface_override_preserves_immutable_contract():
    saved = {"seed": 85, "gamma": 0.99}
    wider = {
        "seed": 85, "gamma": 0.99,
        "action_interface": {"heading_command_span_deg": 90.0},
    }
    changed_ppo = {
        "seed": 85, "gamma": 0.95,
        "action_interface": {"heading_command_span_deg": 90.0},
    }
    assert not resume_configs_compatible(saved, wider)
    assert resume_configs_compatible(
        saved, wider, allow_action_interface_override=True)
    assert not resume_configs_compatible(
        saved, changed_ppo, allow_action_interface_override=True)


def test_resume_accepts_legacy_implicit_disabled_break_defaults():
    saved = {
        "seed": 85,
        "action_interface": {
            "heading_command_span_deg": 90.0,
            "heading_command_bins_deg": [-90.0, -45.0, 0.0, 45.0, 90.0],
        },
    }
    current = {
        "seed": 85,
        "action_interface": {
            **saved["action_interface"],
            "threat_relative_break_skill": False,
            "threat_break_option": False,
            "threat_break_min_steps": 10,
            "threat_break_clear_steps": 5,
            "threat_break_recovery_steps": 10,
            "threat_break_recovery_ta_max_deg": 180.0,
        },
    }
    assert resume_configs_compatible(saved, current)
    changed = {
        **current,
        "action_interface": {
            **current["action_interface"], "threat_break_option": True,
        },
    }
    assert not resume_configs_compatible(saved, changed)


def test_resume_reward_override_is_limited_to_reward_contract_fields():
    saved = {
        "seed": 85, "reward_variant": "R1_attack_pbrs",
        "environment_contract": {
            "reward_contract": "v5", "reward_variant": "R1_attack_pbrs",
            "timeout_outcome_scale": 0.0, "max_steps": 1000,
        },
    }
    timeout_reward = {
        "seed": 85, "reward_variant": "R1_timeout025",
        "environment_contract": {
            "reward_contract": "v6", "reward_variant": "R1_timeout025",
            "timeout_outcome_scale": 0.25, "max_steps": 1000,
        },
    }
    changed_dynamics = {
        **timeout_reward,
        "environment_contract": {
            **timeout_reward["environment_contract"], "max_steps": 500,
        },
    }
    assert resume_configs_compatible(
        saved, timeout_reward, allow_reward_override=True)
    assert not resume_configs_compatible(
        saved, changed_dynamics, allow_reward_override=True)


def test_timeout025_variant_maps_tactical_timeout_to_small_zero_sum_utility():
    env = make_environment(8511, reward_variant="R1_timeout025")
    try:
        assert env.config.TeamOutcomeReward_scale == pytest.approx(1.0)
        assert env.config.TimeoutOutcomeReward_scale == pytest.approx(0.25)
    finally:
        env.close()


def test_wide_heading_command_ablation_changes_only_heading_mapping():
    baseline = make_environment(8510, heading_command_span_deg=30.0)
    wider = make_environment(8510, heading_command_span_deg=90.0)
    try:
        np.testing.assert_allclose(
            np.rad2deg(baseline.task.norm_delta_heading),
            [-30.0, -15.0, 0.0, 15.0, 30.0])
        np.testing.assert_allclose(
            np.rad2deg(wider.task.norm_delta_heading),
            [-90.0, -45.0, 0.0, 45.0, 90.0])
        assert baseline.action_space == wider.action_space
    finally:
        baseline.close()
        wider.close()


def test_threat_relative_break_skill_uses_missile_side_only_under_warning():
    env = make_environment(
        8512, threat_observation_v2=True,
        heading_command_span_deg=90.0,
        threat_relative_break_skill=True)
    try:
        observation = np.zeros(26, dtype=np.float64)
        observation[20] = -1.0
        assert env.task._resolve_heading_command(
            observation, 0) == pytest.approx(-np.pi / 2)
        assert env.task._resolve_heading_command(
            observation, 4) == pytest.approx(np.pi / 2)

        observation[24] = 1.0
        observation[20] = 1.0
        assert env.task._resolve_heading_command(
            observation, 0) == pytest.approx(-np.pi / 2)
        assert env.task._resolve_heading_command(
            observation, 4) == pytest.approx(np.pi / 2)
        observation[20] = -1.0
        assert env.task._resolve_heading_command(
            observation, 0) == pytest.approx(np.pi / 2)
        assert env.task._resolve_heading_command(
            observation, 4) == pytest.approx(-np.pi / 2)
        assert env.task._resolve_heading_command(
            observation, 2) == pytest.approx(0.0)
    finally:
        env.close()


def test_break_skill_evasive_reference_invokes_away_skill_for_both_sides():
    observations = np.zeros((2, 26), dtype=np.float32)
    observations[:, 19] = 0.5
    observations[:, 20] = [-1.0, 1.0]
    observations[:, 24] = 1.0
    actions = scripted_evasive_fire_break_skill(observations)
    assert np.array_equal(actions[:, 1], [0, 0])


def test_threat_break_option_holds_break_then_reacquires_opponent():
    env = make_environment(
        8513, threat_observation_v2=True,
        heading_command_span_deg=90.0,
        threat_relative_break_skill=True,
        threat_break_option=True,
        threat_break_min_steps=3,
        threat_break_clear_steps=2,
        threat_break_recovery_steps=2)
    try:
        env.reset()
        agent_id = next(iter(env.agents))
        observation = np.zeros(26, dtype=np.float64)
        observation[11] = 0.4
        observation[14] = -1.0
        observation[20] = 1.0
        observation[24] = 1.0

        # Trigger plus the complete minimum-duration/clear-hysteresis phase.
        assert env.task._resolve_heading_command(
            observation, 0, agent_id) == pytest.approx(-np.pi / 2)
        observation[24] = 0.0
        for _ in range(2):
            assert env.task._resolve_heading_command(
                observation, 4, agent_id) == pytest.approx(-np.pi / 2)

        # Two recovery decisions turn toward the opponent, then control returns.
        assert env.task._resolve_heading_command(
            observation, 4, agent_id) == pytest.approx(-0.4)
        assert env.task._resolve_heading_command(
            observation, 0, agent_id) == pytest.approx(-0.4)
        assert env.task._resolve_heading_command(
            observation, 2, agent_id) == pytest.approx(0.0)
    finally:
        env.close()


def test_threat_break_option_requires_stateless_break_skill():
    with pytest.raises(ValueError, match="requires ThreatRelativeBreakSkill"):
        make_environment(
            8514, threat_observation_v2=True,
            threat_break_option=True)


def test_threat_break_pulse_is_urgent_bounded_and_interruptible():
    env = make_environment(
        8514, threat_observation_v2=True,
        heading_command_span_deg=90.0,
        threat_relative_break_skill=True,
        threat_break_pulse=True,
        threat_break_pulse_steps=4,
        threat_break_pulse_urgency=0.5)
    try:
        env.reset()
        agent_id = next(iter(env.agents))
        observation = np.zeros(26, dtype=np.float64)
        observation[20] = 1.0
        observation[24] = 1.0

        # A warning below the urgency gate keeps the stateless semantics and
        # does not latch a pulse.
        observation[25] = 0.49
        assert env.task._resolve_heading_command(
            observation, 0, agent_id) == pytest.approx(-np.pi / 2)
        assert env.task._threat_break_pulse_state[agent_id]["remaining"] == 0

        # Trigger plus three held decisions gives exactly four decisions.
        observation[25] = 0.75
        assert env.task._resolve_heading_command(
            observation, 0, agent_id) == pytest.approx(-np.pi / 2)
        observation[24] = 0.0
        for _ in range(3):
            assert env.task._resolve_heading_command(
                observation, 4, agent_id) == pytest.approx(-np.pi / 2)

        # No warning-clear wait and no recovery: PPO's physical command is
        # restored immediately after the fixed pulse.
        assert env.task._resolve_heading_command(
            observation, 4, agent_id) == pytest.approx(np.pi / 2)
    finally:
        env.close()


def test_threat_break_pulse_and_legacy_option_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        make_environment(
            8515, threat_observation_v2=True,
            threat_relative_break_skill=True,
            threat_break_option=True,
            threat_break_pulse=True)


def test_threat_break_option_skips_recovery_against_nose_on_opponent():
    env = make_environment(
        8515, threat_observation_v2=True,
        heading_command_span_deg=90.0,
        threat_relative_break_skill=True,
        threat_break_option=True,
        threat_break_min_steps=1,
        threat_break_clear_steps=1,
        threat_break_recovery_steps=2,
        threat_break_recovery_ta_max_deg=90.0)
    try:
        env.reset()
        agent_id = next(iter(env.agents))
        observation = np.zeros(26, dtype=np.float64)
        observation[11] = 0.4
        observation[12] = np.deg2rad(120.0)
        observation[14] = -1.0
        observation[20] = 1.0
        observation[24] = 1.0
        assert env.task._resolve_heading_command(
            observation, 0, agent_id) == pytest.approx(-np.pi / 2)

        # Once clear, the hostile nose-on geometry skips forced recovery and
        # returns the policy's physical +90-degree command immediately.
        observation[24] = 0.0
        assert env.task._resolve_heading_command(
            observation, 4, agent_id) == pytest.approx(np.pi / 2)
        assert env.task._threat_break_state[agent_id]["mode"] == "idle"
    finally:
        env.close()


def test_threat_observation_v2_preserves_legacy_indices():
    legacy = make_environment(8501, reward_variant="R2_threat_pbrs")
    enhanced = make_environment(
        8501, reward_variant="R2_threat_pbrs", threat_observation_v2=True)
    try:
        legacy_obs = legacy.reset()
        enhanced_obs = enhanced.reset()
        assert legacy_obs.shape == (2, 24)
        assert enhanced_obs.shape == (2, 26)
        np.testing.assert_allclose(enhanced_obs[:, :24], legacy_obs)
        np.testing.assert_allclose(enhanced_obs[:, 24:], 0.0)
    finally:
        legacy.close()
        enhanced.close()


@pytest.mark.parametrize("variant", tuple(EXPECTED))
def test_reward_variants_have_normalized_zero_sum_contract(variant):
    env = make_environment(8500, reward_variant=variant)
    try:
        obs = env.reset()
        assert obs.shape == (2, 24)
        assert float(env.config.TeamOutcomeReward_scale) == 1.0
        expected_timeout_scale = 0.25 if variant == "R1_timeout025" else 0.0
        assert float(env.config.TimeoutOutcomeReward_scale) == pytest.approx(
            expected_timeout_scale)
        assert [type(item).__name__ for item in env.task.reward_functions] == (
            EXPECTED[variant])
        env.max_steps = 1
        hold = np.asarray([[1, 2, 1, 0], [1, 2, 1, 0]], dtype=np.int64)
        _obs, reward, done, info = env.step(hold)
        assert np.all(done)
        assert np.isclose(float(np.sum(reward)), 0.0)
        assert "TeamOutcomeReward" in info["team_reward_components"]
        reconstructed = sum(
            float(values[0])
            for values in info["team_reward_components"].values())
        assert np.isclose(float(reward[0, 0]), reconstructed)
        if info["team_outcome"]["reason"].startswith("timeout"):
            expected = expected_timeout_scale * int(
                info["team_outcome"]["ego_result"])
            assert info["team_terminal_reward"] == pytest.approx(
                [expected, -expected])
    finally:
        env.close()


@pytest.mark.parametrize("variant", ("R1_attack_pbrs", "R2_threat_pbrs"))
def test_potential_shaping_is_bounded_and_auditable(variant):
    env = make_environment(85, reward_variant=variant)
    try:
        obs = env.reset()
        reward_name = EXPECTED[variant][0]
        for _ in range(32):
            actions = np.tile(
                np.asarray([1, 2, 1, 0], dtype=np.int64), (2, 1))
            obs, reward, done, info = env.step(actions)
            assert np.isclose(float(np.sum(reward)), 0.0)
            component = info["team_reward_components"].get(
                reward_name, [0.0, 0.0])
            assert abs(float(component[0])) <= 0.201
            assert np.isclose(float(component[0]), -float(component[1]))
            if np.all(done):
                break
    finally:
        env.close()


def test_r3_accepted_launch_cost_is_small_and_recorded_once():
    env = make_environment(85, reward_variant="R3_constraints")
    try:
        obs = env.reset()
        action = np.tile(np.asarray([1, 2, 1, 0]), (2, 1))
        action[:, 3] = (obs[:, 23] >= 0.5).astype(np.int64)
        _obs, reward, _done, info = env.step(action)
        assert info["weapon_rejected_requests"] == 0
        assert info["weapon_accepted_launches"] == int(action[:, 3].sum())
        raw = info["reward_components"]
        for agent_id, fired in zip(env.agents, action[:, 3]):
            expected = -0.01 if fired else 0.0
            assert np.isclose(
                raw[str(agent_id)]["ShootPenaltyReward"], expected)
        assert np.isclose(float(np.sum(reward)), 0.0)
    finally:
        env.close()
