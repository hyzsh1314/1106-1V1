"""Regression checks for the unified weapon reward contracts."""

from types import SimpleNamespace

import numpy as np

from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
from envs.JSBSim.envs.singlecombat_env import SingleCombatEnv
from envs.JSBSim.reward_functions.altitude_reward import AltitudeReward


class _AltitudeAgent(object):
    def get_position(self):
        return np.asarray((0.0, 0.0, 3000.0), dtype=np.float64)

    def get_velocity(self):
        return np.asarray((0.0, 0.0, 0.0), dtype=np.float64)


def test_altitude_cost_scales_with_physical_decision_interval():
    config = SimpleNamespace(
        AltitudeReward_safe_altitude=4.0,
        AltitudeReward_danger_altitude=3.5,
        AltitudeReward_Kv=0.2,
        AltitudeReward_penalty_version="continuous_v1",
        AltitudeReward_time_scaled=True,
    )
    agent = _AltitudeAgent()
    fast_env = SimpleNamespace(
        agents={"A": agent}, time_interval=0.1)
    slow_env = SimpleNamespace(
        agents={"A": agent}, time_interval=0.2)

    fast = AltitudeReward(config).get_reward(None, fast_env, "A")
    slow = AltitudeReward(config).get_reward(None, slow_env, "A")

    assert np.isclose(slow, 2.0 * fast)


def test_singlecombat_reward_contracts_are_task_specific():
    shoot = SingleCombatEnv("1v1/ShootMissile/Selfplay")
    dodge = SingleCombatEnv("1v1/DodgeMissile/Selfplay")
    try:
        assert [type(item).__name__ for item in shoot.task.reward_functions] == [
            "PBGRDiscountedPostureReward",
            "AltitudeReward",
            "ShootPenaltyReward",
        ]
        assert [type(item).__name__ for item in dodge.task.reward_functions] == [
            "MissilePostureReward",
            "AltitudeReward",
            "MissileEvasionReward",
        ]
    finally:
        shoot.close()
        dodge.close()


def test_shooting_scenarios_emit_zero_sum_shaping_contract():
    single = SingleCombatEnv("1v1/ShootMissile/Selfplay")
    multiple = MultipleCombatEnv("2v2/ShootMissile/HierarchySelfplay")
    try:
        single.seed(20261103)
        single.reset()
        single_hold = np.asarray([20, 20, 20, 15, 0], dtype=np.int64)
        single_result = single.step([single_hold.copy(), single_hold.copy()])
        assert np.isclose(float(np.sum(single_result[1])), 0.0)
        assert single_result[3]["formal_reward_contract"][
            "zero_sum_shaping"] is True

        multiple.seed(20261103)
        multiple.reset()
        multiple_hold = np.asarray([1, 2, 1, 0], dtype=np.int64)
        multiple_result = multiple.step(
            [multiple_hold.copy() for _ in range(4)])
        assert np.isclose(float(np.sum(multiple_result[2])), 0.0)
        assert multiple_result[4]["formal_reward_contract"][
            "zero_sum_shaping"] is True
    finally:
        single.close()
        multiple.close()
