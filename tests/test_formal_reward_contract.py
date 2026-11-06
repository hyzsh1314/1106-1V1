from types import SimpleNamespace

import numpy as np

from envs.JSBSim.envs import MultipleCombatEnv
from envs.JSBSim.reward_functions.altitude_reward import AltitudeReward
from envs.JSBSim.reward_functions.posture_reward import PostureReward
from envs.JSBSim.tasks.weapon_contract import (
    TeamOutcome, team_outcome_terminal_rewards,
)


class _Agent:
    def __init__(self, position, velocity=(0.0, 0.0, 0.0), alive=True):
        self._position = np.asarray(position, dtype=np.float64)
        self._velocity = np.asarray(velocity, dtype=np.float64)
        self.is_alive = alive
        self.enemies = []

    def get_position(self):
        return self._position

    def get_velocity(self):
        return self._velocity


def test_timeout_adjudication_is_not_a_weapon_kill_reward():
    config = SimpleNamespace(
        TeamOutcomeReward_scale=1.0,
        TimeoutOutcomeReward_scale=0.0,
    )
    tactical_win = TeamOutcome(
        "ego", 1, -1, "ego_win", "timeout_tactical_advantage",
        True, False, 1000, 1, 1, 0.4, 0.2,
    )
    weapon_win = TeamOutcome(
        "ego", 1, -1, "ego_win", "enemy_team_eliminated",
        True, False, 100, 1, 0,
    )

    assert team_outcome_terminal_rewards(config, tactical_win) == [0.0, -0.0]
    assert team_outcome_terminal_rewards(config, weapon_win) == [1.0, -1.0]


def test_altitude_penalty_is_continuous_and_matches_documented_range():
    config = SimpleNamespace(
        AltitudeReward_safe_altitude=4.0,
        AltitudeReward_danger_altitude=3.5,
        AltitudeReward_Kv=0.2,
        AltitudeReward_penalty_version="continuous_v1",
    )
    reward = AltitudeReward(config)
    agent = _Agent((0.0, 0.0, 3500.0))
    env = SimpleNamespace(agents={"A": agent})

    at_boundary = reward.get_reward(None, env, "A")
    agent._position[-1] = 3499.999
    just_below = reward.get_reward(None, env, "A")
    agent._position[-1] = 0.0
    at_sea_level = reward.get_reward(None, env, "A")

    assert at_boundary == 0.0
    assert -1e-6 < just_below < 0.0
    assert at_sea_level == -1.0


def test_altitude_penalty_keeps_legacy_default_for_old_scenarios():
    config = SimpleNamespace(
        AltitudeReward_safe_altitude=4.0,
        AltitudeReward_danger_altitude=3.5,
        AltitudeReward_Kv=0.2,
    )
    reward = AltitudeReward(config)
    agent = _Agent((0.0, 0.0, 3500.0))
    env = SimpleNamespace(agents={"A": agent})

    assert reward.get_reward(None, env, "A") == -1.0
    agent._position[-1] = 0.0
    assert reward.get_reward(None, env, "A") == -2.0


def test_posture_mean_alive_is_team_size_invariant():
    config = SimpleNamespace(PostureReward_enemy_aggregation="mean_alive")
    reward = PostureReward(config)
    reward.orientation_fn = lambda _ao, _ta: 2.0
    reward.range_fn = lambda _distance: 3.0
    ego = _Agent((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    ego.enemies = [
        _Agent((1000.0, 0.0, 0.0), alive=True),
        _Agent((2000.0, 0.0, 0.0), alive=True),
        _Agent((3000.0, 0.0, 0.0), alive=False),
    ]
    env = SimpleNamespace(agents={"A": ego})

    assert reward.get_reward(None, env, "A") == 6.0


def test_formal_no_weapon_timeout_is_explicit_zero_sum_draw():
    env = MultipleCombatEnv("2v2/NoWeapon/PBGR_R3_NoWeapon")
    try:
        env.reset()
        env.max_steps = 1
        actions = np.zeros((4, 4), dtype=np.int64)
        _obs, _central, rewards, dones, info = env.step(actions)

        assert np.all(dones)
        assert info["team_outcome"]["label"] == "draw"
        assert info["team_terminal_reward"] == [0.0, 0.0]
        assert info["formal_reward_contract"]["version"] == "R3-ZS-WDL-v1"
        assert np.isclose(rewards[:2].mean(), -rewards[2:].mean())
    finally:
        env.close()


def test_formal_terminal_win_reward_is_symmetric_and_explicit():
    env = MultipleCombatEnv("2v2/NoWeapon/PBGR_R3_NoWeapon")
    try:
        env.reset()
        for enemy_id in env.enm_ids:
            env.agents[enemy_id].crash()
        actions = np.zeros((4, 4), dtype=np.int64)
        _obs, _central, rewards, dones, info = env.step(actions)

        assert np.all(dones)
        assert info["team_outcome"]["label"] == "ego_win"
        assert info["team_terminal_reward"] == [200.0, -200.0]
        assert np.isclose(rewards[:2].mean(), -rewards[2:].mean())
        assert rewards[:2].mean() > 0.0
        assert rewards[2:].mean() < 0.0
    finally:
        env.close()
