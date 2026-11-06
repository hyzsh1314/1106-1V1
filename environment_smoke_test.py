"""One-step, no-training smoke tests for the extracted WVR environment."""

from __future__ import annotations

import numpy as np

from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
from envs.JSBSim.envs.singlecombat_env import SingleCombatEnv
from envs.JSBSim.envs.singlecontrol_env import SingleControlEnv


def _one_step_single_control() -> None:
    env = SingleControlEnv("1/heading")
    try:
        obs = env.reset()
        action = np.asarray([env.action_space.sample() for _ in range(env.num_agents)])
        next_obs, reward, done, info = env.step(action)
        assert obs.shape == next_obs.shape
        assert np.isfinite(next_obs).all() and np.isfinite(reward).all()
        assert isinstance(info, dict)
    finally:
        env.close()


def _one_step_1v1() -> None:
    env = SingleCombatEnv("1v1/NoWeapon/Selfplay")
    try:
        obs = env.reset()
        action = np.asarray([env.action_space.sample() for _ in range(env.num_agents)])
        next_obs, reward, done, info = env.step(action)
        assert obs.shape == next_obs.shape
        assert np.isfinite(next_obs).all() and np.isfinite(reward).all()
        assert done.shape == (env.num_agents, 1)
        assert isinstance(info, dict)
    finally:
        env.close()


def _one_step_2v2_no_weapon() -> None:
    env = MultipleCombatEnv("2v2/NoWeapon/PBGR_R6_Entity_NoWeapon")
    try:
        obs, state = env.reset()
        action = np.asarray([env.action_space.sample() for _ in range(env.num_agents)])
        next_obs, next_state, reward, done, info = env.step(action)
        assert obs.shape == next_obs.shape == (4, 144)
        assert state.shape == next_state.shape == (4, 576)
        assert np.isfinite(next_obs).all() and np.isfinite(next_state).all()
        assert np.isfinite(reward).all() and done.shape == (4, 1)
        assert isinstance(info, dict)
    finally:
        env.close()


def _one_step_2v2_weapon() -> None:
    env = MultipleCombatEnv("2v2/ShootMissile/PBGR_V22_PLATFORM_CORRECT")
    try:
        obs, state = env.reset()
        action = np.asarray([env.action_space.sample() for _ in range(env.num_agents)])
        next_obs, next_state, reward, done, info = env.step(action)
        assert obs.shape == next_obs.shape == (4, 288)
        assert state.shape == next_state.shape == (4, 1152)
        assert np.isfinite(next_obs).all() and np.isfinite(next_state).all()
        assert np.isfinite(reward).all() and done.shape == (4, 1)
        assert isinstance(info, dict)
    finally:
        env.close()


def main() -> None:
    _one_step_single_control()
    _one_step_1v1()
    _one_step_2v2_no_weapon()
    _one_step_2v2_weapon()
    print("WVR environment smoke test: PASS (no training performed)")


if __name__ == "__main__":
    main()

