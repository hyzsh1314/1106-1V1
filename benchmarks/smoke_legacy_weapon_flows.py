"""Exercise representative launch flows in the retained weapon scenarios."""

from __future__ import annotations

import numpy as np

from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
from envs.JSBSim.envs.singlecombat_env import SingleCombatEnv


def assert_advanced_missiles(env, expected):
    assert len(env._tempsims) == expected
    assert all(
        missile.is_done or missile._t > 0.0
        for missile in env._tempsims.values()
    )


def smoke_singlecombat_shoot():
    env = SingleCombatEnv("1v1/ShootMissile/Selfplay")
    try:
        env.seed(20261103)
        env.reset()
        fire = np.asarray([20, 20, 20, 15, 1], dtype=np.int64)
        env.step([fire.copy(), fire.copy()])
        assert_advanced_missiles(env, 2)
        assert all(agent.num_left_missiles == 1 for agent in env.agents.values())
        print("PASS singlecombat_shoot launch-before-dynamics")
    finally:
        env.close()


def smoke_singlecombat_dodge():
    env = SingleCombatEnv("1v1/DodgeMissile/Selfplay")
    try:
        env.seed(20261103)
        env.reset()
        # Isolate the rule-launch pipeline from scenario-specific WEZ values.
        env.task.max_attack_angle = 180.0
        env.task.max_attack_distance = np.inf
        env.task.min_attack_interval = 0
        hold = np.asarray([20, 20, 20, 15], dtype=np.int64)
        for _ in range(env.task.lock_duration["A0100"].maxlen):
            env.step([hold.copy(), hold.copy()])
        assert_advanced_missiles(env, 2)
        print("PASS singlecombat_dodge rule-launch-before-dynamics")
    finally:
        env.close()


def smoke_multiplecombat(config, action):
    env = MultipleCombatEnv(config)
    try:
        env.seed(20261103)
        if getattr(env, "_initial_geometry_protocol", None) is not None:
            env.set_initial_geometry_reset_index(0)
        env.reset()
        env.step([action.copy() for _ in range(4)])
        assert_advanced_missiles(env, 4)
        assert all(agent.num_left_missiles == 1 for agent in env.agents.values())
        print("PASS", config, "launch-before-dynamics")
    finally:
        env.close()


def main():
    smoke_singlecombat_shoot()
    smoke_singlecombat_dodge()
    smoke_multiplecombat(
        "2v2/ShootMissile/HierarchySelfplay",
        np.asarray([1, 2, 1, 1], dtype=np.int64),
    )
    smoke_multiplecombat(
        "2v2/ShootMissile/PBGR_R1",
        np.asarray([1, 2, 1, 1], dtype=np.int64),
    )


if __name__ == "__main__":
    main()
