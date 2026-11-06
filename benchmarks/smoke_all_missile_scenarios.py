"""Construct and step every configured physical-missile scenario once."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
from envs.JSBSim.envs.singlecombat_env import SingleCombatEnv


SINGLE_CONFIGS = (
    "1v1/ShootMissile/Selfplay",
    "1v1/ShootMissile/HierarchySelfplay",
    "1v1/ShootMissile/HierarchySelfplay_nolimit",
    "1v1/ShootMissile/HierarchyVsBaseline",
    "1v1/ShootMissile/HierarchyVsBaseline_nolimit",
    "1v1/DodgeMissile/Selfplay",
    "1v1/DodgeMissile/vsBaseline",
    "1v1/DodgeMissile/HierarchySelfplay",
    "1v1/DodgeMissile/HierarchyVsBaseline",
    "1v1/DodgeMissile/test/opposite",
    "1v1/DodgeMissile/test/pursue",
)

MULTIPLE_CONFIGS = (
    "2v2/ShootMissile/HierarchySelfplay",
    "2v2/ShootMissile/PBGR_R1",
    "2v2/ShootMissile/PBGR_P5_R1",
    "2v2/ShootMissile/PBGR_BRIC_V20",
    "2v2/ShootMissile/PBGR_V22_PLATFORM_CORRECT",
)


def hold_action(space):
    if isinstance(space, spaces.MultiDiscrete):
        action = np.asarray((space.nvec - 1) // 2, dtype=np.int64)
        if action.size and int(space.nvec[-1]) == 2:
            action[-1] = 0
        return action
    if isinstance(space, spaces.Tuple):
        values = []
        for component in space.spaces:
            if isinstance(component, spaces.MultiDiscrete):
                values.extend(((component.nvec - 1) // 2).tolist())
            elif isinstance(component, spaces.Discrete):
                values.append(0)
            else:
                raise TypeError("unsupported action component: {0}".format(
                    type(component).__name__))
        return np.asarray(values, dtype=np.int64)
    raise TypeError("unsupported action space: {0}".format(
        type(space).__name__))


def run_case(env_type, config):
    env = env_type(config)
    try:
        env.seed(20261103)
        env.reset()
        action = hold_action(env.action_space)
        result = env.step([action.copy() for _ in range(env.num_agents)])
        arrays = result[:-1]
        if not all(np.all(np.isfinite(np.asarray(value))) for value in arrays):
            raise AssertionError("non-finite environment output")
        print("PASS", config, "step", result[-1]["current_step"])
    finally:
        env.close()


def main():
    for config in SINGLE_CONFIGS:
        run_case(SingleCombatEnv, config)
    for config in MULTIPLE_CONFIGS:
        run_case(MultipleCombatEnv, config)


if __name__ == "__main__":
    main()
