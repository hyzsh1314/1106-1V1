"""Fail-closed audit for the formal 1v1 population-PPO baseline."""

from __future__ import annotations

import argparse
import json

import numpy as np

from benchmarks.train_ppo_1v1_shoot_smoke import make_environment
from envs.JSBSim.utils.utils import LLA2NEU


STATE_KEYS = (
    "ic_long_gc_deg", "ic_lat_geod_deg", "ic_h_sl_ft",
    "ic_psi_true_deg", "ic_u_fps", "ic_p_rad_sec",
    "ic_q_rad_sec", "ic_r_rad_sec",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=85)
    parser.add_argument("--resets", type=int, default=256)
    return parser.parse_args()


def states_after_reset(env):
    env.reset()
    return {
        str(uid): {key: float(sim.init_state[key]) for key in STATE_KEYS}
        for uid, sim in env.agents.items()
    }


def state_matrix(states):
    return np.asarray([
        [states[uid][key] for key in STATE_KEYS]
        for uid in sorted(states)
    ], dtype=np.float64)


def main():
    args = parse_args()
    if args.resets < 32:
        raise ValueError("at least 32 resets are required for the audit")

    env = make_environment(args.seed)
    replica = make_environment(args.seed)
    different_seed = make_environment(args.seed + 1)
    try:
        config = env.config
        agent_ids = [str(uid) for uid in env.agents]
        if agent_ids != ["A0100", "B0100"]:
            raise AssertionError("unexpected or reordered 1v1 agent ids")
        if [env.agents[uid].color for uid in env.agents] != ["Red", "Blue"]:
            raise AssertionError("red/blue identities do not match the config")
        if any(env.agents[uid].model != "f16" for uid in env.agents):
            raise AssertionError("aircraft models are not symmetric")
        if any(env.agents[uid].num_missiles != 2 for uid in env.agents):
            raise AssertionError("missile loads are not symmetric")
        if env.sim_freq != 60 or env.agent_interaction_steps != 12:
            raise AssertionError("the 0.2 s decision interval changed")
        if env.max_steps != 1000:
            raise AssertionError("the 200 s episode horizon changed")
        if str(config.task) != "hierarchical_singlecombat_shoot":
            raise AssertionError("unexpected 1v1 task implementation")
        if (float(env.task.max_attack_angle) != 45.0 or
                float(env.task.max_attack_distance) != 14000.0 or
                int(env.task.min_attack_interval) != 25):
            raise AssertionError("weapon launch envelope changed")
        if not bool(config.FormalRewardContract_enabled):
            raise AssertionError("formal zero-sum reward contract is disabled")
        if bool(config.ResolvePendingMissilesAfterElimination_enabled):
            raise AssertionError("the 1v1 benchmark must use first-kill termination")
        if float(config.PPO_gamma) != float(config.PotentialReward_gamma):
            raise AssertionError("PPO and PBRS discount factors differ")
        if env.task.lowlevel_policy.training or any(
                parameter.requires_grad
                for parameter in env.task.lowlevel_policy.parameters()):
            raise AssertionError("the hierarchical low-level policy is not frozen")

        samples = {uid: [] for uid in agent_ids}
        engagement_ranges = []
        reproducible = True
        distinct_seed_seen = False
        for _ in range(args.resets):
            states = states_after_reset(env)
            replica_states = states_after_reset(replica)
            other_states = states_after_reset(different_seed)
            reproducible &= np.array_equal(
                state_matrix(states), state_matrix(replica_states))
            distinct_seed_seen |= not np.array_equal(
                state_matrix(states), state_matrix(other_states))
            for uid in agent_ids:
                samples[uid].append([states[uid][key] for key in STATE_KEYS])
            local_positions = []
            for uid in agent_ids:
                state = states[uid]
                local_positions.append(LLA2NEU(
                    state["ic_long_gc_deg"], state["ic_lat_geod_deg"],
                    state["ic_h_sl_ft"] * 0.3048,
                    env.center_lon, env.center_lat, env.center_alt))
            engagement_ranges.append(float(np.linalg.norm(
                local_positions[1] - local_positions[0])))

        if not reproducible or not distinct_seed_seen:
            raise AssertionError("seed reproducibility or seed diversity failed")
        arrays = {
            uid: np.asarray(values, dtype=np.float64)
            for uid, values in samples.items()
        }
        combined = np.concatenate(list(arrays.values()), axis=0)
        limits = {
            "ic_h_sl_ft": (16000.0, 25000.0),
            "ic_psi_true_deg": (0.0, 360.0),
            "ic_u_fps": (600.0, 900.0),
            "ic_p_rad_sec": (-0.01, 0.01),
            "ic_q_rad_sec": (-0.01, 0.01),
            "ic_r_rad_sec": (-0.01, 0.01),
        }
        for key, (lower, upper) in limits.items():
            column = combined[:, STATE_KEYS.index(key)]
            if np.any(column < lower) or np.any(column > upper):
                raise AssertionError("{} randomization is out of bounds".format(key))
            if float(np.std(column)) <= 0.0:
                raise AssertionError("{} is not randomized".format(key))
        if not all(7900.0 <= value <= 16100.0
                   for value in engagement_ranges):
            raise AssertionError("relative engagement range is out of bounds")

        # A deterministic no-op flight command is sufficient to validate the
        # packed observation/action shapes, finite outputs and zero-sum reward.
        obs = env.reset()
        action = np.tile(np.asarray([1, 2, 1, 0], dtype=np.int64), (2, 1))
        next_obs, reward, done, info = env.step(action)
        if obs.shape != (2, 24) or next_obs.shape != (2, 24):
            raise AssertionError("unexpected observation shape")
        if not np.all(np.isfinite(next_obs)) or not np.all(np.isfinite(reward)):
            raise AssertionError("non-finite environment output")
        if abs(float(np.asarray(reward).sum())) > 1.0e-6:
            raise AssertionError("formal reward is not zero-sum")

        report = {
            "status": "PASS",
            "seed": args.seed,
            "resets": args.resets,
            "agents": agent_ids,
            "decision_interval_s": env.time_interval,
            "episode_horizon_s": env.max_steps * env.time_interval,
            "observation_shape": list(obs.shape),
            "action_dims": [3, 5, 3, 2],
            "engagement_range_m": {
                "minimum": min(engagement_ranges),
                "maximum": max(engagement_ranges),
                "mean": float(np.mean(engagement_ranges)),
            },
            "randomization_min": {
                key: float(combined[:, index].min())
                for index, key in enumerate(STATE_KEYS)
            },
            "randomization_max": {
                key: float(combined[:, index].max())
                for index, key in enumerate(STATE_KEYS)
            },
            "seed_reproducible": bool(reproducible),
            "different_seed_distinct": bool(distinct_seed_seen),
            "zero_sum_step_error": abs(float(np.asarray(reward).sum())),
            "reward_contract": str(config.FormalRewardContract_version),
            "weapon_envelope": {
                "max_attack_angle_deg": float(env.task.max_attack_angle),
                "max_attack_distance_m": float(env.task.max_attack_distance),
                "min_attack_interval_steps": int(env.task.min_attack_interval),
                "missiles_per_aircraft": 2,
            },
            "lowlevel_frozen": True,
            "info_contract_present": "formal_reward_contract" in info,
            "done_shape": list(np.asarray(done).shape),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        env.close()
        replica.close()
        different_seed.close()


if __name__ == "__main__":
    main()
