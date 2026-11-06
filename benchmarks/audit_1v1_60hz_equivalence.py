"""Compare optimized 1v1 stepping with the pre-optimization 60 Hz path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import MethodType

import numpy as np

from benchmarks.train_ppo_1v1_shoot_smoke import (
    DEFAULT_REWARD_VARIANT,
    make_environment,
)
from envs.JSBSim.core.simulatior import MissileSimulator
from envs.env_wrappers import configure_worker_threads


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=(85, 185, 285))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def legacy_substep(env):
    """Exact implementation used before the no-temporary-simulator fast path."""

    target_segment_starts = {
        id(sim): np.asarray(sim.get_position(), dtype=np.float64).copy()
        for sim in env._jsbsims.values()
    }
    for sim in env._jsbsims.values():
        sim.run()
    deferred_missiles = []
    for sim in env._tempsims.values():
        if isinstance(sim, MissileSimulator):
            sim.run(
                defer_target_effect=True,
                target_segment_start=target_segment_starts.get(
                    id(sim.target_aircraft)),
            )
            deferred_missiles.append(sim)
        else:
            sim.run()
    MissileSimulator.resolve_deferred_target_effects(deferred_missiles)


def install_legacy_paths(env):
    env._run_simulation_substep = MethodType(legacy_substep, env)

    def sequential_normalize(environment, actions):
        # The legacy path rebuilt each agent observation instead of consuming
        # the observation returned at the preceding decision boundary.
        cached = getattr(environment, "_decision_observations", None)
        if hasattr(environment, "_decision_observations"):
            del environment._decision_observations
        try:
            return {
                agent_id: env.task.normalize_action(
                    environment, agent_id, actions[agent_id])
                for agent_id in environment.agents.keys()
            }
        finally:
            if cached is not None:
                environment._decision_observations = cached

    env.task.normalize_actions = sequential_normalize


def deterministic_actions(rng, step):
    actions = np.empty((2, 4), dtype=np.int64)
    actions[:, 0] = rng.integers(0, 3, size=2)
    actions[:, 1] = rng.integers(0, 5, size=2)
    actions[:, 2] = rng.integers(0, 3, size=2)
    # Repeated requests exercise legal launches, rejected requests, guidance,
    # hits/misses, and the no-missile fast path in the same audit.
    actions[:, 3] = int(step % 3 != 0)
    return actions


def comparable_info(info):
    keys = (
        "weapon_events",
        "weapon_event_counts",
        "weapon_launch_count",
        "weapon_hit_count",
        "weapon_miss_count",
        "weapon_invalidated_count",
        "weapon_unresolved_count",
        "weapon_legality",
        "remaining_missiles",
        "team_outcome",
        "episode_terminated",
        "episode_truncated",
        "aircraft_status",
    )
    return {key: info.get(key) for key in keys}


def run_seed(seed, steps):
    legacy = make_environment(seed, reward_variant=DEFAULT_REWARD_VARIANT)
    optimized = make_environment(seed, reward_variant=DEFAULT_REWARD_VARIANT)
    install_legacy_paths(legacy)
    rng = np.random.default_rng(seed + 99173)
    maximum_observation_error = 0.0
    maximum_reward_error = 0.0
    resets = 0
    accepted_launches = 0
    try:
        legacy_obs = legacy.reset()
        optimized_obs = optimized.reset()
        np.testing.assert_array_equal(optimized_obs, legacy_obs)
        for step in range(steps):
            actions = deterministic_actions(rng, step)
            legacy_result = legacy.step(actions)
            optimized_result = optimized.step(actions)
            legacy_obs, legacy_reward, legacy_done, legacy_info = legacy_result
            opt_obs, opt_reward, opt_done, opt_info = optimized_result

            maximum_observation_error = max(
                maximum_observation_error,
                float(np.max(np.abs(opt_obs - legacy_obs))),
            )
            maximum_reward_error = max(
                maximum_reward_error,
                float(np.max(np.abs(opt_reward - legacy_reward))),
            )
            np.testing.assert_array_equal(opt_obs, legacy_obs)
            np.testing.assert_array_equal(opt_reward, legacy_reward)
            np.testing.assert_array_equal(opt_done, legacy_done)
            if comparable_info(opt_info) != comparable_info(legacy_info):
                raise AssertionError(
                    f"weapon/termination telemetry differs at seed={seed} "
                    f"step={step}")
            accepted_launches += int(
                opt_info.get("weapon_accepted_launches", 0))

            if bool(np.any(opt_done)):
                resets += 1
                legacy_obs = legacy.reset()
                optimized_obs = optimized.reset()
                np.testing.assert_array_equal(optimized_obs, legacy_obs)
    finally:
        legacy.close()
        optimized.close()
    return {
        "seed": seed,
        "steps": steps,
        "resets": resets,
        "accepted_launches": accepted_launches,
        "max_observation_abs_error": maximum_observation_error,
        "max_reward_abs_error": maximum_reward_error,
        "exact_observation_match": maximum_observation_error == 0.0,
        "exact_reward_match": maximum_reward_error == 0.0,
        "weapon_and_termination_telemetry_match": True,
    }


def main():
    args = parse_args()
    configure_worker_threads()
    results = [run_seed(seed, args.steps) for seed in args.seeds]
    report = {
        "contract": {
            "sim_freq_hz": 60,
            "agent_interaction_steps": 12,
            "frequency_changed": False,
            "comparison": (
                "legacy sequential low-level + legacy substep versus "
                "batched low-level + no-temp fast substep"),
        },
        "passed": all(
            result["exact_observation_match"]
            and result["exact_reward_match"]
            and result["weapon_and_termination_telemetry_match"]
            for result in results
        ),
        "results": results,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
