"""Reproducible throughput benchmark for the formal 60 Hz 1v1 platform.

The benchmark deliberately excludes environment construction and reset from
the timed region.  It reports high-level environment decisions, agent
transitions, and JSBSim physics substeps separately so that performance
changes cannot be hidden by changing the simulation/control frequencies.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from functools import partial
from pathlib import Path

import numpy as np

from benchmarks.train_ppo_1v1_shoot_smoke import (
    DEFAULT_REWARD_VARIANT,
    SCENARIO,
    make_environment,
)
from envs.env_wrappers import (
    DummyVecEnv,
    SubprocVecEnv,
    configure_worker_threads,
)


SIM_FREQ_HZ = 60
AGENT_INTERACTION_STEPS = 12
NUM_AGENTS = 2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=85)
    parser.add_argument("--start-method", choices=("spawn", "forkserver"),
                        default="spawn")
    parser.add_argument("--in-series", type=int, default=1)
    parser.add_argument("--fire-mode", choices=("hold", "request"),
                        default="hold")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def constant_actions(num_envs: int, fire_mode: str) -> np.ndarray:
    """Return neutral hierarchical commands with hold/fire as requested."""

    action = np.asarray(
        [1, 2, 1, int(fire_mode == "request")], dtype=np.int64)
    return np.broadcast_to(action, (num_envs, NUM_AGENTS, 4)).copy()


def build_vector_environment(args):
    factories = [
        partial(
            make_environment,
            args.seed + index * 1000,
            scenario=SCENARIO,
            reward_variant=DEFAULT_REWARD_VARIANT,
        )
        for index in range(args.num_envs)
    ]
    if args.num_envs == 1:
        return DummyVecEnv(factories)
    return SubprocVecEnv(
        factories, context=args.start_method, in_series=args.in_series)


def accumulate_telemetry(infos, counters):
    for info in infos:
        counters["fire_requests"] += int(
            info.get("weapon_fire_requests", 0))
        counters["accepted_launches"] += int(
            info.get("weapon_accepted_launches", 0))
        counters["rejected_requests"] += int(
            info.get("weapon_rejected_requests", 0))


def main():
    args = parse_args()
    if min(args.num_envs, args.steps, args.repeats) <= 0:
        raise ValueError("num-envs, steps, and repeats must be positive")
    if args.warmup_steps < 0:
        raise ValueError("warmup-steps cannot be negative")
    if args.num_envs % args.in_series:
        raise ValueError("num-envs must be divisible by in-series")

    # Match the formal subprocess workers.  Tiny recurrent policies become
    # slower and much noisier when each environment starts a BLAS thread pool.
    configure_worker_threads()
    environment = build_vector_environment(args)
    actions = constant_actions(args.num_envs, args.fire_mode)
    counters = {
        "episodes": 0,
        "fire_requests": 0,
        "accepted_launches": 0,
        "rejected_requests": 0,
    }
    durations = []
    try:
        environment.reset()
        for _ in range(args.warmup_steps):
            _, _, dones, _ = environment.step(actions)
            counters["episodes"] += int(np.any(dones, axis=1).sum())

        for _ in range(args.repeats):
            repeat_counters = {
                "episodes": 0,
                "fire_requests": 0,
                "accepted_launches": 0,
                "rejected_requests": 0,
            }
            start = time.perf_counter()
            for _ in range(args.steps):
                _, _, dones, infos = environment.step(actions)
                repeat_counters["episodes"] += int(
                    np.any(dones, axis=1).sum())
                accumulate_telemetry(infos, repeat_counters)
            durations.append(time.perf_counter() - start)
            for name, value in repeat_counters.items():
                counters[name] += value
    finally:
        environment.close()

    decisions_per_repeat = args.num_envs * args.steps
    decision_rates = [decisions_per_repeat / value for value in durations]
    median_decisions_per_s = statistics.median(decision_rates)
    result = {
        "contract": {
            "scenario": SCENARIO,
            "reward_variant": DEFAULT_REWARD_VARIANT,
            "sim_freq_hz": SIM_FREQ_HZ,
            "agent_interaction_steps": AGENT_INTERACTION_STEPS,
            "decision_interval_s": (
                AGENT_INTERACTION_STEPS / SIM_FREQ_HZ),
            "num_agents": NUM_AGENTS,
            "frequency_changed": False,
        },
        "configuration": {
            "num_envs": args.num_envs,
            "steps_per_repeat": args.steps,
            "warmup_steps": args.warmup_steps,
            "repeats": args.repeats,
            "seed": args.seed,
            "start_method": (
                "dummy" if args.num_envs == 1 else args.start_method),
            "in_series": args.in_series,
            "fire_mode": args.fire_mode,
        },
        "timing": {
            "durations_s": durations,
            "environment_decisions_per_s": decision_rates,
            "median_environment_decisions_per_s": median_decisions_per_s,
            "median_agent_transitions_per_s": (
                median_decisions_per_s * NUM_AGENTS),
            "median_physics_substeps_per_s": (
                median_decisions_per_s * AGENT_INTERACTION_STEPS),
        },
        "telemetry": counters,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
