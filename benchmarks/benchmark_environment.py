"""Benchmark single-process and subprocess WVR environments.

Examples
--------
python -m benchmarks.benchmark_environment --scenario r6 --num-envs 1 2 4 8
python -m benchmarks.benchmark_environment --scenario v22 --num-envs 1 2 4
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
from envs.env_wrappers import ShareSubprocVecEnv


SCENARIOS = {
    "r6": "2v2/NoWeapon/PBGR_R6_Entity_NoWeapon",
    "v22": "2v2/ShootMissile/PBGR_V22_PLATFORM_CORRECT",
}


def runtime_metadata():
    result = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "numpy_version": np.__version__,
    }
    try:
        import jsbsim

        result["jsbsim_version"] = getattr(jsbsim, "__version__", "unknown")
    except ImportError:
        result["jsbsim_version"] = None
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        result.update({
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": cuda_available,
            "cuda_device": (
                torch.cuda.get_device_name(0) if cuda_available else None
            ),
        })
    except ImportError:
        result.update({
            "torch_version": None,
            "torch_cuda_version": None,
            "cuda_available": False,
            "cuda_device": None,
        })
    return result


def make_environment(config_name, seed, reset_index=0, reuse_fdm=False):
    env = MultipleCombatEnv(config_name, reuse_fdm=reuse_fdm)
    env.seed(int(seed))
    if env.get_initial_geometry_state() is not None:
        env.set_initial_geometry_reset_index(int(reset_index))
    return env


def neutral_action(env, weapon_mode="hold"):
    nvec = np.asarray(env.action_space.nvec, dtype=np.int64)
    row = nvec // 2
    if row.size >= 5:
        row[-2] = 0
        row[-1] = int(weapon_mode == "fire")
    return np.repeat(row[None, :], int(env.num_agents), axis=0)


def _done(value):
    return bool(np.all(np.asarray(value, dtype=np.bool_)))


def _percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def benchmark_single(config_name, steps, warmup, repeats, seed,
                     reuse_fdm=False, weapon_mode="hold"):
    env = make_environment(config_name, seed, reuse_fdm=reuse_fdm)
    try:
        action = neutral_action(env, weapon_mode=weapon_mode)
        env.reset()
        for _ in range(warmup):
            result = env.step(action)
            if _done(result[3]):
                env.reset()
        durations = []
        completed = []
        for _ in range(repeats):
            env.reset()
            start = time.perf_counter()
            count = 0
            for _ in range(steps):
                result = env.step(action)
                count += 1
                if _done(result[3]):
                    env.reset()
            durations.append(time.perf_counter() - start)
            completed.append(count)
        reset_times = []
        for _ in range(min(10, repeats + 2)):
            start = time.perf_counter()
            env.reset()
            reset_times.append(time.perf_counter() - start)
        return _summarize(
            env, 1, durations, completed, reset_times, steps, warmup
        )
    finally:
        env.close()


def benchmark_subprocess(config_name, num_envs, steps, warmup, repeats, seed,
                         reuse_fdm=False, weapon_mode="hold"):
    factories = [
        (
            lambda index=index: make_environment(
                config_name, int(seed) + index, reset_index=0,
                reuse_fdm=reuse_fdm
            )
        )
        for index in range(num_envs)
    ]
    vec = ShareSubprocVecEnv(factories, context="spawn", in_series=1)
    probe = make_environment(config_name, seed, reuse_fdm=reuse_fdm)
    try:
        action = neutral_action(probe, weapon_mode=weapon_mode)
        actions = np.repeat(action[None, ...], num_envs, axis=0)
        vec.reset()
        for _ in range(warmup):
            vec.step(actions)
        durations = []
        completed = []
        for _ in range(repeats):
            vec.reset()
            start = time.perf_counter()
            for _ in range(steps):
                vec.step(actions)
            durations.append(time.perf_counter() - start)
            completed.append(steps * num_envs)
        reset_times = []
        for _ in range(min(10, repeats + 2)):
            start = time.perf_counter()
            vec.reset()
            reset_times.append(time.perf_counter() - start)
        return _summarize(
            probe, num_envs, durations, completed, reset_times, steps, warmup
        )
    finally:
        vec.close()
        probe.close()


def _summarize(env, num_envs, durations, completed, reset_times, steps, warmup):
    rates = [count / duration for count, duration in zip(completed, durations)]
    median_rate = float(statistics.median(rates))
    agents = int(env.num_agents)
    substeps = int(env.agent_interaction_steps)
    sim_hz = float(getattr(env.config, "sim_freq"))
    simulated_seconds_per_decision = substeps / sim_hz
    return {
        "num_envs": int(num_envs),
        "num_agents": agents,
        "physics_substeps_per_decision": substeps,
        "steps_per_repeat_per_env": int(steps),
        "warmup_steps": int(warmup),
        "repeat_wall_seconds": durations,
        "decision_steps_per_second": median_rate,
        "agent_transitions_per_second": median_rate * agents,
        "aircraft_integrator_substeps_per_second": (
            median_rate * agents * substeps
        ),
        "simulated_seconds_per_wall_second": (
            median_rate * simulated_seconds_per_decision
        ),
        "reset_seconds_p50": _percentile(reset_times, 50),
        "reset_seconds_p95": _percentile(reset_times, 95),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="r6")
    parser.add_argument("--config-name", default=None)
    parser.add_argument("--num-envs", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--reuse-fdm", action="store_true")
    parser.add_argument("--weapon-mode", choices=("hold", "fire"), default="hold")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.steps <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise ValueError("steps/repeats must be positive and warmup non-negative")
    if any(value <= 0 for value in args.num_envs):
        raise ValueError("num-envs values must be positive")
    config_name = args.config_name or SCENARIOS[args.scenario]
    report = {
        "schema_version": "WVRBenchmarkV1",
        "scenario": args.scenario,
        "config_name": config_name,
        "seed": args.seed,
        "reuse_fdm": bool(args.reuse_fdm),
        "weapon_mode": args.weapon_mode,
        "runtime": runtime_metadata(),
        "results": [],
    }
    for num_envs in args.num_envs:
        if num_envs == 1:
            result = benchmark_single(
                config_name, args.steps, args.warmup, args.repeats, args.seed,
                reuse_fdm=args.reuse_fdm, weapon_mode=args.weapon_mode
            )
        else:
            result = benchmark_subprocess(
                config_name, num_envs, args.steps, args.warmup,
                args.repeats, args.seed, reuse_fdm=args.reuse_fdm,
                weapon_mode=args.weapon_mode
            )
        report["results"].append(result)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

