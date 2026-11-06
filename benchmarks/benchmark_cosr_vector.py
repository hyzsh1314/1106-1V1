"""Measure synchronous COSR subprocess sampling throughput."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import yaml

from ac_cosr_wvr.env.scenario_sampler import ScenarioSampler
from ac_cosr_wvr.env.vector_env import COSRSubprocVectorEnv
from benchmarks.benchmark_environment import runtime_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "cosr" / "autogun_v1.yaml"


def _load_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def benchmark(config_path, n_envs, steps, warmup, repeats, seed):
    config = _load_config(config_path)
    sampler = ScenarioSampler.from_config_path(config_path, seed, split_code=1)
    next_episode = 0

    def scenarios(count):
        nonlocal next_episode
        values = [
            sampler.sample(next_episode + offset, 0, "validation")
            for offset in range(count)
        ]
        next_episode += count
        return values

    actions = np.zeros((n_envs, 2, 4), dtype=np.float64)
    actions[..., 3] = 0.55
    durations = []
    with COSRSubprocVectorEnv(
        config, n_envs=n_envs, logging_mode="train"
    ) as env:
        env.reset(scenarios(n_envs))
        for _ in range(warmup):
            result = env.step(actions)
            done = np.flatnonzero(result["terminated"] | result["truncated"])
            if done.size:
                env.reset_at(done.tolist(), scenarios(done.size))
        for _ in range(repeats):
            env.reset(scenarios(n_envs))
            start = time.perf_counter()
            for _ in range(steps):
                result = env.step(actions)
                done = np.flatnonzero(
                    result["terminated"] | result["truncated"]
                )
                if done.size:
                    env.reset_at(done.tolist(), scenarios(done.size))
            durations.append(time.perf_counter() - start)
    rates = [n_envs * steps / duration for duration in durations]
    rate = float(statistics.median(rates))
    substeps = int(round(
        float(config["physics"]["control_interval_s"])
        * float(config["physics"]["sim_freq_hz"])
    ))
    return {
        "n_envs": n_envs,
        "repeat_wall_seconds": durations,
        "decision_steps_per_second": rate,
        "agent_transitions_per_second": rate * 2,
        "aircraft_integrator_substeps_per_second": rate * 2 * substeps,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--num-envs", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    report = {
        "schema_version": "COSRVectorBenchmarkV1",
        "config": str(args.config.resolve()),
        "runtime": runtime_metadata(),
        "results": [
            benchmark(
                args.config, n_envs, args.steps, args.warmup,
                args.repeats, args.seed
            )
            for n_envs in args.num_envs
        ],
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

