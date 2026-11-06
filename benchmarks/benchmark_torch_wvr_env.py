"""Benchmark the complete static-shape Torch 1v1 WVR environment."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from envs.torch6dof import TorchWVRConfig, TorchWVRVecEnv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=96)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=85)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    env = TorchWVRVecEnv(
        TorchWVRConfig(num_envs=args.num_envs), device=device,
        seed=args.seed, compile_dynamics=True)
    controls = torch.tensor(
        [0.0, 0.0, 0.0, 0.72], device=device).repeat(
            args.num_envs, 2, 1)
    fire_request = torch.ones(
        args.num_envs, 2, dtype=torch.bool, device=device)
    with torch.inference_mode():
        for _ in range(args.warmup):
            env.step(controls, fire_request)
        torch.cuda.synchronize()
        durations, event_rows = [], []
        for repeat in range(args.repeats):
            env.reset()
            launches = torch.zeros((), dtype=torch.long, device=device)
            hits = torch.zeros_like(launches)
            episodes = torch.zeros_like(launches)
            timeouts = torch.zeros_like(launches)
            zero_sum_error = torch.zeros((), device=device)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(args.iterations):
                _, reward, done, info = env.step(controls, fire_request)
                launches += info["accepted_launches"].sum()
                hits += info["hits"].sum()
                episodes += done.sum()
                timeouts += info["timeout"].sum()
                zero_sum_error = torch.maximum(
                    zero_sum_error, torch.abs(reward.sum(-1)).max())
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - start)
            event_rows.append({
                "repeat": repeat + 1,
                "accepted_launches": int(launches.cpu()),
                "hits": int(hits.cpu()),
                "completed_episodes": int(episodes.cpu()),
                "timeouts": int(timeouts.cpu()),
                "maximum_zero_sum_error": float(zero_sum_error.cpu()),
            })
    rates = [args.num_envs * args.iterations / value for value in durations]
    report = {
        "backend": "torch_wvr_vec_env_v1",
        "scope": "aircraft + launch contract + missiles + reward + termination + reset",
        "num_envs": args.num_envs,
        "sim_freq_hz": 60,
        "physics_substeps_per_decision": 12,
        "iterations_per_repeat": args.iterations,
        "durations_s": durations,
        "environment_decisions_per_s": rates,
        "median_environment_decisions_per_s": statistics.median(rates),
        "events": event_rows,
        "finite_aircraft": bool(torch.isfinite(env.aircraft_state).all()),
        "finite_missiles": bool(torch.isfinite(env.missile_state).all()),
        "gpu_memory_allocated_mb": (
            torch.cuda.memory_allocated() / 1024.0 / 1024.0),
        "scientific_result": False,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
