"""Benchmark the experimental Torch 6-DoF backend at the project 60 Hz rate."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from envs.torch6dof import CompiledTorchF16Runner, TorchF16BatchDynamics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=96)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def mark_compiled_step(enabled):
    if enabled:
        torch.compiler.cudagraph_mark_step_begin()


def initial_batch(model, num_envs, device):
    aircraft = num_envs * 2
    position = torch.zeros(aircraft, 3, device=device)
    position[:, 0] = torch.arange(
        aircraft, device=device, dtype=torch.float32).remainder(2) * 10000.0
    position[:, 2] = -6000.0
    heading = torch.arange(
        aircraft, device=device, dtype=torch.float32).remainder(2) * torch.pi
    speed = torch.full((aircraft,), 250.0, device=device)
    state = model.level_state(position, heading, speed)
    controls = torch.empty(aircraft, 4, device=device)
    controls[:, 0] = 0.05
    controls[:, 1] = -0.03
    controls[:, 2] = 0.02
    controls[:, 3] = 0.75
    return state, controls


def main():
    args = parse_args()
    if min(args.num_envs, args.iterations, args.repeats) <= 0:
        raise ValueError("num-envs, iterations and repeats must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = TorchF16BatchDynamics().to(device).eval()
    runner = model
    if args.compile:
        runner = CompiledTorchF16Runner(model)
    initial_state, controls = initial_batch(model, args.num_envs, device)

    with torch.inference_mode():
        state = initial_state
        for index in range(args.warmup):
            mark_compiled_step(args.compile)
            state = runner(state, controls, substeps=12)
            if (index + 1) % 100 == 0:
                state = initial_state
        synchronize(device)

        durations = []
        for _ in range(args.repeats):
            state = initial_state
            synchronize(device)
            start = time.perf_counter()
            for index in range(args.iterations):
                mark_compiled_step(args.compile)
                state = runner(state, controls, substeps=12)
                if (index + 1) % 100 == 0:
                    state = initial_state
            synchronize(device)
            durations.append(time.perf_counter() - start)
    rates = [args.num_envs * args.iterations / value for value in durations]
    median_rate = statistics.median(rates)
    result = {
        "backend": "experimental_torch6dof_v1",
        "device": str(device),
        "compiled": bool(args.compile),
        "num_envs": args.num_envs,
        "aircraft": args.num_envs * 2,
        "sim_freq_hz": 60,
        "physics_substeps_per_decision": 12,
        "iterations_per_repeat": args.iterations,
        "durations_s": durations,
        "environment_decisions_per_s": rates,
        "median_environment_decisions_per_s": median_rate,
        "aircraft_physics_substeps_per_s": median_rate * 2 * 12,
        "torch_version": torch.__version__,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
