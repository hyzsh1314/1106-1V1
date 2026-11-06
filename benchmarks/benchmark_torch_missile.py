"""Benchmark four batched WVR missile slots per 1v1 Torch environment."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from envs.torch6dof import TorchWVRMissileBatch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=96)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    model = TorchWVRMissileBatch().to(device).eval()
    runner = model
    if args.compile:
        runner = torch.compile(
            model, fullgraph=True, dynamic=False,
            options={"triton.cudagraphs": False})
    slots = args.num_envs * 4
    position = torch.zeros(slots, 3, device=device)
    position[:, 1] = torch.arange(slots, device=device) % 4 * 200.0
    velocity = torch.zeros_like(position)
    velocity[:, 0] = 320.0
    target_position = position.clone()
    target_position[:, 0] += 10000.0
    target_velocity = torch.zeros_like(position)
    target_velocity[:, 1] = 180.0
    initial = model.launch(position, velocity)

    with torch.inference_mode():
        state = initial
        for index in range(args.warmup):
            state = runner(state, target_position, target_velocity)
            if (index + 1) % 100 == 0:
                state = initial
        torch.cuda.synchronize()
        durations = []
        for _ in range(args.repeats):
            state = initial
            start = time.perf_counter()
            for index in range(args.iterations * 12):
                state = runner(state, target_position, target_velocity)
                if (index + 1) % 1200 == 0:
                    state = initial
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - start)
    rates = [args.num_envs * args.iterations / value for value in durations]
    report = {
        "backend": "torch_generic_wvr_pn_r2",
        "compiled": bool(args.compile),
        "num_envs": args.num_envs,
        "missile_slots": slots,
        "sim_freq_hz": 60,
        "physics_substeps_per_decision": 12,
        "durations_s": durations,
        "environment_decisions_per_s": rates,
        "median_environment_decisions_per_s": statistics.median(rates),
        "missile_updates_per_s": (
            statistics.median(rates) * 4 * 12),
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
