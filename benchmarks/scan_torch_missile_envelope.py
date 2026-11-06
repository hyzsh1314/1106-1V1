"""Scan the generic Torch WVR missile envelope and audit its launch mask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from envs.torch6dof import TorchWVRMissileBatch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def predicted_dynamic_range(closure, target_aspect, los_rate):
    closure_credit = 8.0 * closure.clamp(-150.0, 550.0)
    tail_penalty = 2200.0 * ((1.0 - torch.cos(target_aspect)) * 0.5)
    rate_penalty = 5000.0 * (los_rate / 0.20).clamp(0.0, 1.0)
    return (7500.0 + closure_credit - tail_penalty - rate_penalty).clamp(
        2500.0, 14000.0)


def main():
    args = parse_args()
    device = torch.device(args.device)
    missile = TorchWVRMissileBatch().to(device).eval()
    ranges_km = torch.arange(2.0, 15.0, 1.0, device=device)
    target_heading_deg = torch.tensor(
        [0.0, 45.0, 90.0, 135.0, 180.0], device=device)
    turn_g = torch.tensor([0.0, 3.0, 6.0], device=device)
    grid = torch.cartesian_prod(ranges_km, target_heading_deg, turn_g)
    distance = 1000.0 * grid[:, 0]
    heading = torch.deg2rad(grid[:, 1])
    target_turn_g = grid[:, 2]
    speed = 250.0
    launch_position = torch.zeros(grid.shape[0], 3, device=device)
    launch_position[:, 2] = 6000.0
    launch_velocity = torch.zeros_like(launch_position)
    launch_velocity[:, 0] = speed
    target_position = launch_position.clone()
    target_position[:, 0] += distance
    target_velocity = torch.stack((
        speed * torch.cos(heading), speed * torch.sin(heading),
        torch.zeros_like(heading)), dim=-1)
    state = missile.launch(launch_position, launch_velocity)
    initial_relative_velocity = target_velocity - launch_velocity
    closure = -initial_relative_velocity[:, 0]
    target_aspect = torch.pi - heading
    los_rate = torch.abs(initial_relative_velocity[:, 1]) / distance
    predicted_range = predicted_dynamic_range(
        closure, target_aspect, los_rate)
    predicted_legal = distance <= predicted_range

    with torch.inference_mode():
        for _ in range(round(
                missile.parameters.maximum_flight_time_s / missile.dt) + 1):
            state = missile(state, target_position, target_velocity)
            turn_rate = (target_turn_g * missile.parameters.gravity_m_s2
                         / speed)
            heading = heading + missile.dt * turn_rate
            target_velocity = torch.stack((
                speed * torch.cos(heading), speed * torch.sin(heading),
                torch.zeros_like(heading)), dim=-1)
            target_position = target_position + missile.dt * target_velocity
    hit = state[:, 12] == float(missile.HIT)
    true_positive = int((predicted_legal & hit).sum().cpu())
    false_positive = int((predicted_legal & ~hit).sum().cpu())
    true_negative = int((~predicted_legal & ~hit).sum().cpu())
    false_negative = int((~predicted_legal & hit).sum().cpu())
    scenarios = []
    for heading_value in target_heading_deg.cpu().tolist():
        for turn_value in turn_g.cpu().tolist():
            selection = ((grid[:, 1] == heading_value)
                         & (grid[:, 2] == turn_value))
            hit_ranges = grid[selection & hit, 0]
            legal_ranges = grid[selection & predicted_legal, 0]
            scenarios.append({
                "target_heading_deg": heading_value,
                "target_turn_g": turn_value,
                "maximum_hit_range_km": (
                    float(hit_ranges.max().cpu())
                    if hit_ranges.numel() else None),
                "maximum_legal_range_km": (
                    float(legal_ranges.max().cpu())
                    if legal_ranges.numel() else None),
                "hit_count": int(hit_ranges.numel()),
            })
    report = {
        "model": "TorchWVRMissileBatch",
        "grid_cases": int(grid.shape[0]),
        "range_km": ranges_km.cpu().tolist(),
        "target_heading_deg": target_heading_deg.cpu().tolist(),
        "target_turn_g": turn_g.cpu().tolist(),
        "confusion_matrix": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
        },
        "legal_precision": true_positive / max(
            true_positive + false_positive, 1),
        "hit_recall": true_positive / max(true_positive + false_negative, 1),
        "scenarios": scenarios,
        "scientific_result": False,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
