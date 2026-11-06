"""Audit frozen LAG low-level command tracking on the Torch dynamics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from benchmarks.benchmark_torch6dof_ppo import lowlevel_controls
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.utils.utils import get_root_dir
from envs.torch6dof import CompiledTorchF16Runner, TorchF16BatchDynamics
from envs.torch6dof.f16_dynamics import quaternion_to_matrix


COMMANDS = {
    "neutral": (1, 2, 1),
    "climb": (0, 2, 1),
    "descend": (2, 2, 1),
    "turn_negative": (1, 0, 1),
    "turn_positive": (1, 4, 1),
    "accelerate": (1, 2, 0),
    "decelerate": (1, 2, 2),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--torch-speed-adapter", action=argparse.BooleanOptionalAction,
        default=True)
    parser.add_argument("--neutral-throttle-bias", type=float, default=-0.265)
    parser.add_argument("--speed-command-gain", type=float, default=4.0)
    parser.add_argument("--high-speed-throttle-bias", type=float,
                        default=-0.18)
    return parser.parse_args()


def own_observation(state, size=39):
    rotation = quaternion_to_matrix(state[..., 6:10])
    roll = torch.atan2(rotation[..., 2, 1], rotation[..., 2, 2])
    pitch = torch.asin((-rotation[..., 2, 0]).clamp(-1.0, 1.0))
    obs = torch.zeros(state.shape[0], 1, size, device=state.device)
    obs[:, 0, 0] = -state[:, 2] / 5000.0
    obs[:, 0, 1] = torch.sin(roll)
    obs[:, 0, 2] = torch.cos(roll)
    obs[:, 0, 3] = torch.sin(pitch)
    obs[:, 0, 4] = torch.cos(pitch)
    obs[:, 0, 5:8] = state[:, 3:6] / 340.0
    obs[:, 0, 8] = torch.linalg.vector_norm(state[:, 3:6], dim=-1) / 340.0
    return obs


def wrap_angle(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def main():
    args = parse_args()
    device = torch.device(args.device)
    dynamics = TorchF16BatchDynamics().to(device).eval()
    runner = (CompiledTorchF16Runner(dynamics)
              if device.type == "cuda" else dynamics)
    policy = BaselineActor().to(device).eval()
    policy.load_state_dict(torch.load(
        get_root_dir() + "/model/baseline_model.pt",
        map_location=device, weights_only=True))
    policy.requires_grad_(False)
    policy.tpdv = {"dtype": torch.float32, "device": device}
    altitudes = torch.tensor([4000.0, 6000.0, 9000.0], device=device)
    speeds = torch.tensor([200.0, 250.0, 300.0], device=device)
    command_names = list(COMMANDS)
    cases = []
    for command_index, name in enumerate(command_names):
        for altitude in altitudes.cpu().tolist():
            for speed in speeds.cpu().tolist():
                cases.append((command_index, name, altitude, speed))
    count = len(cases)
    position = torch.zeros(count, 3, device=device)
    position[:, 2] = -torch.tensor([case[2] for case in cases], device=device)
    initial_speed = torch.tensor([case[3] for case in cases], device=device)
    state = dynamics.level_state(
        position, torch.zeros(count, device=device), initial_speed)
    initial_state = state.clone()
    action = torch.tensor(
        [(*COMMANDS[case[1]], 0) for case in cases],
        dtype=torch.long, device=device).unsqueeze(1)
    rnn = torch.zeros(count, 1, 128, device=device)
    maximum_rate = torch.zeros(count, device=device)
    minimum_altitude = -state[:, 2]
    finite = torch.ones(count, dtype=torch.bool, device=device)
    decision_steps = round(args.seconds * 5.0)
    with torch.inference_mode():
        # Compile outside the measured trajectory.
        runner(state, torch.zeros(count, 4, device=device), substeps=1)
        for _ in range(decision_steps):
            obs = own_observation(state)
            control, rnn = lowlevel_controls(
                policy, rnn, obs, action,
                torch_speed_adapter=args.torch_speed_adapter,
                neutral_throttle_bias=args.neutral_throttle_bias,
                speed_command_gain=args.speed_command_gain,
                high_speed_throttle_bias=args.high_speed_throttle_bias)
            state = runner(state, control, substeps=12)
            maximum_rate = torch.maximum(
                maximum_rate,
                torch.linalg.vector_norm(state[:, 10:13], dim=-1))
            minimum_altitude = torch.minimum(minimum_altitude, -state[:, 2])
            finite &= torch.isfinite(state).all(-1)
    initial_rotation = quaternion_to_matrix(initial_state[..., 6:10])
    final_rotation = quaternion_to_matrix(state[..., 6:10])
    initial_yaw = torch.atan2(
        initial_rotation[..., 1, 0], initial_rotation[..., 0, 0])
    final_yaw = torch.atan2(
        final_rotation[..., 1, 0], final_rotation[..., 0, 0])
    altitude_delta = -state[:, 2] + initial_state[:, 2]
    speed_delta = (
        torch.linalg.vector_norm(state[:, 3:6], dim=-1) - initial_speed)
    yaw_delta = wrap_angle(final_yaw - initial_yaw)
    summaries = {}
    details = []
    for command_index, name in enumerate(command_names):
        selection = torch.tensor(
            [case[0] == command_index for case in cases], device=device)
        summaries[name] = {
            "mean_altitude_delta_m": float(altitude_delta[selection].mean()),
            "mean_heading_delta_deg": float(torch.rad2deg(
                yaw_delta[selection]).mean()),
            "mean_speed_delta_m_s": float(speed_delta[selection].mean()),
            "maximum_body_rate_rad_s": float(maximum_rate[selection].max()),
            "minimum_altitude_m": float(minimum_altitude[selection].min()),
            "all_finite": bool(finite[selection].all()),
        }
    for index, (_, name, altitude, speed) in enumerate(cases):
        details.append({
            "command": name,
            "initial_altitude_m": altitude,
            "initial_speed_m_s": speed,
            "altitude_delta_m": float(altitude_delta[index]),
            "heading_delta_deg": float(torch.rad2deg(yaw_delta[index])),
            "speed_delta_m_s": float(speed_delta[index]),
            "maximum_body_rate_rad_s": float(maximum_rate[index]),
            "minimum_altitude_m": float(minimum_altitude[index]),
            "finite": bool(finite[index]),
        })
    pair_checks = {
        "altitude_commands_opposite": (
            summaries["climb"]["mean_altitude_delta_m"]
            * summaries["descend"]["mean_altitude_delta_m"] < 0.0),
        "heading_commands_opposite": (
            summaries["turn_negative"]["mean_heading_delta_deg"]
            * summaries["turn_positive"]["mean_heading_delta_deg"] < 0.0),
        "speed_commands_opposite": (
            summaries["accelerate"]["mean_speed_delta_m_s"]
            * summaries["decelerate"]["mean_speed_delta_m_s"] < 0.0),
    }
    safety_passed = bool(finite.all()) and float(minimum_altitude.min()) > 2500.0
    neutral_speed_hold_passed = (
        abs(summaries["neutral"]["mean_speed_delta_m_s"]) <= 5.0
        and max(abs(item["speed_delta_m_s"]) for item in details
                if item["command"] == "neutral") <= 15.0)
    report = {
        "seconds": args.seconds,
        "grid_cases": count,
        "altitudes_m": altitudes.cpu().tolist(),
        "speeds_m_s": speeds.cpu().tolist(),
        "torch_speed_adapter": args.torch_speed_adapter,
        "neutral_throttle_bias": args.neutral_throttle_bias,
        "speed_command_gain": args.speed_command_gain,
        "high_speed_throttle_bias": args.high_speed_throttle_bias,
        "summaries": summaries,
        "pair_checks": pair_checks,
        "safety_passed": safety_passed,
        "directionality_passed": all(pair_checks.values()),
        "neutral_speed_hold_passed": neutral_speed_hold_passed,
        "passed": (safety_passed and all(pair_checks.values())
                   and neutral_speed_hold_passed),
        "details": details,
        "scientific_result": False,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
