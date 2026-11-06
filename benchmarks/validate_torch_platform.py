"""Self-contained numerical and physical-consistency validation for Torch WVR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from envs.torch6dof import TorchF16BatchDynamics, TorchWVRMissileBatch
from envs.torch6dof.f16_dynamics import quaternion_to_matrix


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def rpy(state):
    rotation = quaternion_to_matrix(state[..., 6:10])
    return torch.stack((
        torch.atan2(rotation[..., 2, 1], rotation[..., 2, 2]),
        torch.asin((-rotation[..., 2, 0]).clamp(-1.0, 1.0)),
        torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])), dim=-1)


def run_for(model, state, control, seconds):
    for _ in range(round(seconds / model.dt)):
        state = model(state, control)
    return state


def trajectory_distance(first, second):
    return {
        "position_m": float(torch.linalg.vector_norm(
            first[..., :3] - second[..., :3], dim=-1).mean()),
        "body_velocity_m_s": float(torch.linalg.vector_norm(
            first[..., 3:6] - second[..., 3:6], dim=-1).mean()),
        "attitude_deg": float(torch.rad2deg(torch.linalg.vector_norm(
            rpy(first) - rpy(second), dim=-1)).mean()),
        "body_rate_rad_s": float(torch.linalg.vector_norm(
            first[..., 10:13] - second[..., 10:13], dim=-1).mean()),
    }


def cpu_gpu_parity(device):
    cpu_model = TorchF16BatchDynamics().eval()
    gpu_model = TorchF16BatchDynamics().to(device).eval()
    position = torch.zeros(8, 3)
    position[:, 0] = torch.linspace(-1000.0, 1000.0, 8)
    position[:, 2] = -torch.linspace(4000.0, 9000.0, 8)
    heading = torch.linspace(-1.0, 1.0, 8)
    speed = torch.linspace(190.0, 300.0, 8)
    state = cpu_model.level_state(position, heading, speed)
    control = torch.stack((
        torch.linspace(-0.2, 0.2, 8),
        torch.linspace(0.15, -0.15, 8),
        torch.linspace(-0.1, 0.1, 8),
        torch.linspace(0.55, 0.85, 8)), dim=-1)
    with torch.inference_mode():
        cpu = run_for(cpu_model, state, control, 1.0)
        gpu = run_for(gpu_model, state.to(device), control.to(device), 1.0).cpu()
    difference = torch.abs(cpu - gpu)
    return {
        "max_abs_state_error": float(difference.max()),
        "mean_abs_state_error": float(difference.mean()),
        "passed": bool(float(difference.max()) <= 5.0e-4),
    }


def timestep_convergence(device):
    outcomes = []
    with torch.inference_mode():
        for hz in (60, 120, 240):
            model = TorchF16BatchDynamics(dt=1.0 / hz).to(device).eval()
            state = model.level_state(
                torch.tensor([[0.0, 0.0, -6000.0]], device=device),
                torch.tensor([0.3], device=device),
                torch.tensor([250.0], device=device))
            control = torch.tensor(
                [[0.12, -0.08, 0.04, 0.72]], device=device)
            outcomes.append(run_for(model, state, control, 2.0))
    coarse = trajectory_distance(outcomes[0], outcomes[1])
    fine = trajectory_distance(outcomes[1], outcomes[2])
    keys = tuple(coarse)
    passed = all(fine[key] <= coarse[key] + 1.0e-6 for key in keys)
    return {"60_vs_120": coarse, "120_vs_240": fine, "passed": passed}


def mirror_consistency(device):
    model = TorchF16BatchDynamics().to(device).eval()
    positions = torch.tensor([
        [0.0, 500.0, -6000.0], [0.0, -500.0, -6000.0]], device=device)
    state = model.level_state(
        positions, torch.tensor([0.2, -0.2], device=device),
        torch.tensor([250.0, 250.0], device=device))
    controls = torch.tensor([
        [0.15, -0.08, 0.05, 0.72],
        [-0.15, -0.08, -0.05, 0.72]], device=device)
    with torch.inference_mode():
        result = run_for(model, state, controls, 3.0)
    expected = result[0].clone()
    expected[1] *= -1.0
    expected[4] *= -1.0
    expected[10] *= -1.0
    expected[12] *= -1.0
    # Compare physical observables; quaternion components are convention
    # dependent and are checked through mirrored Euler angles instead.
    physical_indices = torch.tensor(
        [0, 1, 2, 3, 4, 5, 10, 11, 12], device=device)
    state_error = torch.abs(
        expected[physical_indices] - result[1, physical_indices])
    expected_rpy = rpy(result[0:1])[0]
    expected_rpy[0] *= -1.0
    expected_rpy[2] *= -1.0
    attitude_error = torch.rad2deg(torch.abs(
        expected_rpy - rpy(result[1:2])[0]))
    max_position_velocity_rate = float(state_error.max())
    max_attitude_deg = float(attitude_error.max())
    return {
        "max_position_velocity_rate_component_error":
            max_position_velocity_rate,
        "max_attitude_error_deg": max_attitude_deg,
        "passed": bool(max_position_velocity_rate < 5.0
                       and max_attitude_deg < 2.0),
    }


def control_direction(device):
    model = TorchF16BatchDynamics().to(device).eval()
    controls = torch.tensor([
        [0.2, 0.0, 0.0, 0.7], [-0.2, 0.0, 0.0, 0.7],
        [0.0, -0.15, 0.0, 0.7], [0.0, 0.15, 0.0, 0.7],
        [0.0, 0.0, 0.15, 0.7], [0.0, 0.0, -0.15, 0.7],
    ], device=device)
    state = model.level_state(
        torch.tensor([[0.0, 0.0, -6000.0]], device=device).repeat(6, 1),
        torch.zeros(6, device=device), torch.full((6,), 250.0, device=device))
    with torch.inference_mode():
        result = run_for(model, state, controls, 1.0)
    angles = rpy(result)
    pairs = {
        "roll": [float(angles[0, 0]), float(angles[1, 0])],
        "pitch": [float(angles[2, 1]), float(angles[3, 1])],
        "yaw": [float(angles[4, 2]), float(angles[5, 2])],
    }
    passed = all(first * second < 0.0 for first, second in pairs.values())
    return {"responses_rad": pairs, "passed": passed}


def envelope_stability(device):
    model = TorchF16BatchDynamics().to(device).eval()
    altitudes = torch.tensor([3000.0, 6000.0, 9000.0, 12000.0], device=device)
    speeds = torch.tensor([180.0, 240.0, 300.0], device=device)
    controls = torch.tensor([
        [0.0, 0.0, 0.0, 0.65], [0.25, 0.0, 0.0, 0.75],
        [0.0, -0.2, 0.0, 0.8], [0.0, 0.15, 0.0, 0.55],
        [0.15, -0.12, 0.08, 0.8]], device=device)
    grid = torch.cartesian_prod(
        torch.arange(altitudes.numel(), device=device),
        torch.arange(speeds.numel(), device=device),
        torch.arange(controls.shape[0], device=device))
    position = torch.zeros(grid.shape[0], 3, device=device)
    position[:, 2] = -altitudes[grid[:, 0]]
    state = model.level_state(
        position, torch.zeros(grid.shape[0], device=device),
        speeds[grid[:, 1]])
    command = controls[grid[:, 2]]
    finite = True
    maximum_rate = 0.0
    minimum_speed = float("inf")
    maximum_speed = 0.0
    maximum_quaternion_error = 0.0
    with torch.inference_mode():
        for _ in range(10 * 60):
            state = model(state, command)
            finite = finite and bool(torch.isfinite(state).all())
            maximum_rate = max(maximum_rate, float(torch.abs(
                state[..., 10:13]).max()))
            scalar_speed = torch.linalg.vector_norm(state[..., 3:6], dim=-1)
            minimum_speed = min(minimum_speed, float(scalar_speed.min()))
            maximum_speed = max(maximum_speed, float(scalar_speed.max()))
            q_error = torch.abs(torch.linalg.vector_norm(
                state[..., 6:10], dim=-1) - 1.0)
            maximum_quaternion_error = max(
                maximum_quaternion_error, float(q_error.max()))
    passed = (finite and maximum_rate <= 4.0 + 1e-6
              and maximum_quaternion_error <= 1e-5
              and minimum_speed >= 0.0 and maximum_speed < 1000.0)
    return {
        "cases": int(grid.shape[0]), "duration_s": 10,
        "finite": finite, "minimum_speed_m_s": minimum_speed,
        "maximum_speed_m_s": maximum_speed,
        "maximum_body_rate_rad_s": maximum_rate,
        "maximum_quaternion_norm_error": maximum_quaternion_error,
        "passed": passed,
    }


def missile_case(device, hz, target_position, target_velocity,
                 maximum_seconds=40.5):
    model = TorchWVRMissileBatch(dt=1.0 / hz).to(device).eval()
    state = model.launch(
        torch.tensor([[0.0, 0.0, 6000.0]], device=device),
        torch.tensor([[320.0, 0.0, 0.0]], device=device))
    target_position = torch.tensor([target_position], device=device)
    target_velocity = torch.tensor([target_velocity], device=device)
    closest = float("inf")
    terminal_time = maximum_seconds
    with torch.inference_mode():
        for index in range(round(maximum_seconds * hz)):
            state = model(state, target_position, target_velocity)
            target_position = target_position + target_velocity / hz
            closest = min(closest, float(torch.linalg.vector_norm(
                target_position - state[..., :3], dim=-1)[0]))
            if int(state[0, 12].item()) in (model.HIT, model.MISS):
                terminal_time = (index + 1) / hz
                break
    return {
        "status": int(state[0, 12].item()),
        "terminal_time_s": terminal_time,
        "closest_sampled_distance_m": closest,
        "finite": bool(torch.isfinite(state).all()),
    }


def missile_validation(device):
    direct_60 = missile_case(
        device, 60, [5000.0, 0.0, 6000.0], [0.0, 0.0, 0.0])
    direct_120 = missile_case(
        device, 120, [5000.0, 0.0, 6000.0], [0.0, 0.0, 0.0])
    unreachable = missile_case(
        device, 60, [10000.0, 0.0, 6000.0], [900.0, 0.0, 0.0])
    time_difference = abs(
        direct_60["terminal_time_s"] - direct_120["terminal_time_s"])
    passed = (
        direct_60["status"] == TorchWVRMissileBatch.HIT
        and direct_120["status"] == TorchWVRMissileBatch.HIT
        and time_difference <= 0.1
        and unreachable["status"] == TorchWVRMissileBatch.MISS
        and direct_60["finite"] and direct_120["finite"]
        and unreachable["finite"])
    return {
        "direct_60hz": direct_60, "direct_120hz": direct_120,
        "hit_time_difference_s": time_difference,
        "unreachable_60hz": unreachable, "passed": passed,
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    results = {
        "contract": "Torch空战平台模型合同与验证路线.md",
        "device": str(device), "sim_freq_hz": 60,
        "cpu_gpu_parity": cpu_gpu_parity(device),
        "timestep_convergence": timestep_convergence(device),
        "mirror_consistency": mirror_consistency(device),
        "control_direction": control_direction(device),
        "envelope_stability": envelope_stability(device),
        "missile": missile_validation(device),
    }
    results["passed"] = all(
        section.get("passed", False) for name, section in results.items()
        if isinstance(section, dict) and name not in ("contract",))
    rendered = json.dumps(results, indent=2, ensure_ascii=False)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
