"""Open-loop fidelity audit of experimental Torch 6-DoF against JSBSim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.core.simulatior import AircraftSimulator
from envs.torch6dof import TorchF16BatchDynamics
from envs.torch6dof.f16_dynamics import quaternion_to_matrix


CASES = {
    "neutral": [0.0, 0.0, 0.0, 0.70],
    "roll": [0.30, 0.0, 0.0, 0.70],
    "pull": [0.0, -0.20, 0.0, 0.75],
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def wrap_angle(value):
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def torch_observables(state):
    rotation = quaternion_to_matrix(state[..., 6:10])
    velocity_ned = torch.matmul(
        rotation, state[..., 3:6].unsqueeze(-1)).squeeze(-1)
    r00, r10, r20 = rotation[..., 0, 0], rotation[..., 1, 0], rotation[..., 2, 0]
    r21, r22 = rotation[..., 2, 1], rotation[..., 2, 2]
    roll = torch.atan2(r21, r22)
    pitch = torch.asin((-r20).clamp(-1.0, 1.0))
    yaw = torch.atan2(r10, r00)
    position_neu = torch.stack((
        state[..., 0], state[..., 1], -state[..., 2]), dim=-1)
    velocity_neu = torch.stack((
        velocity_ned[..., 0], velocity_ned[..., 1],
        -velocity_ned[..., 2]), dim=-1)
    return (
        position_neu.detach().cpu().numpy()[0],
        velocity_neu.detach().cpu().numpy()[0],
        torch.stack((roll, pitch, yaw), dim=-1).detach().cpu().numpy()[0],
        state[..., 10:13].detach().cpu().numpy()[0],
    )


def jsbsim_observables(simulator):
    body_velocity = np.asarray(simulator.get_property_values([
        c.velocities_u_mps, c.velocities_v_mps, c.velocities_w_mps,
    ]), dtype=np.float64)
    rates = np.asarray(simulator.get_property_values([
        c.velocities_p_rad_sec, c.velocities_q_rad_sec,
        c.velocities_r_rad_sec,
    ]), dtype=np.float64)
    controls = np.asarray(simulator.get_property_values([
        c.fcs_left_aileron_pos_norm, c.fcs_elevator_pos_norm,
        c.fcs_rudder_pos_norm, c.fcs_throttle_pos_norm,
    ]), dtype=np.float64)
    return (
        np.asarray(simulator.get_position(), dtype=np.float64).copy(),
        np.asarray(simulator.get_velocity(), dtype=np.float64).copy(),
        np.asarray(simulator.get_rpy(), dtype=np.float64).copy(),
        rates,
        body_velocity, controls,
    )


def run_case(name, command, seconds, device):
    altitude_m = 20000.0 * 0.3048
    speed_m_s = 800.0 * 0.3048
    initial = {
        "ic_long_gc_deg": 120.0,
        "ic_lat_geod_deg": 60.0,
        "ic_h_sl_ft": 20000.0,
        "ic_psi_true_deg": 0.0,
        "ic_u_fps": 800.0,
        "ic_v_fps": 0.0,
        "ic_w_fps": 0.0,
        "ic_p_rad_sec": 0.0,
        "ic_q_rad_sec": 0.0,
        "ic_r_rad_sec": 0.0,
    }
    simulator = AircraftSimulator(
        uid="A0100", color="Red", model="f16", init_state=initial,
        origin=(120.0, 60.0, 0.0), sim_freq=60, reuse_fdm=True,
        num_missiles=0)
    model = TorchF16BatchDynamics().to(device).eval()
    position = torch.tensor(
        [[0.0, 0.0, -altitude_m]], device=device)
    state = model.level_state(
        position, torch.zeros(1, device=device),
        torch.full((1,), speed_m_s, device=device))
    control = torch.tensor([command], dtype=torch.float32, device=device)
    simulator.set_property_values([
        c.fcs_aileron_cmd_norm, c.fcs_elevator_cmd_norm,
        c.fcs_rudder_cmd_norm, c.fcs_throttle_cmd_norm,
    ], command)
    steps = int(round(seconds * 60.0))
    checkpoints = {
        max(1, int(round(value * 60.0))): value
        for value in (1.0, 5.0, seconds) if value <= seconds
    }
    samples = []
    try:
        with torch.inference_mode():
            for step in range(1, steps + 1):
                simulator.run()
                state = model(state, control, substeps=1)
                if step not in checkpoints:
                    continue
                js_position, js_velocity, js_rpy, js_rates, js_body, js_control = (
                    jsbsim_observables(simulator))
                th_position, th_velocity, th_rpy, th_rates = (
                    torch_observables(state))
                attitude_error = np.asarray([
                    wrap_angle(th_rpy[index] - js_rpy[index])
                    for index in range(3)
                ])
                samples.append({
                    "time_s": checkpoints[step],
                    "jsbsim_position_neu_m": js_position.tolist(),
                    "torch_position_neu_m": th_position.tolist(),
                    "jsbsim_velocity_neu_m_s": js_velocity.tolist(),
                    "torch_velocity_neu_m_s": th_velocity.tolist(),
                    "jsbsim_rpy_deg": np.rad2deg(js_rpy).tolist(),
                    "torch_rpy_deg": np.rad2deg(th_rpy).tolist(),
                    "jsbsim_body_rates_rad_s": js_rates.tolist(),
                    "torch_body_rates_rad_s": th_rates.tolist(),
                    "jsbsim_surface_norm": js_control.tolist(),
                    "torch_actuator_norm": state[0, 13:17].detach().cpu().tolist(),
                    "position_error_m": float(np.linalg.norm(
                        th_position - js_position)),
                    "velocity_error_m_s": float(np.linalg.norm(
                        th_velocity - js_velocity)),
                    "body_velocity_error_m_s": float(np.linalg.norm(
                        state[0, 3:6].detach().cpu().numpy() - js_body)),
                    "attitude_error_deg": float(np.rad2deg(
                        np.linalg.norm(attitude_error))),
                    "body_rate_error_rad_s": float(np.linalg.norm(
                        th_rates - js_rates)),
                    "torch_finite": bool(torch.isfinite(state).all()),
                })
    finally:
        simulator.close()
    return {"case": name, "control": command, "samples": samples}


def main():
    args = parse_args()
    device = torch.device(args.device)
    results = [
        run_case(name, command, args.seconds, device)
        for name, command in CASES.items()
    ]
    finals = [result["samples"][-1] for result in results]
    # Conservative initial acceptance gates for policy-transfer experiments.
    gates = {
        "position_error_m_at_10s_max": 500.0,
        "velocity_error_m_s_at_10s_max": 25.0,
        "attitude_error_deg_at_10s_max": 15.0,
    }
    passed = all(
        sample["position_error_m"] <= gates["position_error_m_at_10s_max"]
        and sample["velocity_error_m_s"] <= gates[
            "velocity_error_m_s_at_10s_max"]
        and sample["attitude_error_deg"] <= gates[
            "attitude_error_deg_at_10s_max"]
        and sample["torch_finite"]
        for sample in finals
    )
    report = {
        "backend": "experimental_torch6dof_v1",
        "reference": "JSBSim F-16A Block-32 XML",
        "sim_freq_hz": 60,
        "duration_s": args.seconds,
        "acceptance_gates": gates,
        "passed": passed,
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
