"""Closed-loop Torch-to-JSBSim policy transfer audit.

The same deterministic high-level checkpoint and frozen recurrent low-level
controller drive a Torch pair and a JSBSim pair from matched initial states.
This is an engineering transfer gate, not a combat-performance result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.benchmark_torch6dof_ppo import (
    lowlevel_controls, observations,
)
from benchmarks.train_ppo_1v1_shoot_smoke import MultiDiscreteActorCritic
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.core.simulatior import AircraftSimulator
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.utils.utils import get_root_dir
from envs.torch6dof import CompiledTorchF16Runner, TorchF16BatchDynamics
from envs.torch6dof.f16_dynamics import quaternion_to_matrix


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--decisions", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction,
                        default=True)
    return parser.parse_args()


def quaternion_from_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.asarray([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float32)


def proxy_state(simulators, device):
    rows = []
    for simulator in simulators:
        position_neu = np.asarray(simulator.get_position(), dtype=np.float32)
        position_ned = position_neu.copy()
        position_ned[2] *= -1.0
        body_velocity = np.asarray(simulator.get_property_values([
            c.velocities_u_mps, c.velocities_v_mps, c.velocities_w_mps,
        ]), dtype=np.float32)
        rpy = np.asarray(simulator.get_rpy(), dtype=np.float32)
        rates = np.asarray(simulator.get_property_values([
            c.velocities_p_rad_sec, c.velocities_q_rad_sec,
            c.velocities_r_rad_sec,
        ]), dtype=np.float32)
        tail = np.zeros(7, dtype=np.float32)
        rows.append(np.concatenate((
            position_ned, body_velocity, quaternion_from_rpy(rpy), rates,
            tail)))
    return torch.as_tensor(np.asarray(rows), device=device).reshape(1, 2, -1)


def observables(state):
    rotation = quaternion_to_matrix(state[..., 6:10])
    velocity_ned = torch.matmul(
        rotation, state[..., 3:6].unsqueeze(-1)).squeeze(-1)
    rpy = torch.stack((
        torch.atan2(rotation[..., 2, 1], rotation[..., 2, 2]),
        torch.asin((-rotation[..., 2, 0]).clamp(-1.0, 1.0)),
        torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])), dim=-1)
    return state[..., :3], velocity_ned, rpy


def wrap_angle(value):
    return torch.remainder(value + torch.pi, 2.0 * torch.pi) - torch.pi


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    payload = torch.load(args.checkpoint, map_location=device,
                         weights_only=False)
    policy = MultiDiscreteActorCritic(
        obs_dim=int(payload["obs_dim"]), action_dims=payload["action_dims"],
        hidden_size=int(payload["hidden_size"]),
        fire_mask_obs_index=23).to(device).eval()
    policy.load_state_dict(payload["model"])
    lowlevel = BaselineActor().to(device).eval()
    lowlevel.load_state_dict(torch.load(
        get_root_dir() + "/model/baseline_model.pt",
        map_location=device, weights_only=True))
    lowlevel.requires_grad_(False)
    lowlevel.tpdv = {"dtype": torch.float32, "device": device}
    dynamics = TorchF16BatchDynamics().to(device).eval()
    runner = CompiledTorchF16Runner(dynamics) if args.compile else dynamics

    origin = (120.0, 60.0, 0.0)
    initial = [
        {"ic_long_gc_deg": 120.0, "ic_lat_geod_deg": 59.955,
         "ic_h_sl_ft": 20000.0, "ic_psi_true_deg": 0.0,
         "ic_u_fps": 800.0},
        {"ic_long_gc_deg": 120.0, "ic_lat_geod_deg": 60.045,
         "ic_h_sl_ft": 20000.0, "ic_psi_true_deg": 180.0,
         "ic_u_fps": 800.0},
    ]
    simulators = [
        AircraftSimulator(
            uid=uid, color=color, model="f16", init_state=state,
            origin=origin, sim_freq=60, reuse_fdm=True, num_missiles=0)
        for uid, color, state in zip(("A0100", "B0100"),
                                     ("Red", "Blue"), initial)
    ]
    torch_state = proxy_state(simulators, device)
    torch_rnn = torch.zeros(2, 1, 128, device=device)
    js_rnn = torch.zeros_like(torch_rnn)
    head_equal = torch.zeros(4, device=device)
    first_divergence = None
    samples = []
    try:
        with torch.inference_mode():
            # Compile outside the measured/audited trajectory.
            runner(torch_state.reshape(-1, dynamics.state_size),
                   torch.tensor([[0.0, 0.0, 0.0, 0.7]], device=device
                                ).repeat(2, 1), substeps=1)
            for decision in range(args.decisions):
                js_state = proxy_state(simulators, device)
                torch_obs = observations(torch_state)
                js_obs = observations(js_state)
                torch_action, _, _ = policy.act(
                    torch_obs.reshape(-1, 24), deterministic=True)
                js_action, _, _ = policy.act(
                    js_obs.reshape(-1, 24), deterministic=True)
                equal = (torch_action == js_action).float().mean(0)
                head_equal += equal
                if first_divergence is None and not bool(
                        torch.all(torch_action == js_action)):
                    first_divergence = decision
                torch_control, torch_rnn = lowlevel_controls(
                    lowlevel, torch_rnn, torch_obs,
                    torch_action.reshape(1, 2, 4))
                js_control, js_rnn = lowlevel_controls(
                    lowlevel, js_rnn, js_obs,
                    js_action.reshape(1, 2, 4))
                torch_state = runner(
                    torch_state.reshape(-1, dynamics.state_size),
                    torch_control, substeps=12).reshape(
                        1, 2, dynamics.state_size)
                for index, simulator in enumerate(simulators):
                    simulator.set_property_values([
                        c.fcs_aileron_cmd_norm, c.fcs_elevator_cmd_norm,
                        c.fcs_rudder_cmd_norm, c.fcs_throttle_cmd_norm,
                    ], js_control[index].detach().cpu().tolist())
                for _ in range(12):
                    for simulator in simulators:
                        simulator.run()
                js_state = proxy_state(simulators, device)
                t_pos, t_vel, t_rpy = observables(torch_state)
                j_pos, j_vel, j_rpy = observables(js_state)
                sample = {
                    "decision": decision + 1,
                    "time_s": (decision + 1) * 0.2,
                    "position_error_m": float(torch.linalg.vector_norm(
                        t_pos - j_pos, dim=-1).mean()),
                    "velocity_error_m_s": float(torch.linalg.vector_norm(
                        t_vel - j_vel, dim=-1).mean()),
                    "attitude_error_deg": float(torch.rad2deg(
                        torch.linalg.vector_norm(
                            wrap_angle(t_rpy - j_rpy), dim=-1)).mean()),
                    "finite": bool(torch.isfinite(torch_state).all()),
                }
                if decision in (0, 24, 49, 99, args.decisions - 1):
                    samples.append(sample)
    finally:
        for simulator in simulators:
            simulator.close()

    final = sample
    agreement = (head_equal / args.decisions).detach().cpu().tolist()
    gates = {
        "maneuver_head_agreement_min": 0.80,
        "final_position_error_m_max": 1000.0,
        "final_attitude_error_deg_max": 30.0,
    }
    passed = (
        min(agreement[:3]) >= gates["maneuver_head_agreement_min"]
        and final["position_error_m"] <= gates["final_position_error_m_max"]
        and final["attitude_error_deg"] <= gates[
            "final_attitude_error_deg_max"]
        and final["finite"])
    report = {
        "checkpoint": str(args.checkpoint),
        "source_backend": payload.get("backend"),
        "target_backend": "JSBSim F-16",
        "scientific_result": False,
        "sim_freq_hz": 60,
        "decisions": args.decisions,
        "duration_s": args.decisions * 0.2,
        "action_head_agreement": dict(zip(
            ("altitude", "heading", "velocity", "fire"), agreement)),
        "first_action_divergence_decision": first_divergence,
        "acceptance_gates": gates,
        "passed": passed,
        "final": final,
        "samples": samples,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
