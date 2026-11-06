#!/usr/bin/env python3
"""Paired TorchWVR snapshot intervention diagnostic for V7.1 only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from benchmarks.benchmark_torch6dof_ppo import lowlevel_controls
from benchmarks.drvoi.targets import TargetStandardizer, response_state_v7
from benchmarks.drvoi_v7_1.history import NEUTRAL_ACTION
from benchmarks.drvoi_v7_1.snapshot import capture_wvr_snapshot, restore_wvr_snapshot
from benchmarks.train_torch_wvr_ppo import fixed_mixture_actions
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.utils.utils import get_root_dir
from envs.torch6dof import TorchWVRConfig, TorchWVRVecEnv


HEAD_NAMES = ("alt", "heading", "speed", "fire")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20185)
    parser.add_argument("--standardizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_probe_actions(num_envs: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    neutral = torch.tensor(NEUTRAL_ACTION, device=device).expand(num_envs, -1).clone()
    head = torch.arange(num_envs, device=device).remainder(4)
    variant = (torch.arange(num_envs, device=device) // 4)
    actions = neutral.clone()
    actions[head == 0, 0] = torch.where(
        variant[head == 0].remainder(2) == 0, 0, 2)
    heading_codes = torch.tensor([0, 1, 3, 4], device=device)
    actions[head == 1, 1] = heading_codes[
        variant[head == 1].remainder(4)]
    actions[head == 2, 2] = torch.where(
        variant[head == 2].remainder(2) == 0, 0, 2)
    actions[head == 3, 3] = 1
    return actions, head


def load_scale(path: Path, device) -> torch.Tensor:
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("standardizer")
    if state is None and "primary" in payload:
        state = payload["primary"]["standardizer"]
    if state is None:
        raise ValueError("artifact does not contain a response standardizer")
    return TargetStandardizer.from_state_dict(state).scale.to(device)


def summarize_effect(effect, actions, head):
    result = {}
    for head_index, name in enumerate(HEAD_NAMES):
        selected = head == head_index
        values = effect[selected]
        action_codes = actions[selected, head_index]
        by_code = {}
        for code in torch.unique(action_codes).tolist():
            code_values = values[action_codes == code]
            mean = code_values.mean(0)
            consistency = mean.abs() / code_values.abs().mean(0).clamp_min(1e-12)
            by_code[str(int(code))] = {
                "count": int(code_values.shape[0]),
                "mean_by_dimension": mean.cpu().tolist(),
                "rms_by_dimension": code_values.square().mean(0).sqrt().cpu().tolist(),
                "direction_consistency_by_dimension": consistency.cpu().tolist(),
            }
        result[name] = {
            "rms_by_dimension": values.square().mean(0).sqrt().cpu().tolist(),
            "rms_all_dimensions": float(values.square().mean().sqrt()),
            "by_action_code": by_code,
        }
    return result


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.num_envs % 24:
        raise ValueError("num-envs must be divisible by 24 for balanced probes")
    device = torch.device("cuda")
    env = TorchWVRVecEnv(TorchWVRConfig(
        num_envs=args.num_envs, reward_mode="competitive",
        attack_quality_weight=0.0, observation_version="v4",
        use_dynamic_launch_envelope=True),
        device=device, seed=args.seed, compile_dynamics=True)
    lowlevel = BaselineActor().to(device).eval()
    lowlevel.load_state_dict(torch.load(
        get_root_dir() + "/model/baseline_model.pt",
        map_location=device, weights_only=True))
    lowlevel.requires_grad_(False)
    lowlevel.tpdv = {"dtype": torch.float32, "device": device}
    lowlevel_hidden = torch.zeros(args.num_envs * 2, 1, 128, device=device)
    env_index = torch.arange(args.num_envs, device=device)
    learner_side = env_index.remainder(2)
    opponent_side = 1 - learner_side
    opponent_kind = (env_index // 2).remainder(3)
    probe_actions, probe_head = build_probe_actions(args.num_envs, device)
    neutral = torch.tensor(NEUTRAL_ACTION, device=device).expand(args.num_envs, -1)
    scale = load_scale(args.standardizer, device)
    snapshot = capture_wvr_snapshot(
        env, lowlevel_hidden, probe_actions=probe_actions, probe_head=probe_head)

    neutral_future = {}
    opponent_controls = []
    hidden, _ = restore_wvr_snapshot(env, snapshot)
    with torch.inference_mode():
        for horizon in range(1, 6):
            obs = env.observations()
            actions = fixed_mixture_actions(obs, opponent_kind)
            actions[env_index, learner_side] = neutral
            controls, hidden = lowlevel_controls(
                lowlevel, hidden, obs, actions, torch_speed_adapter=True)
            controls = controls.reshape(args.num_envs, 2, 4)
            opponent_controls.append(
                controls[env_index, opponent_side].clone())
            obs, _, done, _ = env.step(
                controls, torch.zeros(args.num_envs, 2, dtype=torch.bool, device=device))
            if bool(done.any()):
                raise RuntimeError("paired probe crossed an episode boundary")
            if horizon in (1, 3, 5):
                neutral_future[horizon] = response_state_v7(
                    obs[env_index, learner_side], "rel")

        hidden, extra = restore_wvr_snapshot(env, snapshot)
        effects = {}
        for horizon in range(1, 6):
            obs = env.observations()
            actions = fixed_mixture_actions(obs, opponent_kind)
            actions[env_index, learner_side] = (
                probe_actions if horizon == 1 else neutral)
            controls, hidden = lowlevel_controls(
                lowlevel, hidden, obs, actions, torch_speed_adapter=True)
            controls = controls.reshape(args.num_envs, 2, 4)
            controls[env_index, opponent_side] = opponent_controls[horizon - 1]
            fire = torch.zeros(args.num_envs, 2, dtype=torch.bool, device=device)
            if horizon == 1:
                fire[env_index, learner_side] = probe_actions[:, 3] == 1
            obs, _, done, _ = env.step(controls, fire)
            if bool(done.any()):
                raise RuntimeError("paired probe crossed an episode boundary")
            if horizon in (1, 3, 5):
                candidate = response_state_v7(obs[env_index, learner_side], "rel")
                effects[horizon] = (candidate - neutral_future[horizon]) / scale

    report = {
        "suite": "DRVOI-RHPPO-V7.1-paired-intervention-diagnostic",
        "protocol_version": "DRVOI-RHPPO-V7.1-history-response-1",
        "scientific_result": False,
        "development_only": True,
        "connected_to_online_response_or_voi": False,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "standardizer": str(args.standardizer),
        "snapshot_fields": sorted(snapshot.environment),
        "same_rng_restored": True,
        "same_lowlevel_hidden_restored": True,
        "opponent_controls_fixed_from_neutral_branch": True,
        "effects": {
            str(horizon): summarize_effect(
                effect, extra["probe_actions"], extra["probe_head"])
            for horizon, effect in effects.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
