"""Evaluate two frozen 1v1 PPO checkpoints with physical-side balancing."""

from __future__ import annotations

import argparse
import csv
from functools import partial
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.train_ppo_1v1_shoot_smoke import (
    MultiDiscreteActorCritic, make_environment, select_device)
from envs.env_wrappers import DummyVecEnv, SubprocVecEnv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learner-checkpoint", type=Path, required=True)
    parser.add_argument("--opponent-checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=192)
    parser.add_argument("--num-envs", type=int, default=96)
    parser.add_argument("--seed", type=int, default=880085)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"),
                        default="auto")
    parser.add_argument("--start-method", choices=("spawn", "forkserver"),
                        default="forkserver")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_model(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    if config.get("schema") != "WVRPopulationPPOBaselineV2":
        raise ValueError(f"unsupported checkpoint contract: {path}")
    model = MultiDiscreteActorCritic(
        obs_dim=int(config["obs_dim"]),
        action_dims=[int(value) for value in config["action_dims"]],
        hidden_size=int(config["hidden_size"]),
        fire_mask_obs_index=int(config["fire_mask_obs_index"]),
        altitude_safety_obs_index=(
            int(config.get("altitude_safety_obs_index", 0))
            if config.get("altitude_safety_mask", False) else None),
        altitude_descend_mask_threshold=float(
            config.get("altitude_descend_mask_threshold", 0.8)),
        altitude_force_climb_threshold=float(
            config.get("altitude_force_climb_threshold", 0.65)),
    ).to(device)
    model.load_state_dict(payload["learner"])
    model.eval()
    model.requires_grad_(False)
    return model, config


def evaluate_side(learner, opponent, *, learner_side, episodes, num_envs,
                  seed, device, start_method, scenario, reward_variant,
                  threat_observation_v2=False,
                  heading_command_span_deg=30.0,
                  threat_relative_break_skill=False,
                  threat_break_option=False,
                  threat_break_min_steps=10,
                  threat_break_clear_steps=5,
                  threat_break_recovery_steps=10,
                  threat_break_recovery_ta_max_deg=180.0,
                  threat_break_pulse=False,
                  threat_break_pulse_steps=4,
                  threat_break_pulse_urgency=0.5):
    factories = [partial(
        make_environment, seed + index * 1009,
        scenario=scenario, reward_variant=reward_variant,
        threat_observation_v2=threat_observation_v2,
        heading_command_span_deg=heading_command_span_deg,
        threat_relative_break_skill=threat_relative_break_skill,
        threat_break_option=threat_break_option,
        threat_break_min_steps=threat_break_min_steps,
        threat_break_clear_steps=threat_break_clear_steps,
        threat_break_recovery_steps=threat_break_recovery_steps,
        threat_break_recovery_ta_max_deg=(
            threat_break_recovery_ta_max_deg),
        threat_break_pulse=threat_break_pulse,
        threat_break_pulse_steps=threat_break_pulse_steps,
        threat_break_pulse_urgency=threat_break_pulse_urgency)
        for index in range(num_envs)]
    env = (DummyVecEnv(factories) if num_envs == 1 else
           SubprocVecEnv(factories, context=start_method, in_series=1))
    obs = env.reset()
    learner_index = 0 if learner_side == "ego" else 1
    opponent_index = 1 - learner_index
    records = []
    try:
        while len(records) < episodes:
            with torch.inference_mode():
                learner_action, _, _ = learner.act(torch.as_tensor(
                    obs[:, learner_index], dtype=torch.float32,
                    device=device), deterministic=True)
                opponent_action, _, _ = opponent.act(torch.as_tensor(
                    obs[:, opponent_index], dtype=torch.float32,
                    device=device), deterministic=True)
            joint = np.empty((num_envs, 2, 4), dtype=np.int64)
            joint[:, learner_index] = learner_action.cpu().numpy()
            joint[:, opponent_index] = opponent_action.cpu().numpy()
            obs, _reward, done, infos = env.step(joint)
            done = np.asarray(done, dtype=np.bool_).reshape(num_envs, 2)
            flat_infos = np.asarray(infos, dtype=object).reshape(-1)
            for index in np.flatnonzero(np.all(done, axis=1)):
                if len(records) >= episodes:
                    break
                outcome = flat_infos[int(index)].get("team_outcome", {})
                ego_result = int(outcome.get("ego_result", 0))
                result = ego_result if learner_index == 0 else -ego_result
                reason = str(outcome.get("reason", "unknown"))
                elimination = reason in {
                    "ego_team_eliminated", "enemy_team_eliminated",
                    "simultaneous_elimination",
                }
                records.append({
                    "learner_side": learner_side,
                    "result": result,
                    "kill_result": result if elimination else 0,
                    "timeout_result": (
                        result if reason.startswith("timeout") else 0),
                    "reason": reason,
                    "episode_steps": int(outcome.get("episode_steps", 0)),
                })
    finally:
        env.close()
    return records


def main():
    args = parse_args()
    if args.episodes < 2 or args.num_envs <= 0:
        raise ValueError("episodes must be at least two and num-envs positive")
    device = select_device(args.device)
    learner, learner_config = load_model(args.learner_checkpoint, device)
    opponent, opponent_config = load_model(args.opponent_checkpoint, device)
    comparable = ("scenario", "reward_variant", "obs_dim", "action_dims",
                  "hidden_size", "fire_mask_obs_index", "action_interface")
    learner_config.setdefault("action_interface", {})
    opponent_config.setdefault("action_interface", {})
    if any(learner_config[key] != opponent_config[key] for key in comparable):
        raise ValueError("cross-play checkpoints are not environment-compatible")
    first = args.episodes // 2
    records = evaluate_side(
        learner, opponent, learner_side="ego", episodes=first,
        num_envs=args.num_envs, seed=args.seed, device=device,
        start_method=args.start_method, scenario=learner_config["scenario"],
        reward_variant=learner_config.get("reward_variant"),
        threat_observation_v2=bool(
            learner_config.get("threat_observation_v2", False)),
        heading_command_span_deg=float(
            learner_config["action_interface"].get(
                "heading_command_span_deg", 30.0)),
        threat_relative_break_skill=bool(
            learner_config["action_interface"].get(
                "threat_relative_break_skill", False)),
        threat_break_option=bool(learner_config["action_interface"].get(
            "threat_break_option", False)),
        threat_break_min_steps=int(learner_config["action_interface"].get(
            "threat_break_min_steps", 10)),
        threat_break_clear_steps=int(learner_config["action_interface"].get(
            "threat_break_clear_steps", 5)),
        threat_break_recovery_steps=int(
            learner_config["action_interface"].get(
                "threat_break_recovery_steps", 10)),
        threat_break_recovery_ta_max_deg=float(
            learner_config["action_interface"].get(
                "threat_break_recovery_ta_max_deg", 180.0)),
        threat_break_pulse=bool(learner_config["action_interface"].get(
            "threat_break_pulse", False)),
        threat_break_pulse_steps=int(
            learner_config["action_interface"].get(
                "threat_break_pulse_steps", 4)),
        threat_break_pulse_urgency=float(
            learner_config["action_interface"].get(
                "threat_break_pulse_urgency", 0.5)))
    records.extend(evaluate_side(
        learner, opponent, learner_side="enemy",
        episodes=args.episodes - first, num_envs=args.num_envs,
        seed=args.seed + 500_000, device=device,
        start_method=args.start_method, scenario=learner_config["scenario"],
        reward_variant=learner_config.get("reward_variant"),
        threat_observation_v2=bool(
            learner_config.get("threat_observation_v2", False)),
        heading_command_span_deg=float(
            learner_config["action_interface"].get(
                "heading_command_span_deg", 30.0)),
        threat_relative_break_skill=bool(
            learner_config["action_interface"].get(
                "threat_relative_break_skill", False)),
        threat_break_option=bool(learner_config["action_interface"].get(
            "threat_break_option", False)),
        threat_break_min_steps=int(learner_config["action_interface"].get(
            "threat_break_min_steps", 10)),
        threat_break_clear_steps=int(learner_config["action_interface"].get(
            "threat_break_clear_steps", 5)),
        threat_break_recovery_steps=int(
            learner_config["action_interface"].get(
                "threat_break_recovery_steps", 10)),
        threat_break_recovery_ta_max_deg=float(
            learner_config["action_interface"].get(
                "threat_break_recovery_ta_max_deg", 180.0)),
        threat_break_pulse=bool(learner_config["action_interface"].get(
            "threat_break_pulse", False)),
        threat_break_pulse_steps=int(
            learner_config["action_interface"].get(
                "threat_break_pulse_steps", 4)),
        threat_break_pulse_urgency=float(
            learner_config["action_interface"].get(
                "threat_break_pulse_urgency", 0.5))))
    wins = sum(record["result"] > 0 for record in records)
    draws = sum(record["result"] == 0 for record in records)
    losses = len(records) - wins - draws
    kill_wins = sum(record["kill_result"] > 0 for record in records)
    kill_losses = sum(record["kill_result"] < 0 for record in records)
    reasons = {}
    for record in records:
        reasons[record["reason"]] = reasons.get(record["reason"], 0) + 1
    summary = {
        "schema": "WVR1v1CheckpointCrossplayV1",
        "learner_checkpoint": str(args.learner_checkpoint.resolve()),
        "opponent_checkpoint": str(args.opponent_checkpoint.resolve()),
        "episodes": len(records), "side_balanced": True,
        "wins": wins, "draws": draws, "losses": losses,
        "win_rate": wins / len(records),
        "draw_rate": draws / len(records),
        "loss_rate": losses / len(records),
        "kill_wins": kill_wins,
        "kill_losses": kill_losses,
        "kill_win_rate": kill_wins / len(records),
        "kill_loss_rate": kill_losses / len(records),
        "timeout_rate": sum(reason.startswith("timeout") for reason in
                            (record["reason"] for record in records)) /
                        len(records),
        "mean_episode_steps": float(np.mean([
            record["episode_steps"] for record in records])),
        "termination_reason_counts": reasons,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "episodes.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
