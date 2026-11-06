"""Evaluate a frozen 1v1 PPO checkpoint against a fixed scripted opponent."""

from __future__ import annotations

import argparse
import csv
from functools import partial
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.train_ppo_1v1_shoot_smoke import (
    MultiDiscreteActorCritic,
    make_environment,
    select_device,
)
from envs.env_wrappers import DummyVecEnv, SubprocVecEnv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=96)
    parser.add_argument("--seed", type=int, default=950085)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"),
                        default="auto")
    parser.add_argument("--start-method", choices=("spawn", "forkserver"),
                        default="spawn")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--opponent-policy", choices=(
            "pursuit_fire", "hold", "evasive_fire", "standoff_fire"),
        default="pursuit_fire")
    return parser.parse_args()


def scripted_pursuit_fire(observations: np.ndarray) -> np.ndarray:
    """Deterministic high-level pursuit with pre-action legal fire masking."""

    observations = np.asarray(observations, dtype=np.float32)
    count = observations.shape[0]
    actions = np.empty((count, 4), dtype=np.int64)
    actions[:, 0] = 1  # hold altitude
    actions[:, 1] = 2  # hold heading when nearly nose-on
    off_boresight = observations[:, 11]
    side = observations[:, 14]
    turning = off_boresight > np.deg2rad(8.0)
    actions[turning & (side >= 0.0), 1] = 4
    actions[turning & (side < 0.0), 1] = 0
    actions[:, 2] = np.where(observations[:, 13] > 0.8, 0, 1)
    actions[:, 3] = (observations[:, 23] >= 0.5).astype(np.int64)
    return actions


def scripted_hold(observations: np.ndarray) -> np.ndarray:
    """Deterministic non-firing straight-and-level reference opponent."""

    return np.tile(
        np.asarray([1, 2, 1, 0], dtype=np.int64),
        (len(observations), 1))


def scripted_evasive_fire(observations: np.ndarray) -> np.ndarray:
    """Pursue normally, then beam away and change altitude under warning."""

    observations = np.asarray(observations, dtype=np.float32)
    actions = scripted_pursuit_fire(observations)
    warning = observations[:, 19] > 0.0
    missile_side = observations[:, 20]
    missile_delta_altitude = observations[:, 16]
    actions[warning & (missile_side >= 0.0), 1] = 0
    actions[warning & (missile_side < 0.0), 1] = 4
    actions[warning & (missile_delta_altitude >= 0.0), 0] = 2
    actions[warning & (missile_delta_altitude < 0.0), 0] = 0
    actions[warning, 2] = 0
    return actions


def scripted_evasive_fire_break_skill(observations: np.ndarray) -> np.ndarray:
    """Evasive reference using action 0 as threat-relative break-away."""
    observations = np.asarray(observations, dtype=np.float32)
    actions = scripted_evasive_fire(observations)
    warning = (observations[:, 24] >= 0.5 if observations.shape[1] > 24
               else observations[:, 19] > 0.0)
    actions[warning, 1] = 0
    return actions


def scripted_standoff_fire(observations: np.ndarray) -> np.ndarray:
    """Fire when legal but turn away if the opponent closes inside 9 km."""

    observations = np.asarray(observations, dtype=np.float32)
    actions = scripted_pursuit_fire(observations)
    close = observations[:, 13] < 0.9
    target_side = observations[:, 14]
    actions[close & (target_side >= 0.0), 1] = 0
    actions[close & (target_side < 0.0), 1] = 4
    actions[close, 2] = 2
    return actions


OPPONENT_POLICIES = {
    "pursuit_fire": scripted_pursuit_fire,
    "hold": scripted_hold,
    "evasive_fire": scripted_evasive_fire,
    "standoff_fire": scripted_standoff_fire,
}


def _info_counts(info, learner_id):
    event_counts = {
        key: 0 for key in (
            "launches", "hits", "redundant_hits", "misses", "invalidated")
    }
    for event in info.get("weapon_events", ()):
        if (not isinstance(event, dict) or
                str(event.get("shooter_id")) != learner_id):
            continue
        event_type = str(event.get("event_type", ""))
        key = {
            "launch": "launches",
            "hit": "hits",
            "redundant_hit": "redundant_hits",
            "miss": "misses",
            "invalidated": "invalidated",
        }.get(event_type)
        if key is not None:
            event_counts[key] += 1
    legality = info.get("weapon_legality", {}).get(learner_id, {})
    return {
        **event_counts,
        "requests": int(bool(legality.get("requested", False))),
        "rejected": int(bool(legality.get("requested", False)) and
                        not bool(legality.get("accepted", False))),
        "opportunities": int(bool(legality.get("can_fire", False))),
    }


def evaluate_side(model, *, learner_side, episodes, num_envs, seed,
                  device, start_method, scenario, reward_variant,
                  opponent_policy, threat_observation_v2=False,
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
    factories = [
        partial(
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
        for index in range(num_envs)
    ]
    env = (DummyVecEnv(factories) if num_envs == 1 else
           SubprocVecEnv(factories, context=start_method, in_series=1))
    obs = env.reset()
    agent_ids = ("A0100", "B0100")
    learner_index = 0 if learner_side == "ego" else 1
    opponent_index = 1 - learner_index
    learner_id = agent_ids[learner_index]
    accumulators = {
        key: np.zeros(num_envs, dtype=np.int64)
        for key in (
            "launches", "hits", "redundant_hits", "misses", "invalidated",
            "requests", "rejected", "opportunities")
    }
    first_launch_steps = np.full(num_envs, -1, dtype=np.int64)
    opportunities_before_first_launch = np.zeros(num_envs, dtype=np.int64)
    records = []
    try:
        while len(records) < episodes:
            learned_obs = torch.as_tensor(
                obs[:, learner_index], dtype=torch.float32, device=device)
            with torch.inference_mode():
                learned_action, _, _ = model.act(
                    learned_obs, deterministic=True)
            scripted_action = opponent_policy(obs[:, opponent_index])
            joint = np.empty((num_envs, 2, 4), dtype=np.int64)
            joint[:, learner_index] = learned_action.cpu().numpy()
            joint[:, opponent_index] = scripted_action
            obs, _reward, done, infos = env.step(joint)
            done = np.asarray(done, dtype=np.bool_).reshape(num_envs, 2)
            flat_infos = np.asarray(infos, dtype=object).reshape(-1)
            for index, info in enumerate(flat_infos):
                if not isinstance(info, dict):
                    continue
                counts = _info_counts(info, learner_id)
                if first_launch_steps[index] < 0:
                    opportunities_before_first_launch[index] += counts[
                        "opportunities"]
                    if counts["launches"]:
                        first_launch_steps[index] = int(
                            info.get("current_step", 0))
                for key, value in counts.items():
                    accumulators[key][index] += value
            for index in np.flatnonzero(np.all(done, axis=1)):
                if len(records) >= episodes:
                    break
                info = flat_infos[int(index)]
                outcome = info.get("team_outcome", {})
                ego_result = int(outcome.get("ego_result", 0))
                result = ego_result if learner_index == 0 else -ego_result
                reason = str(outcome.get("reason", "unknown"))
                elimination = reason in {
                    "ego_team_eliminated", "enemy_team_eliminated",
                    "simultaneous_elimination",
                }
                statuses = info.get("aircraft_status", {})
                learner_status = statuses.get(learner_id, {})
                opponent_id = agent_ids[opponent_index]
                opponent_status = statuses.get(opponent_id, {})
                terminations = info.get("termination", {})
                learner_termination = terminations.get(learner_id, {})
                opponent_termination = terminations.get(opponent_id, {})
                def death_cause(status):
                    if bool(status.get("shotdown", False)):
                        return "shotdown"
                    if bool(status.get("crash", False)):
                        return "crash"
                    if bool(status.get("alive", False)):
                        return "alive"
                    return "unknown"
                unresolved = max(
                    0,
                    int(accumulators["launches"][index]) -
                    int(accumulators["hits"][index]) -
                    int(accumulators["redundant_hits"][index]) -
                    int(accumulators["misses"][index]) -
                    int(accumulators["invalidated"][index]),
                )
                records.append({
                    "learner_side": learner_side,
                    "result": result,
                    "label": "win" if result > 0 else (
                        "loss" if result < 0 else "draw"),
                    "reason": reason,
                    "kill_result": result if elimination else 0,
                    "timeout_result": (
                        result if reason.startswith("timeout") else 0),
                    "episode_steps": int(outcome.get("episode_steps", 0)),
                    "first_launch_step": int(first_launch_steps[index]),
                    "opportunities_before_first_launch": int(
                        opportunities_before_first_launch[index]),
                    "learner_death_cause": death_cause(learner_status),
                    "opponent_death_cause": death_cause(opponent_status),
                    "learner_termination_condition": str(
                        learner_termination.get("reason", "unknown")),
                    "opponent_termination_condition": str(
                        opponent_termination.get("reason", "unknown")),
                    "unresolved": unresolved,
                    **{key: int(values[index])
                       for key, values in accumulators.items()},
                })
                for values in accumulators.values():
                    values[index] = 0
                first_launch_steps[index] = -1
                opportunities_before_first_launch[index] = 0
    finally:
        env.close()
    return records


def main():
    args = parse_args()
    if args.episodes < 2 or args.num_envs <= 0:
        raise ValueError("episodes must be at least two and num-envs positive")
    payload = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    if config.get("schema") != "WVRPopulationPPOBaselineV2":
        raise ValueError("checkpoint does not use the V2 acceptance contract")
    device = select_device(args.device)
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
    scenario = str(config["scenario"])
    reward_variant = config.get("reward_variant")
    threat_observation_v2 = bool(config.get("threat_observation_v2", False))
    heading_command_span_deg = float(
        config.get("action_interface", {}).get(
            "heading_command_span_deg", 30.0))
    threat_relative_break_skill = bool(
        config.get("action_interface", {}).get(
            "threat_relative_break_skill", False))
    threat_break_option = bool(config.get("action_interface", {}).get(
        "threat_break_option", False))
    threat_break_min_steps = int(config.get("action_interface", {}).get(
        "threat_break_min_steps", 10))
    threat_break_clear_steps = int(config.get("action_interface", {}).get(
        "threat_break_clear_steps", 5))
    threat_break_recovery_steps = int(config.get("action_interface", {}).get(
        "threat_break_recovery_steps", 10))
    threat_break_recovery_ta_max_deg = float(
        config.get("action_interface", {}).get(
            "threat_break_recovery_ta_max_deg", 180.0))
    threat_break_pulse = bool(config.get("action_interface", {}).get(
        "threat_break_pulse", False))
    threat_break_pulse_steps = int(config.get("action_interface", {}).get(
        "threat_break_pulse_steps", 4))
    threat_break_pulse_urgency = float(
        config.get("action_interface", {}).get(
            "threat_break_pulse_urgency", 0.5))
    opponent_policy = OPPONENT_POLICIES[args.opponent_policy]
    if (threat_relative_break_skill and
            args.opponent_policy == "evasive_fire"):
        opponent_policy = scripted_evasive_fire_break_skill

    first = args.episodes // 2
    records = evaluate_side(
        model, learner_side="ego", episodes=first,
        num_envs=args.num_envs, seed=args.seed, device=device,
        start_method=args.start_method, scenario=scenario,
        reward_variant=reward_variant, opponent_policy=opponent_policy,
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
    records.extend(evaluate_side(
        model, learner_side="enemy", episodes=args.episodes - first,
        num_envs=args.num_envs, seed=args.seed + 500_000, device=device,
        start_method=args.start_method, scenario=scenario,
        reward_variant=reward_variant, opponent_policy=opponent_policy,
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
        threat_break_pulse_urgency=threat_break_pulse_urgency))

    wins = sum(item["result"] > 0 for item in records)
    draws = sum(item["result"] == 0 for item in records)
    losses = sum(item["result"] < 0 for item in records)
    timeouts = sum(item["reason"].startswith("timeout") for item in records)
    kill_wins = sum(item["kill_result"] > 0 for item in records)
    kill_draws = sum(item["kill_result"] == 0 for item in records)
    kill_losses = sum(item["kill_result"] < 0 for item in records)
    timeout_wins = sum(item["timeout_result"] > 0 for item in records)
    timeout_losses = sum(item["timeout_result"] < 0 for item in records)
    termination_reason_counts = {}
    for item in records:
        reason = item["reason"]
        termination_reason_counts[reason] = (
            termination_reason_counts.get(reason, 0) + 1)
    launches = sum(item["launches"] for item in records)
    requests = sum(item["requests"] for item in records)
    rejected = sum(item["rejected"] for item in records)
    opportunities = sum(item["opportunities"] for item in records)
    hits = sum(item["hits"] for item in records)
    redundant_hits = sum(item["redundant_hits"] for item in records)
    misses = sum(item["misses"] for item in records)
    invalidated = sum(item["invalidated"] for item in records)
    unresolved = sum(item["unresolved"] for item in records)
    resolved = hits + redundant_hits + misses + invalidated
    launched_episodes = sum(item["launches"] > 0 for item in records)
    first_launch_values = [
        item["first_launch_step"] for item in records
        if item["first_launch_step"] >= 0]
    learner_death_causes = {}
    opponent_death_causes = {}
    learner_termination_conditions = {}
    opponent_termination_conditions = {}
    for item in records:
        for target, key in (
                (learner_death_causes, item["learner_death_cause"]),
                (opponent_death_causes, item["opponent_death_cause"])):
            target[key] = target.get(key, 0) + 1
        for target, key in (
                (learner_termination_conditions,
                 item["learner_termination_condition"]),
                (opponent_termination_conditions,
                 item["opponent_termination_condition"])):
            target[key] = target.get(key, 0) + 1
    summary = {
        "schema": "WVR1v1AcceptanceEvaluationV1",
        "checkpoint": str(args.checkpoint.resolve()),
        "opponent": f"scripted_{args.opponent_policy}_v1",
        "scenario": scenario,
        "reward_variant": reward_variant,
        "episodes": len(records),
        "side_balanced": True,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / len(records),
        "draw_rate": draws / len(records),
        "loss_rate": losses / len(records),
        "kill_wins": kill_wins,
        "kill_draws": kill_draws,
        "kill_losses": kill_losses,
        "kill_win_rate": kill_wins / len(records),
        "kill_loss_rate": kill_losses / len(records),
        "timeout_adjudicated_wins": timeout_wins,
        "timeout_adjudicated_losses": timeout_losses,
        "timeouts": timeouts,
        "timeout_rate": timeouts / len(records),
        "termination_reason_counts": termination_reason_counts,
        "timeout_tactical_advantage": termination_reason_counts.get(
            "timeout_tactical_advantage", 0),
        "timeout_tactical_tie": termination_reason_counts.get(
            "timeout_tactical_tie", 0),
        "timeout_plain": termination_reason_counts.get("timeout", 0),
        "learner_launches": launches,
        "learner_hits": hits,
        "learner_redundant_hits": redundant_hits,
        "learner_misses": misses,
        "learner_invalidated": invalidated,
        "learner_unresolved": unresolved,
        "learner_resolved_weapon_outcomes": resolved,
        "weapon_event_conservation_ok": (
            launches == resolved + unresolved),
        "resolved_weapon_hit_rate": (
            (hits + redundant_hits) / resolved if resolved else None),
        "learner_fire_requests": requests,
        "learner_rejected_requests": rejected,
        "legal_launch_acceptance_rate": (
            (requests - rejected) / requests if requests else None),
        "learner_legal_opportunities": opportunities,
        "legal_opportunity_exploitation_rate": (
            launches / opportunities if opportunities else None),
        "episodes_with_launch_rate": launched_episodes / len(records),
        "mean_first_launch_step": (
            float(np.mean(first_launch_values))
            if first_launch_values else None),
        "mean_opportunities_before_first_launch": float(np.mean([
            item["opportunities_before_first_launch"] for item in records
        ])),
        "learner_death_causes": learner_death_causes,
        "opponent_death_causes": opponent_death_causes,
        "learner_termination_conditions": learner_termination_conditions,
        "opponent_termination_conditions": opponent_termination_conditions,
        "mean_episode_steps": float(np.mean([
            item["episode_steps"] for item in records])),
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
