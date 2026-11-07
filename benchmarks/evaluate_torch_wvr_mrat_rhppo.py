#!/usr/bin/env python3
"""Evaluate a frozen MRAT-RHPPO checkpoint against one fixed opponent."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from benchmarks.drvoi_v12 import (
    MRATArbitrator,
    MRATRouter,
    PersistentReachabilityShield,
    ROUTER_FEATURE_DIM,
    build_execution_features,
    build_router_features,
    conservative_router_logits,
    load_frozen_reachability_predictor,
    load_frozen_v9_execution,
)
from envs.torch6dof import TorchWVRConfig, TorchWVRVecEnv


PROTOCOL_VERSION = "MRAT-RHPPO-V12-FIXED-EVAL-2"
OPPONENT_ID = {"hold": 0, "pursuit": 1, "evasive": 2}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--v9-checkpoint", type=Path, required=True)
    parser.add_argument("--response-checkpoint", type=Path, required=True)
    parser.add_argument("--option-q-checkpoint", type=Path, required=True)
    parser.add_argument("--reachability-checkpoint", type=Path, required=True)
    parser.add_argument("--router-score-checkpoint", type=Path)
    parser.add_argument("--opponent", choices=tuple(OPPONENT_ID), required=True)
    parser.add_argument("--episodes", type=int, default=192)
    parser.add_argument("--num-envs", type=int, default=96)
    parser.add_argument("--seed", type=int, default=10085)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-records", type=Path)
    parser.add_argument("--segment-records", type=Path)
    parser.add_argument(
        "--routing-mode", choices=(
            "router", "shield_only", "gate_latched", "gate_sprt_latched"),
        default="router")
    parser.add_argument("--gate-sprt-evasive-rate", type=float, default=0.67)
    parser.add_argument("--gate-sprt-easy-rate", type=float, default=0.325)
    parser.add_argument("--gate-sprt-alpha", type=float, default=0.20)
    parser.add_argument("--gate-sprt-beta", type=float, default=0.20)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_hash(state: torch.Tensor) -> str:
    return hashlib.sha256(
        state.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(path)


def _segment_boundary(arbitration, reset: torch.Tensor) -> torch.Tensor:
    """Close an option on every new decision, including same-expert reuse."""
    return (arbitration.decision | arbitration.switched) & ~reset.bool()


def _routing_proposal(router_logits: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "router":
        return router_logits.argmax(-1)
    if mode == "shield_only":
        return torch.zeros(
            router_logits.shape[0], dtype=torch.long,
            device=router_logits.device)
    raise ValueError(f"unsupported routing mode: {mode}")


def _update_gate_latch(latched: torch.Tensor, gate_allowed: torch.Tensor,
                       reset: torch.Tensor) -> torch.Tensor:
    """Latch LOS for the rest of an episode after the first allowed gate."""
    if not (latched.shape == gate_allowed.shape == reset.shape):
        raise ValueError("gate latch inputs must have identical shapes")
    return torch.where(
        reset.bool(), gate_allowed.bool(),
        latched.bool() | gate_allowed.bool())


def _update_gate_sprt(
        evidence: torch.Tensor,
        gate_allowed: torch.Tensor,
        reset: torch.Tensor,
        evasive_rate: float,
        easy_rate: float,
        alpha: float,
        beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update a Bernoulli SPRT and return evidence plus latch trigger."""
    if not (evidence.shape == gate_allowed.shape == reset.shape):
        raise ValueError("gate SPRT inputs must have identical shapes")
    for name, value in (
            ("evasive_rate", evasive_rate), ("easy_rate", easy_rate),
            ("alpha", alpha), ("beta", beta)):
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must lie in (0,1)")
    if evasive_rate <= easy_rate:
        raise ValueError("evasive gate rate must exceed easy gate rate")
    evidence = torch.where(reset.bool(), torch.zeros_like(evidence), evidence)
    positive = torch.log(torch.as_tensor(
        evasive_rate / easy_rate, device=evidence.device,
        dtype=evidence.dtype))
    negative = torch.log(torch.as_tensor(
        (1.0 - evasive_rate) / (1.0 - easy_rate),
        device=evidence.device, dtype=evidence.dtype))
    evidence = evidence + torch.where(gate_allowed.bool(), positive, negative)
    upper = torch.log(torch.as_tensor(
        (1.0 - beta) / alpha, device=evidence.device,
        dtype=evidence.dtype))
    lower = torch.log(torch.as_tensor(
        beta / (1.0 - alpha), device=evidence.device,
        dtype=evidence.dtype))
    evidence = evidence.clamp_min(lower)
    return evidence, evidence >= upper


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("MRAT-RHPPO evaluation requires CUDA")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if payload.get("algorithm") != "mrat_rhppo":
        raise ValueError("checkpoint is not MRAT-RHPPO")
    checkpoint_hash = _sha256(args.checkpoint)
    config = payload["args"]
    reachability_hash = _sha256(args.reachability_checkpoint)
    router = MRATRouter(ROUTER_FEATURE_DIM, hidden_size=128).to(device).eval()
    router.load_state_dict(payload["router"])
    router.requires_grad_(False)
    reachability, reach_payload = load_frozen_reachability_predictor(
        args.reachability_checkpoint, device)
    router_score_path = args.router_score_checkpoint
    if router_score_path is None and config.get("router_score_checkpoint"):
        router_score_path = Path(config["router_score_checkpoint"])
    routing_predictor = reachability
    routing_payload = reach_payload
    routing_score_hash = None
    if router_score_path is not None:
        routing_predictor, routing_payload = (
            load_frozen_reachability_predictor(router_score_path, device))
        routing_score_hash = _sha256(router_score_path)
    v9_bundle = load_frozen_v9_execution(
        args.v9_checkpoint, args.response_checkpoint,
        args.option_q_checkpoint, device)
    executor = v9_bundle.executor
    shield = PersistentReachabilityShield(
        bad_threshold=float(config["bad_threshold"]),
        safe_threshold=float(config["safe_threshold"]),
        intercept_advantage=float(config["intercept_advantage"]),
        force_steps=int(config["force_steps"]),
        release_steps=int(config["release_steps"]))
    arbitrator = MRATArbitrator(
        minimum_dwell=int(config["minimum_dwell"]),
        maximum_dwell=int(config["maximum_dwell"]))
    environment_config = {
        "num_envs": args.num_envs,
        "reward_mode": "competitive",
        "attack_quality_weight": 0.0,
        "observation_version": "v4",
        "use_dynamic_launch_envelope": True,
        "sim_freq_hz": 60,
        "substeps_per_decision": 12,
        "max_episode_steps": 1000,
        "missiles_per_aircraft": 2,
    }
    environment_config_hash = hashlib.sha256(json.dumps(
        environment_config, sort_keys=True).encode()).hexdigest()
    env = TorchWVRVecEnv(TorchWVRConfig(
        num_envs=args.num_envs,
        reward_mode="competitive",
        attack_quality_weight=0.0,
        observation_version="v4",
        use_dynamic_launch_envelope=True,
    ), device=device, seed=args.seed, compile_dynamics=True)
    env_index = torch.arange(args.num_envs, device=device)
    learner_side = env_index.remainder(2)
    execution_state = executor.initial_state(
        args.num_envs, device, learner_side=learner_side,
        training_opponent_kind=torch.full(
            (args.num_envs,), OPPONENT_ID[args.opponent], device=device))

    episode_reward = torch.zeros(args.num_envs, device=device)
    episode_launches = torch.zeros(args.num_envs, dtype=torch.long, device=device)
    episode_hits = torch.zeros_like(episode_launches)
    episode_requests = torch.zeros_like(episode_launches)
    episode_rejected = torch.zeros_like(episode_launches)
    episode_legal = torch.zeros_like(episode_launches)
    episode_learned_steps = torch.zeros_like(episode_launches)
    episode_intercept_steps = torch.zeros_like(episode_launches)
    episode_forced_steps = torch.zeros_like(episode_launches)
    episode_switches = torch.zeros_like(episode_launches)
    episode_option_boundaries = torch.zeros_like(episode_launches)
    episode_same_expert_reselections = torch.zeros_like(episode_launches)
    episode_routing_gate_allowed = torch.zeros_like(episode_launches)
    episode_first_gate_allowed = torch.full_like(episode_launches, -1)
    episode_first_gate_latch = torch.full_like(episode_launches, -1)
    episode_first_intercept = torch.full_like(episode_launches, -1)
    initial_hash = [_state_hash(env.aircraft_state[i])
                    for i in range(args.num_envs)]
    initial_episode_id = env.episode_id.clone()
    previous_expert = torch.zeros(args.num_envs, dtype=torch.long, device=device)
    segment_start = env.episode_step.clone()
    segment_forced = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    raw_segments: list[dict] = []
    accepted_episode_keys: set[tuple[int, int]] = set()
    episode_records: list[dict] = []
    target_red = args.episodes // 2
    target_blue = args.episodes - target_red
    learner_red = learner_blue = 0
    decisions = 0
    lcb_learned_sum = lcb_intercept_sum = regret_sum = 0.0
    routing_lcb_learned_sum = routing_lcb_intercept_sum = 0.0
    routing_gate_allowed_steps = 0
    force_condition_steps = 0
    gate_latched = torch.zeros(
        args.num_envs, dtype=torch.bool, device=device)
    gate_sprt_evidence = torch.zeros(args.num_envs, device=device)

    with torch.inference_mode():
        while learner_red < target_red or learner_blue < target_blue:
            observation = env.observations()
            learner_observation = observation[env_index, learner_side]
            execution_features = build_execution_features(
                learner_observation, execution_state)
            reach = reachability.predict_both(execution_features)
            routing_score = (
                routing_predictor.predict_both(execution_features)
                if router_score_path is not None else reach)
            router_features = build_router_features(
                routing_score["standardized_features"], routing_score,
                horizon=float(routing_payload["censor_horizon"]))
            outputs = router(router_features)
            route_logits = outputs["router_logits"]
            if router_score_path is not None:
                route_logits, routing_gate_allowed = conservative_router_logits(
                    route_logits, routing_score["lcb"][:, 0],
                    routing_score["lcb"][:, 1],
                    float(config.get("router_intercept_margin", 0.0)))
            else:
                routing_gate_allowed = torch.ones(
                    args.num_envs, dtype=torch.bool, device=device)
            reset = execution_state["reset"].bool()
            if args.routing_mode == "gate_latched":
                gate_latched = _update_gate_latch(
                    gate_latched, routing_gate_allowed, reset)
                proposed = gate_latched.long()
            elif args.routing_mode == "gate_sprt_latched":
                gate_sprt_evidence, sprt_trigger = _update_gate_sprt(
                    gate_sprt_evidence, routing_gate_allowed, reset,
                    args.gate_sprt_evasive_rate, args.gate_sprt_easy_rate,
                    args.gate_sprt_alpha, args.gate_sprt_beta)
                gate_latched = _update_gate_latch(
                    gate_latched, sprt_trigger, reset)
                proposed = gate_latched.long()
            else:
                proposed = _routing_proposal(route_logits, args.routing_mode)
            if arbitrator.expert is None:
                current_expert = torch.zeros_like(proposed)
                current_dwell = torch.zeros_like(proposed)
            else:
                current_expert = arbitrator.expert
                current_dwell = arbitrator.dwell
            normalized_dwell = (
                current_dwell.float() /
                int(config["maximum_dwell"])).clamp(0.0, 1.0)
            termination = router.termination_distribution(
                router_features, current_expert, normalized_dwell).probs >= 0.5
            if args.routing_mode in ("gate_latched", "gate_sprt_latched"):
                termination = torch.zeros_like(termination)
            shield_step = shield.step(
                reach["lcb"][:, 0], reach["lcb"][:, 1], reset)
            arbitration = arbitrator.step(
                proposed, termination, shield_step.forced_intercept, reset)
            episode_step = env.episode_step.long()
            episode_first_gate_allowed = torch.where(
                (episode_first_gate_allowed < 0) & routing_gate_allowed,
                episode_step, episode_first_gate_allowed)
            episode_first_gate_latch = torch.where(
                (episode_first_gate_latch < 0) & gate_latched,
                episode_step, episode_first_gate_latch)
            episode_first_intercept = torch.where(
                (episode_first_intercept < 0)
                & (arbitration.expert == 1),
                episode_step, episode_first_intercept)

            closed = _segment_boundary(arbitration, reset)
            for index in torch.nonzero(closed, as_tuple=False).flatten().cpu().tolist():
                raw_segments.append({
                    "env_index": index,
                    "episode_id": int(initial_episode_id[index]),
                    "start_step": int(segment_start[index]),
                    "end_step": int(env.episode_step[index]),
                    "duration": int(env.episode_step[index] - segment_start[index]),
                    "expert": int(previous_expert[index]),
                    "forced": bool(segment_forced[index]),
                    "ended_by": (
                        "switch" if bool(arbitration.switched[index])
                        else "same_expert_reselection"),
                })
            segment_start = torch.where(
                closed | reset, env.episode_step, segment_start)
            previous_expert = torch.where(
                closed | reset, arbitration.expert, previous_expert)
            segment_forced = torch.where(
                closed | reset, arbitration.forced,
                segment_forced | arbitration.forced)

            execution_state["selected_expert"] = arbitration.expert.clone()
            execution_state["expert_age"] = (
                arbitration.dwell - 1).clamp_min(0)
            controls, fire, execution_state = executor.prepare_step(
                env, execution_state,
                torch.ones(args.num_envs, dtype=torch.bool, device=device))
            _, reward, done, info = env.step(controls, fire)
            execution_state = executor.post_step(
                env, execution_state, done, info)
            learner_reward = reward[env_index, learner_side]
            episode_reward += learner_reward
            episode_launches += info[
                "accepted_launches"][env_index, learner_side].long()
            episode_hits += info["hits"][env_index, learner_side].long()
            episode_requests += info[
                "fire_requests"][env_index, learner_side].long()
            episode_rejected += info[
                "rejected_fire_requests"][env_index, learner_side].long()
            episode_legal += info[
                "legal_fire_opportunities"][env_index, learner_side].long()
            episode_learned_steps += (arbitration.expert == 0).long()
            episode_intercept_steps += (arbitration.expert == 1).long()
            episode_forced_steps += arbitration.forced.long()
            episode_switches += (arbitration.switched & ~reset).long()
            episode_option_boundaries += closed.long()
            episode_same_expert_reselections += (
                arbitration.decision & ~arbitration.switched & ~reset).long()
            episode_routing_gate_allowed += routing_gate_allowed.long()
            decisions += args.num_envs
            lcb_learned_sum += float(reach["lcb"][:, 0].sum())
            lcb_intercept_sum += float(reach["lcb"][:, 1].sum())
            selected_lcb = reach["lcb"].gather(
                1, arbitration.expert[:, None]).squeeze(1)
            regret_sum += float((reach["lcb"].amax(-1) - selected_lcb).sum())
            force_condition_steps += int(shield_step.force_condition.sum())
            routing_lcb_learned_sum += float(
                routing_score["lcb"][:, 0].sum())
            routing_lcb_intercept_sum += float(
                routing_score["lcb"][:, 1].sum())
            routing_gate_allowed_steps += int(routing_gate_allowed.sum())
            if not bool(done.any()):
                continue
            for index in torch.nonzero(done, as_tuple=False).flatten().cpu().tolist():
                side = int(learner_side[index])
                record_episode = (
                    (side == 0 and learner_red < target_red)
                    or (side == 1 and learner_blue < target_blue))
                end_step = int(info["episode_length"][index])
                raw_segments.append({
                    "env_index": index,
                    "episode_id": int(initial_episode_id[index]),
                    "start_step": int(segment_start[index]),
                    "end_step": end_step,
                    "duration": end_step - int(segment_start[index]),
                    "expert": int(previous_expert[index]),
                    "forced": bool(segment_forced[index]),
                    "ended_by": "episode_terminal",
                })
                if record_episode:
                    if side == 0:
                        learner_red += 1
                    else:
                        learner_blue += 1
                    outcome_red = float(info["outcome_red"][index])
                    outcome = outcome_red if side == 0 else -outcome_red
                    key = (index, int(initial_episode_id[index]))
                    accepted_episode_keys.add(key)
                    episode_records.append({
                        "env_index": index,
                        "episode_id": key[1],
                        "initial_state_sha256": initial_hash[index],
                        "learner_side": side,
                        "opponent": args.opponent,
                        "evaluation_seed": args.seed,
                        "checkpoint_sha256": checkpoint_hash,
                        "environment_config_sha256": environment_config_hash,
                        "outcome": int(outcome > 0) - int(outcome < 0),
                        "win": bool(outcome > 0),
                        "loss": bool(outcome < 0),
                        "draw": bool(outcome == 0),
                        "timeout": bool(info["timeout"][index]),
                        "crash": bool(info["crashes"][index, side]),
                        "episode_length": end_step,
                        "reward": float(episode_reward[index]),
                        "accepted_launches": int(episode_launches[index]),
                        "hits": int(episode_hits[index]),
                        "fire_requests": int(episode_requests[index]),
                        "rejected_fire_requests": int(episode_rejected[index]),
                        "legal_fire_opportunities": int(episode_legal[index]),
                        "learned_expert_steps": int(episode_learned_steps[index]),
                        "intercept_expert_steps": int(episode_intercept_steps[index]),
                        "forced_intervention_steps": int(episode_forced_steps[index]),
                        "expert_switches": int(episode_switches[index]),
                        "option_boundaries": int(
                            episode_option_boundaries[index]),
                        "same_expert_reselections": int(
                            episode_same_expert_reselections[index]),
                        "routing_gate_allowed_steps": int(
                            episode_routing_gate_allowed[index]),
                        "first_routing_gate_allowed_step": (
                            None if int(episode_first_gate_allowed[index]) < 0
                            else int(episode_first_gate_allowed[index])),
                        "first_gate_latch_step": (
                            None if int(episode_first_gate_latch[index]) < 0
                            else int(episode_first_gate_latch[index])),
                        "first_intercept_step": (
                            None if int(episode_first_intercept[index]) < 0
                            else int(episode_first_intercept[index])),
                    })
                initial_hash[index] = _state_hash(env.aircraft_state[index])
                initial_episode_id[index] = env.episode_id[index]
                segment_start[index] = 0
                previous_expert[index] = 0
                segment_forced[index] = False
            for tensor in (
                    episode_reward, episode_launches, episode_hits,
                    episode_requests, episode_rejected, episode_legal,
                    episode_learned_steps, episode_intercept_steps,
                    episode_forced_steps, episode_switches,
                    episode_option_boundaries,
                    episode_same_expert_reselections,
                    episode_routing_gate_allowed):
                tensor.masked_fill_(done, 0)
            for tensor in (
                    episode_first_gate_allowed, episode_first_gate_latch,
                    episode_first_intercept):
                tensor.masked_fill_(done, -1)

    segments = [record for record in raw_segments
                if (record["env_index"], record["episode_id"])
                in accepted_episode_keys]
    completed = len(episode_records)
    wins = sum(record["win"] for record in episode_records)
    losses = sum(record["loss"] for record in episode_records)
    draws = sum(record["draw"] for record in episode_records)
    summary = {
        "suite": "MRAT-RHPPO-fixed-opponent",
        "protocol_version": PROTOCOL_VERSION,
        "scientific_result": False,
        "routing_mode": args.routing_mode,
        "gate_sprt": ({
            "evasive_rate": args.gate_sprt_evasive_rate,
            "easy_rate": args.gate_sprt_easy_rate,
            "alpha": args.gate_sprt_alpha,
            "beta": args.gate_sprt_beta,
        } if args.routing_mode == "gate_sprt_latched" else None),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "reachability_checkpoint_sha256": reachability_hash,
        "router_score_checkpoint_sha256": routing_score_hash,
        "router_intercept_margin": float(
            config.get("router_intercept_margin", 0.0)),
        "environment_config": environment_config,
        "environment_config_sha256": environment_config_hash,
        "opponent": args.opponent,
        "evaluation_seed": args.seed,
        "episodes": completed,
        "learner_red_episodes": learner_red,
        "learner_blue_episodes": learner_blue,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / completed,
        "loss_rate": losses / completed,
        "draw_rate": draws / completed,
        "timeout_rate": sum(record["timeout"] for record in episode_records) / completed,
        "crash_rate": sum(record["crash"] for record in episode_records) / completed,
        "mean_episode_reward": sum(record["reward"] for record in episode_records) / completed,
        "accepted_launches": sum(record["accepted_launches"] for record in episode_records),
        "hits": sum(record["hits"] for record in episode_records),
        "fire_requests": sum(record["fire_requests"] for record in episode_records),
        "rejected_fire_requests": sum(
            record["rejected_fire_requests"] for record in episode_records),
        "legal_fire_opportunities": sum(
            record["legal_fire_opportunities"] for record in episode_records),
        "learned_expert_steps": sum(
            record["learned_expert_steps"] for record in episode_records),
        "intercept_expert_steps": sum(
            record["intercept_expert_steps"] for record in episode_records),
        "forced_intervention_steps": sum(
            record["forced_intervention_steps"] for record in episode_records),
        "expert_switches": sum(record["expert_switches"] for record in episode_records),
        "option_boundaries": sum(
            record["option_boundaries"] for record in episode_records),
        "same_expert_reselections": sum(
            record["same_expert_reselections"] for record in episode_records),
        "option_segments": len(segments),
        "mean_option_dwell": sum(record["duration"] for record in segments) /
            max(len(segments), 1),
        "reachability_lcb_learned_mean": lcb_learned_sum / decisions,
        "reachability_lcb_intercept_mean": lcb_intercept_sum / decisions,
        "reachability_regret_mean": regret_sum / decisions,
        "force_condition_fraction": force_condition_steps / decisions,
        "routing_score_lcb_learned_mean": (
            routing_lcb_learned_sum / decisions),
        "routing_score_lcb_intercept_mean": (
            routing_lcb_intercept_sum / decisions),
        "routing_gate_allowed_fraction": (
            routing_gate_allowed_steps / decisions),
        "environment_decisions_executed": decisions,
        "episode_records": str(
            args.episode_records or args.output.with_suffix(".episodes.jsonl")),
        "segment_records": str(
            args.segment_records or args.output.with_suffix(".segments.jsonl")),
    }
    summary["hit_per_launch"] = summary["hits"] / max(
        summary["accepted_launches"], 1)
    summary["legal_request_rate"] = (
        summary["fire_requests"] - summary["rejected_fire_requests"]
    ) / max(summary["fire_requests"], 1)
    episode_path = args.episode_records or args.output.with_suffix(
        ".episodes.jsonl")
    segment_path = args.segment_records or args.output.with_suffix(
        ".segments.jsonl")
    _write_jsonl(episode_path, episode_records)
    _write_jsonl(segment_path, segments)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
