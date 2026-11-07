#!/usr/bin/env python3
"""Collect paired V9/LOS short-horizon reachability targets on Torch WVR."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from benchmarks.drvoi_v12 import (
    EXECUTION_FEATURE_DIM,
    INTERCEPT_EXPERT,
    LEARNED_EXPERT,
    MRATArbitrator,
    MRATRouter,
    OPPONENT_HYPOTHESES,
    PersistentReachabilityShield,
    ROUTER_FEATURE_DIM,
    build_execution_features,
    build_router_features,
    capture_execution_snapshot,
    conservative_router_logits,
    load_frozen_v9_execution,
    load_frozen_reachability_predictor,
    repeat_execution_snapshot,
    restore_execution_snapshot,
    rollout_reachability_branch,
    write_json_atomic,
)
from envs.torch6dof import TorchWVRConfig, TorchWVRVecEnv


PROTOCOL_VERSION = "MRAT-RHPPO-V12-REACH-DATA-4"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path(
            "runs/drvoi_rhppo_v9/coach-online/"
            "development-1m-seed85/latest.pt"))
    parser.add_argument(
        "--response-checkpoint", type=Path,
        default=Path(
            "runs/drvoi_rhppo_v7_1/w3-confirmation/confirmation.pt"))
    parser.add_argument(
        "--option-q-checkpoint", type=Path,
        default=Path(
            "runs/drvoi_rhppo_v8/macro-option/development/"
            "cross-fitted-option-q.pt"))
    parser.add_argument(
        "--behavior-checkpoint", type=Path,
        help=("Optional MRAT checkpoint used to collect on-policy states. "
              "Without it the collector keeps the original frozen-V9 "
              "behavior distribution."))
    parser.add_argument("--reachability-checkpoint", type=Path)
    parser.add_argument("--router-score-checkpoint", type=Path)
    parser.add_argument("--router-intercept-margin", type=float)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-states", type=int, default=50_000)
    parser.add_argument("--num-envs", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--teacher-gamma", type=float, default=0.98)
    parser.add_argument("--boundary-stride", type=int, default=6)
    parser.add_argument("--chunk-collections", type=int, default=8)
    parser.add_argument("--samples-per-member", type=int, default=4)
    parser.add_argument("--seed", type=int, default=85)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--compile-dynamics", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _main_opponent_hypotheses(num_envs: int, device: torch.device) -> torch.Tensor:
    env_index = torch.arange(num_envs, device=device)
    category = env_index.remainder(3)
    evasive_direction = 2 + torch.div(
        env_index, 3, rounding_mode="floor").remainder(2)
    return torch.where(category == 2, evasive_direction, category)


def _save_chunk(
        output_dir: Path,
        chunk_index: int,
        records: dict[str, list[torch.Tensor]],
) -> tuple[Path, int]:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "feature_dim": EXECUTION_FEATURE_DIM,
        "opponent_hypotheses": OPPONENT_HYPOTHESES,
        **{key: torch.cat(values, 0) for key, values in records.items()},
    }
    rows = int(payload["features"].shape[0])
    path = output_dir / "chunks" / f"reachability-{chunk_index:05d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path, rows


def _save_resume(
        path: Path,
        snapshot,
        collection_index: int,
        collected_states: int,
        online_decisions: int,
        model_decisions: int,
        chunk_index: int,
        behavior_state: dict | None = None,
) -> None:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "collection_index": collection_index,
        "collected_states": collected_states,
        "online_environment_decisions": online_decisions,
        "teacher_model_decisions": model_decisions,
        "chunk_index": chunk_index,
        "snapshot": snapshot,
        "behavior_state": behavior_state,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    completed_path = args.output_dir / "COMPLETED.json"
    if args.resume and completed_path.is_file():
        print(completed_path.read_text(encoding="utf-8"))
        return
    if args.target_states <= 0 or args.num_envs <= 0:
        raise ValueError("target-states and num-envs must be positive")
    if args.horizon <= 0 or args.boundary_stride <= 0:
        raise ValueError("horizon and boundary-stride must be positive")
    if not 0.0 < args.teacher_gamma <= 1.0:
        raise ValueError("teacher-gamma must lie in (0,1]")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_frozen_v9_execution(
        args.checkpoint, args.response_checkpoint, args.option_q_checkpoint,
        device=device, samples_per_member=args.samples_per_member)
    executor = bundle.executor
    router = shield = arbitrator = None
    reachability = routing_predictor = None
    routing_payload = None
    behavior_margin = 0.0
    if args.behavior_checkpoint is not None:
        if (args.reachability_checkpoint is None
                or args.router_score_checkpoint is None):
            raise ValueError(
                "on-policy collection requires reachability and router-score "
                "checkpoints")
        behavior_payload = torch.load(
            args.behavior_checkpoint, map_location=device,
            weights_only=False)
        router = MRATRouter(ROUTER_FEATURE_DIM, hidden_size=128).to(device)
        router.load_state_dict(behavior_payload["router"])
        router.eval()
        reachability, _ = load_frozen_reachability_predictor(
            args.reachability_checkpoint, device=device)
        routing_predictor, routing_payload = (
            load_frozen_reachability_predictor(
                args.router_score_checkpoint, device=device))
        behavior_args = behavior_payload.get("args", {})
        shield = PersistentReachabilityShield(
            bad_threshold=float(behavior_args.get("bad_threshold", -0.10)),
            safe_threshold=float(behavior_args.get("safe_threshold", 0.10)),
            intercept_advantage=float(
                behavior_args.get("intercept_advantage", 0.05)),
            force_steps=int(behavior_args.get("force_steps", 12)),
            release_steps=int(behavior_args.get("release_steps", 6)))
        arbitrator = MRATArbitrator(
            minimum_dwell=int(behavior_args.get("minimum_dwell", 6)),
            maximum_dwell=int(behavior_args.get("maximum_dwell", 150)))
        behavior_margin = float(
            behavior_args.get("router_intercept_margin", 0.0)
            if args.router_intercept_margin is None
            else args.router_intercept_margin)
    env = TorchWVRVecEnv(TorchWVRConfig(
        num_envs=args.num_envs,
        reward_mode="competitive",
        attack_quality_weight=0.0,
        observation_version="v4",
        use_dynamic_launch_envelope=True,
    ), device=device, seed=args.seed,
        compile_dynamics=args.compile_dynamics)
    branch_count = 2 * len(OPPONENT_HYPOTHESES)
    branch_env = TorchWVRVecEnv(TorchWVRConfig(
        num_envs=args.num_envs * branch_count,
        reward_mode="competitive",
        attack_quality_weight=0.0,
        observation_version="v4",
        use_dynamic_launch_envelope=True,
    ), device=device, seed=args.seed + 1,
        compile_dynamics=args.compile_dynamics)
    learner_side = torch.arange(args.num_envs, device=device).remainder(2)
    state = executor.initial_state(
        args.num_envs, device, learner_side=learner_side,
        opponent_hypothesis=_main_opponent_hypotheses(
            args.num_envs, device),
        training_opponent_kind=torch.arange(
            args.num_envs, device=device).remainder(3))

    resume_path = args.output_dir / "collector-latest.pt"
    collection_index = collected_states = online_decisions = 0
    model_decisions = chunk_index = 0
    if args.resume and resume_path.is_file():
        resume_payload = torch.load(
            resume_path, map_location=device, weights_only=False)
        if resume_payload.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("resume checkpoint protocol does not match")
        state = restore_execution_snapshot(
            env, resume_payload["snapshot"])
        state.setdefault(
            "training_opponent_kind",
            torch.arange(args.num_envs, device=device).remainder(3))
        collection_index = int(resume_payload["collection_index"])
        collected_states = int(resume_payload["collected_states"])
        online_decisions = int(
            resume_payload["online_environment_decisions"])
        model_decisions = int(resume_payload["teacher_model_decisions"])
        chunk_index = int(resume_payload["chunk_index"])
        if args.behavior_checkpoint is not None:
            saved_behavior = resume_payload.get("behavior_state")
            if not saved_behavior:
                raise ValueError(
                    "on-policy resume checkpoint lacks behavior state")
            shield.load_state_dict(saved_behavior["shield"])
            arbitrator.load_state_dict(saved_behavior["arbitrator"])

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_result": False,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "horizon": args.horizon,
        "teacher_gamma": args.teacher_gamma,
        "boundary_stride": args.boundary_stride,
        "target_states": args.target_states,
        "feature_dim": EXECUTION_FEATURE_DIM,
        "opponent_hypotheses": OPPONENT_HYPOTHESES,
        "checkpoint_sha256": _hash_file(args.checkpoint),
        "response_checkpoint_sha256": _hash_file(args.response_checkpoint),
        "option_q_checkpoint_sha256": _hash_file(args.option_q_checkpoint),
        "behavior_policy": (
            "frozen_v9" if args.behavior_checkpoint is None
            else "mrat_router"),
        "behavior_checkpoint_sha256": (
            None if args.behavior_checkpoint is None
            else _hash_file(args.behavior_checkpoint)),
        "reachability_checkpoint_sha256": (
            None if args.reachability_checkpoint is None
            else _hash_file(args.reachability_checkpoint)),
        "router_score_checkpoint_sha256": (
            None if args.router_score_checkpoint is None
            else _hash_file(args.router_score_checkpoint)),
        "router_intercept_margin": behavior_margin,
        "split_requirement": "group by (env_index, episode_id)",
        "budget_semantics": {
            "online_environment_decisions": "main trajectory only",
            "teacher_model_decisions": "all active counterfactual rows",
        },
    }
    write_json_atomic(args.output_dir / "dataset-manifest.json", manifest)

    records = {key: [] for key in (
        "features", "target_score", "target_legal", "target_first_legal",
        "target_crash", "target_terminal_step", "env_index", "episode_id",
        "target_legal_steps", "target_fire_requests",
        "target_accepted_launches", "target_rejected_fire_requests",
        "target_hits", "target_discounted_return",
        "target_terminal_outcome", "target_timeout", "episode_step",
        "learner_side", "collection_index")}
    started = time.perf_counter()
    active = torch.ones(args.num_envs, dtype=torch.bool, device=device)
    checkpoint_due = False
    if not resume_path.is_file():
        behavior_state = None
        if args.behavior_checkpoint is not None:
            behavior_state = {
                "shield": shield.state_dict(),
                "arbitrator": arbitrator.state_dict(),
            }
        _save_resume(
            resume_path, capture_execution_snapshot(env, state),
            collection_index, collected_states, online_decisions,
            model_decisions, chunk_index, behavior_state)

    while collected_states < args.target_states:
        if collection_index % args.boundary_stride == 0:
            entry = capture_execution_snapshot(env, state)
            observation = env.observations()
            env_index = torch.arange(args.num_envs, device=device)
            learner_observation = observation[env_index, learner_side]
            features = build_execution_features(learner_observation, state)
            tiled = repeat_execution_snapshot(entry, branch_count)
            branch_state = restore_execution_snapshot(branch_env, tiled)
            branch_ids = torch.arange(branch_count, device=device)
            expert = torch.div(
                branch_ids, len(OPPONENT_HYPOTHESES),
                rounding_mode="floor").repeat_interleave(args.num_envs)
            hypothesis = branch_ids.remainder(
                len(OPPONENT_HYPOTHESES)).repeat_interleave(args.num_envs)
            branch_state["selected_expert"] = expert
            branch_state["expert_age"].zero_()
            branch_state["opponent_hypothesis"] = hypothesis
            branch_state.pop("training_opponent_kind", None)
            branch_state["response_coupling_base_batch"] = torch.tensor(
                args.num_envs, device=device, dtype=torch.long)
            branch_learner_side = learner_side.repeat(branch_count)
            result, _ = rollout_reachability_branch(
                branch_env, branch_learner_side, branch_state,
                executor.prepare_step,
                policy_post_step=executor.post_step,
                horizon=args.horizon,
                gamma=args.teacher_gamma)
            target_shape = (2, len(OPPONENT_HYPOTHESES), args.num_envs)
            scores = result.score.reshape(target_shape).permute(2, 0, 1)
            legal = result.legal_within_horizon.reshape(
                target_shape).permute(2, 0, 1)
            first_legal = result.first_legal_step.reshape(
                target_shape).permute(2, 0, 1)
            crash = result.crashed.reshape(target_shape).permute(2, 0, 1)
            terminal_step = result.terminal_step.reshape(
                target_shape).permute(2, 0, 1)
            legal_steps = result.legal_opportunity_steps.reshape(
                target_shape).permute(2, 0, 1)
            fire_requests = result.fire_requests.reshape(
                target_shape).permute(2, 0, 1)
            accepted_launches = result.accepted_launches.reshape(
                target_shape).permute(2, 0, 1)
            rejected_requests = result.rejected_fire_requests.reshape(
                target_shape).permute(2, 0, 1)
            hits = result.hits.reshape(target_shape).permute(2, 0, 1)
            discounted_return = result.discounted_return.reshape(
                target_shape).permute(2, 0, 1)
            terminal_outcome = result.terminal_outcome.reshape(
                target_shape).permute(2, 0, 1)
            timeout = result.timeout.reshape(
                target_shape).permute(2, 0, 1)
            model_decisions += result.model_decisions
            state = restore_execution_snapshot(env, entry)
            records["features"].append(features.cpu())
            records["target_score"].append(scores.cpu())
            records["target_legal"].append(legal.cpu())
            records["target_first_legal"].append(first_legal.cpu())
            records["target_crash"].append(crash.cpu())
            records["target_terminal_step"].append(terminal_step.cpu())
            records["target_legal_steps"].append(legal_steps.cpu())
            records["target_fire_requests"].append(fire_requests.cpu())
            records["target_accepted_launches"].append(
                accepted_launches.cpu())
            records["target_rejected_fire_requests"].append(
                rejected_requests.cpu())
            records["target_hits"].append(hits.cpu())
            records["target_discounted_return"].append(
                discounted_return.cpu())
            records["target_terminal_outcome"].append(
                terminal_outcome.cpu())
            records["target_timeout"].append(timeout.cpu())
            records["env_index"].append(env_index.cpu())
            records["episode_id"].append(env.episode_id.cpu().clone())
            records["episode_step"].append(env.episode_step.cpu().clone())
            records["learner_side"].append(learner_side.cpu())
            records["collection_index"].append(torch.full(
                (args.num_envs,), collection_index, dtype=torch.long))
            collected_states += args.num_envs

            if len(records["features"]) >= args.chunk_collections:
                chunk_path, rows = _save_chunk(
                    args.output_dir, chunk_index, records)
                chunk_index += 1
                records = {key: [] for key in records}
                elapsed = time.perf_counter() - started
                progress = {
                    "protocol_version": PROTOCOL_VERSION,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "collected_states": collected_states,
                    "target_states": args.target_states,
                    "progress_fraction": min(
                        collected_states / args.target_states, 1.0),
                    "online_environment_decisions": online_decisions,
                    "teacher_model_decisions": model_decisions,
                    "states_per_second": collected_states / max(elapsed, 1e-9),
                    "eta_seconds": max(
                        args.target_states - collected_states, 0
                    ) / max(collected_states / max(elapsed, 1e-9), 1e-9),
                    "teacher_model_decisions_per_second": (
                        model_decisions / max(elapsed, 1e-9)),
                    "last_chunk": str(chunk_path),
                    "last_chunk_rows": rows,
                }
                write_json_atomic(args.output_dir / "progress.json", progress)
                checkpoint_due = True
                print(json.dumps(progress, ensure_ascii=False), flush=True)

        # Advance either the original frozen-V9 distribution or the exact
        # deterministic MRAT deployment policy.  The latter is a DAgger-style
        # coverage pass: labels still come from paired counterfactual branches,
        # while the main state distribution follows the deployed router.
        if args.behavior_checkpoint is None:
            state["selected_expert"].fill_(LEARNED_EXPERT)
        else:
            with torch.no_grad():
                observation = env.observations()
                learner_observation = observation[
                    torch.arange(args.num_envs, device=device), learner_side]
                execution_features = build_execution_features(
                    learner_observation, state)
                physical_score = reachability.predict_both(
                    execution_features)
                routing_score = routing_predictor.predict_both(
                    execution_features)
                router_features = build_router_features(
                    routing_score["standardized_features"], routing_score,
                    horizon=float(routing_payload["censor_horizon"]))
                router_outputs = router(router_features)
                route_logits, _ = conservative_router_logits(
                    router_outputs["router_logits"],
                    routing_score["lcb"][:, 0],
                    routing_score["lcb"][:, 1], behavior_margin)
                proposed = route_logits.argmax(-1)
                if arbitrator.expert is None:
                    current_expert = torch.zeros_like(proposed)
                    current_dwell = torch.zeros_like(proposed)
                else:
                    current_expert = arbitrator.expert
                    current_dwell = arbitrator.dwell
                normalized_dwell = (
                    current_dwell.float()
                    / arbitrator.maximum_dwell).clamp(0.0, 1.0)
                terminate = (
                    router.termination_distribution(
                        router_features, current_expert,
                        normalized_dwell).probs >= 0.5)
                reset = state["reset"].bool()
                shield_step = shield.step(
                    physical_score["lcb"][:, 0],
                    physical_score["lcb"][:, 1], reset)
                arbitration = arbitrator.step(
                    proposed, terminate,
                    shield_step.forced_intercept, reset)
                state["selected_expert"] = arbitration.expert.clone()
                state["expert_age"] = (
                    arbitration.dwell - 1).clamp_min(0)
        controls, fire, state = executor.prepare_step(env, state, active)
        _, _, done, info = env.step(controls, fire)
        state = executor.post_step(env, state, done, info)
        online_decisions += args.num_envs
        collection_index += 1
        if checkpoint_due:
            behavior_state = None
            if args.behavior_checkpoint is not None:
                behavior_state = {
                    "shield": shield.state_dict(),
                    "arbitrator": arbitrator.state_dict(),
                }
            _save_resume(
                resume_path, capture_execution_snapshot(env, state),
                collection_index, collected_states, online_decisions,
                model_decisions, chunk_index, behavior_state)
            checkpoint_due = False

    if records["features"]:
        _save_chunk(args.output_dir, chunk_index, records)
        chunk_index += 1
    final_snapshot = capture_execution_snapshot(env, state)
    behavior_state = None
    if args.behavior_checkpoint is not None:
        behavior_state = {
            "shield": shield.state_dict(),
            "arbitrator": arbitrator.state_dict(),
        }
    _save_resume(
        resume_path, final_snapshot, collection_index, collected_states,
        online_decisions, model_decisions, chunk_index, behavior_state)
    elapsed = time.perf_counter() - started
    completed = {
        "protocol_version": PROTOCOL_VERSION,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_result": False,
        "collected_states": collected_states,
        "online_environment_decisions": online_decisions,
        "teacher_model_decisions": model_decisions,
        "wall_time_seconds": elapsed,
        "states_per_second": collected_states / max(elapsed, 1e-9),
        "teacher_model_decisions_per_second": model_decisions / max(elapsed, 1e-9),
        "chunks": chunk_index,
    }
    write_json_atomic(args.output_dir / "COMPLETED.json", completed)
    print(json.dumps(completed, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
