#!/usr/bin/env python3
"""Train the V12 MRAT router/termination policy over frozen V9 and LOS experts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.distributions import Categorical
from torch.nn import functional as F

from benchmarks.drvoi.checkpointing import atomic_torch_save
from benchmarks.drvoi_v12 import (
    FlatOptionBatch,
    MRATArbitrator,
    MRATRouter,
    PersistentReachabilityShield,
    ROUTER_FEATURE_DIM,
    append_update_metrics,
    build_execution_features,
    build_router_features,
    capture_execution_snapshot,
    clipped_policy_loss,
    conservative_router_logits,
    compute_flat_smdp_gae,
    load_frozen_reachability_predictor,
    load_frozen_v9_execution,
    option_termination_loss,
    regularized_step_reward,
    restore_execution_snapshot,
    write_json_atomic,
)
from envs.torch6dof import TorchWVRConfig, TorchWVRVecEnv


ALGORITHM = "mrat_rhppo"
PROTOCOL_VERSION = "MRAT-RHPPO-V12-WVR-1"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v9-checkpoint", type=Path, required=True)
    parser.add_argument("--response-checkpoint", type=Path, required=True)
    parser.add_argument("--option-q-checkpoint", type=Path, required=True)
    parser.add_argument("--reachability-checkpoint", type=Path, required=True)
    parser.add_argument("--router-score-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=96)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=1.0e-3)
    parser.add_argument("--target-kl", type=float, default=0.015)
    parser.add_argument("--minimum-dwell", type=int, default=6)
    parser.add_argument("--maximum-dwell", type=int, default=150)
    parser.add_argument("--force-steps", type=int, default=12)
    parser.add_argument("--release-steps", type=int, default=6)
    parser.add_argument("--bad-threshold", type=float, default=-0.10)
    parser.add_argument("--safe-threshold", type=float, default=0.10)
    parser.add_argument("--intercept-advantage", type=float, default=0.05)
    parser.add_argument("--router-intercept-margin", type=float, default=0.0)
    parser.add_argument("--counterfactual-coef", type=float, default=0.25)
    parser.add_argument("--counterfactual-anneal-fraction", type=float,
                        default=0.10)
    parser.add_argument("--bad-cost-limit", type=float, default=0.05)
    parser.add_argument("--crash-cost-limit", type=float, default=0.01)
    parser.add_argument("--lagrangian-learning-rate", type=float, default=0.01)
    parser.add_argument("--intervention-cost", type=float, default=0.0)
    parser.add_argument("--switch-cost", type=float, default=0.0)
    parser.add_argument(
        "--lineage-online-decisions", type=int, default=0,
        help=("Additional online decisions from prerequisite development "
              "stages not already represented by the loaded datasets."))
    parser.add_argument(
        "--lineage-model-decisions", type=int, default=0,
        help=("Additional model-branch decisions from prerequisite stages "
              "not already represented by the loaded datasets."))
    parser.add_argument("--seed", type=int, default=85)
    parser.add_argument(
        "--stage-label", default="development",
        help="Human-readable experiment stage recorded in checkpoints and logs.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--compile-dynamics", action=argparse.BooleanOptionalAction,
                        default=True)
    return parser.parse_args()


def _counterfactual_coefficient(args, update: int) -> float:
    if args.counterfactual_anneal_fraction <= 0.0:
        return 0.0
    progress = (update - 1) / max(args.updates, 1)
    return args.counterfactual_coef * max(
        1.0 - progress / args.counterfactual_anneal_fraction, 0.0)


def _cat(values: list[torch.Tensor]) -> torch.Tensor:
    if not values:
        raise ValueError("cannot stack an empty segment field")
    return torch.cat(values, 0)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("MRAT-RHPPO training requires CUDA")
    if args.safe_threshold <= args.bad_threshold:
        raise ValueError("safe-threshold must exceed bad-threshold")
    if (args.lineage_online_decisions < 0
            or args.lineage_model_decisions < 0):
        raise ValueError("lineage decision counts must be non-negative")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed_path = args.output_dir / "COMPLETED.json"
    if args.resume and completed_path.is_file():
        print(completed_path.read_text(encoding="utf-8"))
        return

    v9_bundle = load_frozen_v9_execution(
        args.v9_checkpoint, args.response_checkpoint,
        args.option_q_checkpoint, device=device)
    executor = v9_bundle.executor
    reachability, reach_payload = load_frozen_reachability_predictor(
        args.reachability_checkpoint, device=device)
    routing_predictor = reachability
    routing_payload = reach_payload
    if args.router_score_checkpoint is not None:
        routing_predictor, routing_payload = (
            load_frozen_reachability_predictor(
                args.router_score_checkpoint, device=device))
    router = MRATRouter(ROUTER_FEATURE_DIM, hidden_size=128).to(device)
    optimizer = torch.optim.AdamW(
        router.parameters(), lr=args.learning_rate, eps=1.0e-5,
        weight_decay=1.0e-5)
    env = TorchWVRVecEnv(TorchWVRConfig(
        num_envs=args.num_envs,
        reward_mode="competitive",
        attack_quality_weight=0.0,
        observation_version="v4",
        use_dynamic_launch_envelope=True,
    ), device=device, seed=args.seed,
        compile_dynamics=args.compile_dynamics)
    env_index = torch.arange(args.num_envs, device=device)
    learner_side = env_index.remainder(2)
    execution_state = executor.initial_state(
        args.num_envs, device, learner_side=learner_side,
        training_opponent_kind=env_index.remainder(3))
    shield = PersistentReachabilityShield(
        bad_threshold=args.bad_threshold,
        safe_threshold=args.safe_threshold,
        intercept_advantage=args.intercept_advantage,
        force_steps=args.force_steps,
        release_steps=args.release_steps)
    arbitrator = MRATArbitrator(
        minimum_dwell=args.minimum_dwell,
        maximum_dwell=args.maximum_dwell)
    latest_path = args.output_dir / "latest.pt"
    start_update = 1
    continuation_decisions = 0
    prior_v9_decisions = int(v9_bundle.checkpoint_payload[
        "training_environment_decisions"])
    dataset_completed = reach_payload.get("dataset_completed", {})
    prior_a1_online_decisions = int(dataset_completed.get(
        "online_environment_decisions", 0))
    model_branch_decisions = int(dataset_completed.get(
        "teacher_model_decisions", 0))
    routing_dataset_completed = (
        routing_payload.get("dataset_completed", {})
        if args.router_score_checkpoint is not None else {})
    prior_return_online_decisions = int(routing_dataset_completed.get(
        "online_environment_decisions", 0))
    return_model_branch_decisions = int(routing_dataset_completed.get(
        "teacher_model_decisions", 0))
    model_branch_decisions += return_model_branch_decisions
    model_branch_decisions += args.lineage_model_decisions
    prior_online_decisions = (
        prior_v9_decisions + prior_a1_online_decisions
        + prior_return_online_decisions + args.lineage_online_decisions)
    bad_multiplier = 0.0
    crash_multiplier = 0.0
    history = []
    if args.resume and latest_path.is_file():
        payload = torch.load(latest_path, map_location=device, weights_only=False)
        if payload.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("resume checkpoint protocol mismatch")
        router.load_state_dict(payload["router"])
        optimizer.load_state_dict(payload["optimizer"])
        execution_state = restore_execution_snapshot(env, payload["snapshot"])
        execution_state.setdefault(
            "training_opponent_kind", env_index.remainder(3))
        shield.load_state_dict(payload["shield"])
        arbitrator.load_state_dict(payload["arbitrator"])
        start_update = int(payload["update"]) + 1
        continuation_decisions = int(payload["continuation_decisions"])
        model_branch_decisions = int(payload["model_branch_decisions"])
        bad_multiplier = float(payload["bad_multiplier"])
        crash_multiplier = float(payload["crash_multiplier"])
        history = payload.get("history", [])

    metrics_path = args.output_dir / "metrics.jsonl"
    if start_update == 1:
        metrics_path.write_text("", encoding="utf-8")
    elif metrics_path.is_file():
        # Make append logging idempotent if interruption happened after the
        # JSONL write but before the matching atomic checkpoint replacement.
        retained = []
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if int(record.get("update", 0)) < start_update:
                retained.append(line)
        metrics_path.write_text(
            "\n".join(retained) + ("\n" if retained else ""),
            encoding="utf-8")
    started = time.perf_counter()

    for update in range(start_update, args.updates + 1):
        rollout_started = time.perf_counter()
        reachability_seconds = 0.0
        reachability_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        segment_fields = {key: [] for key in (
            "features", "expert", "old_log_prob", "value",
            "segment_return", "next_value", "duration", "done",
            "policy_mask", "counterfactual_mask", "env_id", "teacher_best",
            "routing_gate_allowed")}
        termination_features = []
        termination_expert = []
        termination_dwell = []
        termination_eligible = []
        termination_gate_allowed = []

        active = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
        seg_features = torch.zeros(
            args.num_envs, ROUTER_FEATURE_DIM, device=device)
        seg_expert = torch.zeros(args.num_envs, dtype=torch.long, device=device)
        seg_log_prob = torch.zeros(args.num_envs, device=device)
        seg_value = torch.zeros(args.num_envs, device=device)
        seg_return = torch.zeros(args.num_envs, device=device)
        seg_duration = torch.zeros(
            args.num_envs, dtype=torch.long, device=device)
        seg_policy_mask = torch.zeros(
            args.num_envs, dtype=torch.bool, device=device)
        seg_counterfactual_mask = torch.zeros(
            args.num_envs, dtype=torch.bool, device=device)
        seg_teacher_best = torch.zeros(
            args.num_envs, dtype=torch.long, device=device)
        seg_gate_allowed = torch.ones(
            args.num_envs, dtype=torch.bool, device=device)

        counters = {key: 0.0 for key in (
            "completed", "wins", "losses", "draws", "timeouts", "crashes",
            "reward", "fire_requests", "accepted", "rejected", "legal",
            "hits", "learned_steps", "intercept_steps", "router_steps",
            "forced_steps", "switches", "segments", "dwell_sum",
            "learned_lcb", "intercept_lcb", "regret", "force_condition",
            "bad_cost", "crash_cost", "intervention_regularization",
            "switch_regularization", "routing_lcb_learned",
            "routing_lcb_intercept", "routing_gate_allowed",
            "termination_eligible", "termination_requests")}

        def close_segments(mask: torch.Tensor, next_value: torch.Tensor,
                           terminal: torch.Tensor,
                           count_completed: bool = True) -> None:
            nonlocal active
            if not bool(mask.any()):
                return
            ids = torch.nonzero(mask, as_tuple=False).flatten()
            segment_fields["features"].append(
                seg_features[ids].detach().clone())
            segment_fields["expert"].append(seg_expert[ids].detach().clone())
            segment_fields["old_log_prob"].append(
                seg_log_prob[ids].detach().clone())
            segment_fields["value"].append(seg_value[ids].detach().clone())
            segment_fields["segment_return"].append(
                seg_return[ids].detach().clone())
            segment_fields["next_value"].append(
                next_value[ids].detach().clone())
            segment_fields["duration"].append(
                seg_duration[ids].detach().clone())
            segment_fields["done"].append(terminal[ids].detach().clone())
            segment_fields["policy_mask"].append(
                seg_policy_mask[ids].detach().clone())
            segment_fields["counterfactual_mask"].append(
                seg_counterfactual_mask[ids].clone())
            segment_fields["env_id"].append(ids.clone())
            segment_fields["teacher_best"].append(
                seg_teacher_best[ids].clone())
            segment_fields["routing_gate_allowed"].append(
                seg_gate_allowed[ids].clone())
            if count_completed:
                counters["segments"] += int(len(ids))
                counters["dwell_sum"] += float(seg_duration[ids].sum())
            active[ids] = False

        for _ in range(args.rollout_steps):
            observation = env.observations()
            learner_observation = observation[env_index, learner_side]
            execution_features = build_execution_features(
                learner_observation, execution_state)
            reach_start_event = torch.cuda.Event(enable_timing=True)
            reach_end_event = torch.cuda.Event(enable_timing=True)
            reach_start_event.record()
            reach = reachability.predict_both(execution_features)
            routing_score = (
                routing_predictor.predict_both(execution_features)
                if args.router_score_checkpoint is not None else reach)
            router_features = build_router_features(
                routing_score["standardized_features"], routing_score,
                horizon=float(routing_payload["censor_horizon"]))
            reach_end_event.record()
            reachability_events.append((reach_start_event, reach_end_event))

            outputs = router(router_features)
            if args.router_score_checkpoint is not None:
                route_logits, gate_allowed = conservative_router_logits(
                    outputs["router_logits"], routing_score["lcb"][:, 0],
                    routing_score["lcb"][:, 1],
                    args.router_intercept_margin)
            else:
                route_logits = outputs["router_logits"]
                gate_allowed = torch.ones(
                    args.num_envs, dtype=torch.bool, device=device)
            route_distribution = Categorical(logits=route_logits)
            proposed = route_distribution.sample()
            route_log_prob = route_distribution.log_prob(proposed)
            if arbitrator.expert is None:
                current_expert = torch.zeros_like(proposed)
                current_dwell = torch.zeros_like(proposed)
            else:
                current_expert = arbitrator.expert
                current_dwell = arbitrator.dwell
            normalized_dwell = (
                current_dwell.float() / args.maximum_dwell).clamp(0.0, 1.0)
            termination_distribution = router.termination_distribution(
                router_features, current_expert, normalized_dwell)
            termination_request = termination_distribution.sample().bool()
            reset = execution_state["reset"].bool()
            shield_step = shield.step(
                reach["lcb"][:, 0], reach["lcb"][:, 1], reset)
            arbitration = arbitrator.step(
                proposed, termination_request,
                shield_step.forced_intercept, reset)
            selected_value = outputs["q_values"].gather(
                1, arbitration.expert[:, None]).squeeze(1)
            state_value = (
                route_distribution.probs * outputs["q_values"]).sum(-1)

            boundary = active & (arbitration.decision | arbitration.switched)
            close_segments(
                boundary, state_value,
                torch.zeros_like(boundary))
            start = ~active
            if bool(start.any()):
                seg_features[start] = router_features[start].detach()
                seg_expert[start] = arbitration.expert[start]
                seg_log_prob[start] = route_distribution.log_prob(
                    arbitration.expert)[start].detach()
                seg_value[start] = selected_value[start].detach()
                seg_return[start] = 0.0
                seg_duration[start] = 0
                seg_policy_mask[start] = (
                    arbitration.decision & ~arbitration.forced
                    & gate_allowed)[start]
                seg_counterfactual_mask[start] = (
                    (arbitration.decision | arbitration.switched)
                    & gate_allowed)[start]
                seg_teacher_best[start] = torch.where(
                    gate_allowed, routing_score["lcb"].argmax(-1),
                    torch.zeros_like(arbitration.expert))[start]
                seg_gate_allowed[start] = gate_allowed[start]
                active[start] = True

            eligible_termination = (
                (current_dwell >= args.minimum_dwell)
                & ~shield_step.forced_intercept & ~reset)
            if bool(eligible_termination.any()):
                termination_features.append(router_features.clone())
                termination_expert.append(current_expert.clone())
                termination_dwell.append(normalized_dwell.clone())
                termination_eligible.append(eligible_termination.clone())
                termination_gate_allowed.append(gate_allowed.clone())
                counters["termination_eligible"] += int(
                    eligible_termination.sum())
                counters["termination_requests"] += int((
                    eligible_termination & termination_request).sum())

            execution_state["selected_expert"] = arbitration.expert.clone()
            execution_state["expert_age"] = (
                arbitration.dwell - 1).clamp_min(0)
            all_active = torch.ones_like(active)
            controls, fire, execution_state = executor.prepare_step(
                env, execution_state, all_active)
            _, reward, done, info = env.step(controls, fire)
            learner_reward = reward[env_index, learner_side]
            selected_lcb = reach["lcb"].gather(
                1, arbitration.expert[:, None]).squeeze(1)
            reachability_regret = reach["lcb"].amax(-1) - selected_lcb
            bad_cost = (reachability_regret > 0.10).float()
            crash_cost = info["crashes"][env_index, learner_side].float()
            regularized_reward, intervention_cost, switch_cost = (
                regularized_step_reward(
                    learner_reward, arbitration.expert,
                    arbitration.switched & ~reset,
                    intervention_weight=args.intervention_cost,
                    switch_weight=args.switch_cost))
            constrained_reward = (
                regularized_reward - bad_multiplier * bad_cost
                - crash_multiplier * crash_cost)
            seg_return += torch.pow(
                torch.as_tensor(args.gamma, device=device),
                seg_duration.float()) * constrained_reward
            seg_duration += 1
            close_segments(done & active, torch.zeros_like(state_value), done)
            execution_state = executor.post_step(
                env, execution_state, done, info)

            learner_outcome = torch.where(
                learner_side == 0, info["outcome_red"], -info["outcome_red"])
            counters["completed"] += int(done.sum())
            counters["wins"] += int((done & (learner_outcome > 0)).sum())
            counters["losses"] += int((done & (learner_outcome < 0)).sum())
            counters["draws"] += int((done & (learner_outcome == 0)).sum())
            counters["timeouts"] += int((done & info["timeout"]).sum())
            counters["crashes"] += int(
                (done & info["crashes"][env_index, learner_side]).sum())
            counters["reward"] += float(learner_reward.sum())
            counters["fire_requests"] += int(
                info["fire_requests"][env_index, learner_side].sum())
            counters["accepted"] += int(
                info["accepted_launches"][env_index, learner_side].sum())
            counters["rejected"] += int(
                info["rejected_fire_requests"][env_index, learner_side].sum())
            counters["legal"] += int(
                info["legal_fire_opportunities"][env_index, learner_side].sum())
            counters["hits"] += int(info["hits"][env_index, learner_side].sum())
            counters["learned_steps"] += int((arbitration.expert == 0).sum())
            counters["intercept_steps"] += int((arbitration.expert == 1).sum())
            counters["router_steps"] += int(
                ((arbitration.expert == 1) & ~arbitration.forced).sum())
            counters["forced_steps"] += int(arbitration.forced.sum())
            counters["switches"] += int((arbitration.switched & ~reset).sum())
            counters["learned_lcb"] += float(reach["lcb"][:, 0].sum())
            counters["intercept_lcb"] += float(reach["lcb"][:, 1].sum())
            counters["regret"] += float(reachability_regret.sum())
            counters["force_condition"] += int(shield_step.force_condition.sum())
            counters["bad_cost"] += float(bad_cost.sum())
            counters["crash_cost"] += float(crash_cost.sum())
            counters["intervention_regularization"] += float(
                intervention_cost.sum())
            counters["switch_regularization"] += float(switch_cost.sum())
            counters["routing_lcb_learned"] += float(
                routing_score["lcb"][:, 0].sum())
            counters["routing_lcb_intercept"] += float(
                routing_score["lcb"][:, 1].sum())
            counters["routing_gate_allowed"] += int(gate_allowed.sum())

        # Close rollout-truncated segments with a state-value bootstrap. They
        # continue physically next rollout but are excluded from router PPO
        # unless they began at a genuine option decision.
        observation = env.observations()
        learner_observation = observation[env_index, learner_side]
        execution_features = build_execution_features(
            learner_observation, execution_state)
        reach = reachability.predict_both(execution_features)
        routing_score = (
            routing_predictor.predict_both(execution_features)
            if args.router_score_checkpoint is not None else reach)
        final_features = build_router_features(
            routing_score["standardized_features"], routing_score,
            horizon=float(routing_payload["censor_horizon"]))
        final_outputs = router(final_features)
        if args.router_score_checkpoint is not None:
            final_logits, _ = conservative_router_logits(
                final_outputs["router_logits"],
                routing_score["lcb"][:, 0], routing_score["lcb"][:, 1],
                args.router_intercept_margin)
        else:
            final_logits = final_outputs["router_logits"]
        final_value = (
            torch.softmax(final_logits, -1)
            * final_outputs["q_values"]).sum(-1)
        close_segments(
            active.clone(), final_value,
            torch.zeros(args.num_envs, dtype=torch.bool, device=device),
            count_completed=False)
        torch.cuda.synchronize()
        reachability_seconds = sum(
            start.elapsed_time(end) for start, end in reachability_events
        ) / 1000.0
        rollout_seconds = time.perf_counter() - rollout_started

        batch = FlatOptionBatch(
            features=_cat(segment_fields["features"]),
            expert=_cat(segment_fields["expert"]),
            old_log_prob=_cat(segment_fields["old_log_prob"]),
            value=_cat(segment_fields["value"]),
            segment_return=_cat(segment_fields["segment_return"]),
            next_value=_cat(segment_fields["next_value"]),
            duration=_cat(segment_fields["duration"]),
            done=_cat(segment_fields["done"]),
            policy_mask=_cat(segment_fields["policy_mask"]),
            env_id=_cat(segment_fields["env_id"]),
        )
        teacher_best = _cat(segment_fields["teacher_best"])
        counterfactual_mask = _cat(segment_fields["counterfactual_mask"])
        batch_gate_allowed = _cat(
            segment_fields["routing_gate_allowed"])
        advantage, returns = compute_flat_smdp_gae(
            batch, gamma=args.gamma, gae_lambda=args.gae_lambda)
        normalized_advantage = torch.zeros_like(advantage)
        if bool(batch.policy_mask.any()):
            selected_advantage = advantage[batch.policy_mask]
            normalized_advantage[batch.policy_mask] = (
                selected_advantage - selected_advantage.mean()
            ) / selected_advantage.std(unbiased=False).clamp_min(1.0e-8)

        update_started = time.perf_counter()
        totals = {key: 0.0 for key in (
            "policy_loss", "value_loss", "entropy", "approx_kl",
            "clip_fraction", "counterfactual_loss", "gradient_norm")}
        minibatches = 0
        cf_coefficient = _counterfactual_coefficient(args, update)
        stop_for_kl = False
        for _ in range(args.epochs):
            permutation = torch.randperm(len(batch.expert), device=device)
            for start in range(0, len(permutation), args.minibatch_size):
                ids = permutation[start:start + args.minibatch_size]
                outputs = router(batch.features[ids])
                update_logits = outputs["router_logits"].clone()
                update_logits[:, 1] = torch.where(
                    batch_gate_allowed[ids], update_logits[:, 1],
                    torch.full_like(update_logits[:, 1], -1.0e9))
                distribution = Categorical(logits=update_logits)
                mask = batch.policy_mask[ids]
                if bool(mask.any()):
                    policy_loss, approx_kl, clip_fraction = clipped_policy_loss(
                        distribution.log_prob(batch.expert[ids])[mask],
                        batch.old_log_prob[ids][mask],
                        normalized_advantage[ids][mask], args.clip_ratio)
                else:
                    policy_loss = outputs["router_logits"].sum() * 0.0
                    approx_kl = torch.zeros((), device=device)
                    clip_fraction = torch.zeros((), device=device)
                selected_q = outputs["q_values"].gather(
                    1, batch.expert[ids, None]).squeeze(1)
                value_loss = F.smooth_l1_loss(selected_q, returns[ids])
                entropy = distribution.entropy().mean()
                cf_mask = counterfactual_mask[ids]
                if bool(cf_mask.any()):
                    counterfactual_loss = F.cross_entropy(
                        outputs["router_logits"][cf_mask],
                        teacher_best[ids][cf_mask])
                else:
                    counterfactual_loss = outputs["router_logits"].sum() * 0.0
                loss = (
                    policy_loss + 0.5 * value_loss
                    - args.entropy_coef * entropy
                    + cf_coefficient * counterfactual_loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    router.parameters(), 2.0)
                optimizer.step()
                for key, value in (
                        ("policy_loss", policy_loss),
                        ("value_loss", value_loss),
                        ("entropy", entropy),
                        ("approx_kl", approx_kl),
                        ("clip_fraction", clip_fraction),
                        ("counterfactual_loss", counterfactual_loss),
                        ("gradient_norm", gradient_norm)):
                    totals[key] += float(value.detach())
                minibatches += 1
                if float(approx_kl.detach()) > args.target_kl:
                    stop_for_kl = True
                    break
            if stop_for_kl:
                break

        termination_loss_value = 0.0
        termination_entropy_value = 0.0
        termination_gradient_norm = 0.0
        if termination_features:
            term_features = torch.cat(termination_features, 0)
            term_expert = torch.cat(termination_expert, 0)
            term_dwell = torch.cat(termination_dwell, 0)
            term_eligible = torch.cat(termination_eligible, 0)
            term_gate_allowed = torch.cat(termination_gate_allowed, 0)
            term_distribution = router.termination_distribution(
                term_features, term_expert, term_dwell)
            term_outputs = router(term_features)
            active_q = term_outputs["q_values"].gather(
                1, term_expert[:, None]).squeeze(1)
            term_logits = term_outputs["router_logits"].clone()
            term_logits[:, 1] = torch.where(
                term_gate_allowed, term_logits[:, 1],
                torch.full_like(term_logits[:, 1], -1.0e9))
            term_state_value = (
                torch.softmax(term_logits, -1)
                * term_outputs["q_values"]).sum(-1)
            termination_loss = option_termination_loss(
                term_distribution.probs, active_q, term_state_value,
                term_eligible, deliberation_cost=0.01)
            optimizer.zero_grad(set_to_none=True)
            termination_loss.backward()
            termination_gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(router.parameters(), 2.0))
            optimizer.step()
            termination_loss_value = float(termination_loss.detach())
            eligible_prob = term_distribution.probs[term_eligible].clamp(
                1.0e-8, 1.0 - 1.0e-8)
            if eligible_prob.numel():
                termination_entropy_value = float((
                    -eligible_prob * eligible_prob.log()
                    -(1.0 - eligible_prob)
                    * (1.0 - eligible_prob).log()).mean())
        update_seconds = time.perf_counter() - update_started

        decisions_this_update = args.num_envs * args.rollout_steps
        continuation_decisions += decisions_this_update
        samples = float(decisions_this_update)
        bad_mean = counters["bad_cost"] / samples
        crash_mean = counters["crash_cost"] / samples
        bad_multiplier = max(
            0.0, bad_multiplier + args.lagrangian_learning_rate
            * (bad_mean - args.bad_cost_limit))
        crash_multiplier = max(
            0.0, crash_multiplier + args.lagrangian_learning_rate
            * (crash_mean - args.crash_cost_limit))
        divisor = max(minibatches, 1)
        return_variance = float(returns.var(unbiased=False))
        explained_variance = (
            1.0 - float((returns - batch.value).var(unbiased=False))
            / return_variance if return_variance > 1.0e-12 else 0.0)
        completed_episodes = max(int(counters["completed"]), 1)
        fire_requests = max(int(counters["fire_requests"]), 1)
        accepted_launches = max(int(counters["accepted"]), 1)
        metric = {
            "update": update,
            "environment_decisions": (
                prior_online_decisions + continuation_decisions),
            "completed_episodes": int(counters["completed"]),
            "learner_wins": int(counters["wins"]),
            "learner_losses": int(counters["losses"]),
            "draws": int(counters["draws"]),
            "timeouts": int(counters["timeouts"]),
            "crashes": int(counters["crashes"]),
            "mean_reward_ext": counters["reward"] / samples,
            "fire_requests": int(counters["fire_requests"]),
            "accepted_launches": int(counters["accepted"]),
            "rejected_fire_requests": int(counters["rejected"]),
            "legal_fire_opportunities": int(counters["legal"]),
            "hits": int(counters["hits"]),
            "learned_expert_steps": int(counters["learned_steps"]),
            "intercept_expert_steps": int(counters["intercept_steps"]),
            "router_intervention_steps": int(counters["router_steps"]),
            "forced_intervention_steps": int(counters["forced_steps"]),
            "expert_switches": int(counters["switches"]),
            "completed_option_segments": int(counters["segments"]),
            "mean_option_dwell": (
                counters["dwell_sum"] / max(counters["segments"], 1.0)),
            "reachability_lcb_learned_mean": counters["learned_lcb"] / samples,
            "reachability_lcb_intercept_mean": counters["intercept_lcb"] / samples,
            "reachability_regret_mean": counters["regret"] / samples,
            "force_condition_fraction": counters["force_condition"] / samples,
            "router_policy_loss": totals["policy_loss"] / divisor,
            "router_value_loss": totals["value_loss"] / divisor,
            "router_approx_kl": totals["approx_kl"] / divisor,
            "router_entropy": totals["entropy"] / divisor,
            "termination_loss": termination_loss_value,
            "low_policy_loss": 0.0,
            "low_approx_kl": 0.0,
            "bad_reachability_cost": bad_mean,
            "crash_cost": crash_mean,
            "bad_reachability_multiplier": bad_multiplier,
            "crash_multiplier": crash_multiplier,
            "intervention_regularization_weight": args.intervention_cost,
            "switch_regularization_weight": args.switch_cost,
            "intervention_regularization_cost": (
                counters["intervention_regularization"] / samples),
            "switch_regularization_cost": (
                counters["switch_regularization"] / samples),
            "routing_score_lcb_learned_mean": (
                counters["routing_lcb_learned"] / samples),
            "routing_score_lcb_intercept_mean": (
                counters["routing_lcb_intercept"] / samples),
            "routing_gate_allowed_fraction": (
                counters["routing_gate_allowed"] / samples),
            "router_intercept_margin": args.router_intercept_margin,
            "rollout_s": rollout_seconds,
            "teacher_s": reachability_seconds,
            "update_s": update_seconds,
            "environment_decisions_per_s": decisions_this_update / rollout_seconds,
            "online_environment_decisions": (
                prior_online_decisions + continuation_decisions),
            "model_branch_decisions": model_branch_decisions,
            "wall_time_s": time.perf_counter() - started,
            # Directly persisted comparison ratios; raw numerators and
            # denominators above remain authoritative for audit/recomputation.
            "episode_win_rate": counters["wins"] / completed_episodes,
            "episode_timeout_rate": counters["timeouts"] / completed_episodes,
            "legal_request_rate": (
                counters["fire_requests"] - counters["rejected"]
            ) / fire_requests,
            "hit_per_launch": counters["hits"] / accepted_launches,
            "intercept_fraction": counters["intercept_steps"] / samples,
            "forced_fraction": counters["forced_steps"] / samples,
            "switches_per_episode": counters["switches"] / completed_episodes,
            "termination_request_fraction": (
                counters["termination_requests"]
                / max(counters["termination_eligible"], 1.0)),
            "termination_entropy": termination_entropy_value,
            "router_explained_variance": explained_variance,
            "router_gradient_norm": totals["gradient_norm"] / divisor,
            "termination_gradient_norm": termination_gradient_norm,
            "router_clip_fraction": totals["clip_fraction"] / divisor,
            "simulation_substeps_per_s": (
                decisions_this_update * 12 / rollout_seconds),
            "end_to_end_decisions_per_s": (
                decisions_this_update
                / max(rollout_seconds + update_seconds, 1.0e-9)),
            "counterfactual_loss": totals["counterfactual_loss"] / divisor,
            "counterfactual_coefficient": cf_coefficient,
        }
        append_update_metrics(metrics_path, metric)
        history.append({
            **metric,
            "kl_early_stop": stop_for_kl,
        })
        progress = {
            "protocol_version": PROTOCOL_VERSION,
            "algorithm": ALGORITHM,
            "stage_label": args.stage_label,
            "low_level_policy_frozen": True,
            **history[-1],
            "updates": args.updates,
            "progress_fraction": update / args.updates,
        }
        write_json_atomic(args.output_dir / "progress.json", progress)
        write_json_atomic(args.output_dir / "train.json", history)
        checkpoint = {
            "protocol_version": PROTOCOL_VERSION,
            "algorithm": ALGORITHM,
            "stage_label": args.stage_label,
            "low_level_policy_frozen": True,
            "update": update,
            "router_input_dim": ROUTER_FEATURE_DIM,
            "router": router.state_dict(),
            "optimizer": optimizer.state_dict(),
            "snapshot": capture_execution_snapshot(env, execution_state),
            "shield": shield.state_dict(),
            "arbitrator": arbitrator.state_dict(),
            "continuation_decisions": continuation_decisions,
            "model_branch_decisions": model_branch_decisions,
            "bad_multiplier": bad_multiplier,
            "crash_multiplier": crash_multiplier,
            "history": history,
            "seed": args.seed,
            "v9_checkpoint": str(args.v9_checkpoint),
            "reachability_checkpoint": str(args.reachability_checkpoint),
            "router_score_checkpoint": (
                None if args.router_score_checkpoint is None
                else str(args.router_score_checkpoint)),
            "args": vars(args),
        }
        atomic_torch_save(checkpoint, latest_path)
        print(json.dumps(progress, ensure_ascii=False), flush=True)

    completed = {
        "protocol_version": PROTOCOL_VERSION,
        "algorithm": ALGORITHM,
        "stage_label": args.stage_label,
        "low_level_policy_frozen": True,
        "updates": args.updates,
        "continuation_environment_decisions": continuation_decisions,
        "total_environment_decisions_including_v9": (
            prior_online_decisions + continuation_decisions),
        "total_online_environment_decisions": (
            prior_online_decisions + continuation_decisions),
        "initial_v9_environment_decisions": prior_v9_decisions,
        "a1_online_environment_decisions": prior_a1_online_decisions,
        "return_teacher_online_environment_decisions": (
            prior_return_online_decisions),
        "return_teacher_model_branch_decisions": (
            return_model_branch_decisions),
        "lineage_online_environment_decisions": (
            args.lineage_online_decisions),
        "lineage_model_branch_decisions": args.lineage_model_decisions,
        "model_branch_decisions": model_branch_decisions,
        "last_metrics": history[-1] if history else None,
    }
    write_json_atomic(completed_path, completed)
    print(json.dumps(completed, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
