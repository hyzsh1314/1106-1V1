#!/usr/bin/env python3
"""Train COACH-RHPPO with explicit options and online twin option critics."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from benchmarks.benchmark_torch6dof_ppo import lowlevel_controls
from benchmarks.drvoi import extract_response_state, multi_discrete_to_one_hot
from benchmarks.drvoi.checkpointing import atomic_torch_save
from benchmarks.drvoi.rhppo import dual_channel_gae
from benchmarks.drvoi_v7_1.history import VectorHistoryBuffer
from benchmarks.drvoi_v7_1.macro_options import (
    NUM_MACRO_OPTIONS,
    macro_option_actions,
    macro_option_legal_mask,
)
from benchmarks.drvoi_v9 import (
    AdaptiveOptionRecurrentActorCritic,
    confidence_gated_option_prior,
    linear_anneal,
    option_segment_targets,
    polyak_update,
    progress_violation_events,
)
from benchmarks.train_torch_wvr_mods_rhppo import (
    load_history_response,
    load_option_q,
    option_values,
)
from benchmarks.train_torch_wvr_ppo import fixed_mixture_actions
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.utils.utils import get_root_dir
from envs.torch6dof import TorchWVRConfig, TorchWVRVecEnv


ALGORITHM = "coach_rhppo"
METHOD = "confidence_gated_online_adaptive_counterfactual_hierarchy"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rhppo-checkpoint", type=Path, required=True)
    parser.add_argument("--response-checkpoint", type=Path, required=True)
    parser.add_argument("--option-q-checkpoint", type=Path, required=True)
    parser.add_argument("--response-fold", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=96)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--burn-in", type=int, default=20)
    parser.add_argument("--chunks-per-minibatch", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--option-entropy-coef", type=float, default=0.001)
    parser.add_argument("--option-policy-coef", type=float, default=0.05)
    parser.add_argument("--option-target-kl", type=float, default=0.015)
    parser.add_argument("--samples-per-member", type=int, default=4)
    parser.add_argument("--gap-temperature", type=float, default=0.25)
    parser.add_argument("--option-commitment", type=int, default=6)
    parser.add_argument("--option-prior-initial", type=float, default=2.0)
    parser.add_argument("--option-prior-final", type=float, default=0.5)
    parser.add_argument("--macro-prior-initial", type=float, default=0.04)
    parser.add_argument("--macro-prior-final", type=float, default=0.01)
    parser.add_argument("--option-q-coef", type=float, default=0.5)
    parser.add_argument("--target-tau", type=float, default=0.02)
    parser.add_argument("--progress-window", type=int, default=60)
    parser.add_argument("--progress-range-delta", type=float, default=0.001)
    parser.add_argument("--progress-cost-limit", type=float, default=0.02)
    parser.add_argument("--lagrangian-initial", type=float, default=0.1)
    parser.add_argument("--lagrangian-lr", type=float, default=0.05)
    parser.add_argument("--lagrangian-max", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=85)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _write_json(path: Path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def response_summary(candidates: dict[str, torch.Tensor]) -> torch.Tensor:
    continuous = candidates["continuous"]
    weights = candidates["weights"]
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1.0e-8)
    mean = (continuous * weights.unsqueeze(-1)).sum(1)
    variance = (
        (continuous - mean[:, None, :]).square()
        * weights.unsqueeze(-1)
    ).sum(1)
    return torch.cat((mean, variance.clamp_min(1.0e-8).sqrt()), -1)


def _option_target_values(target_heads: nn.ModuleList, belief: torch.Tensor,
                          response_features: torch.Tensor,
                          legal: torch.Tensor) -> torch.Tensor:
    inputs = torch.cat((belief, response_features), -1)
    impossible = torch.finfo(inputs.dtype).min
    q_values = [
        head(inputs).masked_fill(~legal.bool(), impossible)
        for head in target_heads
    ]
    return torch.minimum(q_values[0], q_values[1]).amax(-1)


def coach_ppo_update(model, target_q_heads, optimizer, rollout, args,
                     option_prior_coefficient, macro_prior_coefficient,
                     lagrangian_multiplier):
    advantage = rollout["advantage_ext"]
    advantage = (advantage - advantage.mean()) / advantage.std(
        unbiased=False).clamp_min(1.0e-8)
    if bool((rollout["progress_cost"] > 0.0).any()):
        cost_advantage = rollout["advantage_int"]
        cost_advantage = (
            cost_advantage - cost_advantage.mean()
        ) / cost_advantage.std(unbiased=False).clamp_min(1.0e-8)
        advantage = advantage - float(lagrangian_multiplier) * cost_advantage
    decision_mask = rollout["option_decision"].bool()
    option_advantage = torch.zeros_like(rollout["option_advantage"])
    if bool(decision_mask.any()):
        selected_advantage = rollout["option_advantage"][decision_mask]
        selected_advantage = (
            selected_advantage - selected_advantage.mean()
        ) / selected_advantage.std(unbiased=False).clamp_min(1.0e-8)
        option_advantage[decision_mask] = selected_advantage
    time_steps, num_envs = advantage.shape
    starts = list(range(0, time_steps, args.sequence_length))
    keys = (
        "low_policy_loss", "option_policy_loss", "value_ext_loss",
        "value_cost_loss", "option_q_loss",
        "macro_prior_loss", "low_entropy", "option_entropy",
        "low_approx_kl", "option_approx_kl", "low_clip_fraction",
        "option_clip_fraction", "option_update_fraction", "grad_norm",
    )
    totals = {key: 0.0 for key in keys}
    updates = 0
    option_kl_stopped = False
    for _ in range(args.epochs):
        start_order = torch.randperm(len(starts)).tolist()
        for start_index in start_order:
            start = starts[start_index]
            end = min(start + args.sequence_length, time_steps)
            env_order = torch.randperm(num_envs, device=advantage.device)
            for offset in range(0, num_envs, args.chunks_per_minibatch):
                env_ids = env_order[offset:offset + args.chunks_per_minibatch]
                warm_start = max(0, start - args.burn_in)
                hidden = rollout["hidden_before"][warm_start, env_ids]
                if warm_start < start:
                    with torch.no_grad():
                        _, hidden = model.sequence_beliefs(
                            rollout["obs"][warm_start:start, env_ids],
                            rollout["prev_action"][warm_start:start, env_ids],
                            hidden,
                            rollout["reset"][warm_start:start, env_ids])
                beliefs, _ = model.sequence_beliefs(
                    rollout["obs"][start:end, env_ids],
                    rollout["prev_action"][start:end, env_ids],
                    hidden.detach(), rollout["reset"][start:end, env_ids])
                flat_belief = beliefs.reshape(-1, model.hidden_size)
                flat_obs = rollout["obs"][start:end, env_ids].reshape(-1, 39)
                flat_option = rollout["option"][start:end, env_ids].reshape(-1)
                flat_action = rollout["action"][start:end, env_ids].reshape(-1, 4)
                evaluated = model.evaluate_conditioned(
                    flat_belief, flat_option, flat_action, flat_obs)
                flat_response = rollout[
                    "response_features"][start:end, env_ids].reshape(
                        -1, model.response_feature_dim)
                flat_legal = rollout[
                    "legal_options"][start:end, env_ids].reshape(
                        -1, model.num_options)
                flat_prior = rollout[
                    "prior_logits"][start:end, env_ids].reshape(
                        -1, model.num_options)
                option_logits = model.option_logits(
                    flat_belief, flat_response, flat_legal,
                    prior_logits=flat_prior,
                    prior_coefficient=option_prior_coefficient)
                option_distribution = torch.distributions.Categorical(
                    logits=option_logits)
                decision = rollout[
                    "option_decision"][start:end, env_ids].reshape(-1).bool()
                new_option_log_prob = torch.where(
                    decision, option_distribution.log_prob(flat_option),
                    torch.zeros_like(evaluated["log_prob"]))
                old_low_log_prob = rollout[
                    "low_log_prob"][start:end, env_ids].reshape(-1)
                old_option_log_prob = rollout[
                    "option_log_prob"][start:end, env_ids].reshape(-1)
                flat_advantage = advantage[start:end, env_ids].reshape(-1)
                low_ratio = torch.exp(
                    evaluated["log_prob"] - old_low_log_prob)
                low_surrogate = torch.minimum(
                    low_ratio * flat_advantage,
                    low_ratio.clamp(
                        1.0 - args.clip_ratio,
                        1.0 + args.clip_ratio) * flat_advantage)
                low_policy_loss = -low_surrogate.mean()
                flat_option_advantage = option_advantage[
                    start:end, env_ids].reshape(-1)
                if bool(decision.any()):
                    option_ratio = torch.exp(
                        new_option_log_prob[decision]
                        - old_option_log_prob[decision])
                    selected_option_advantage = flat_option_advantage[decision]
                    option_surrogate = torch.minimum(
                        option_ratio * selected_option_advantage,
                        option_ratio.clamp(
                            1.0 - args.clip_ratio,
                            1.0 + args.clip_ratio)
                        * selected_option_advantage)
                    option_policy_loss = -option_surrogate.mean()
                else:
                    option_ratio = torch.ones(1, device=flat_belief.device)
                    option_policy_loss = option_logits.sum() * 0.0
                value_target = rollout[
                    "return_ext"][start:end, env_ids].reshape(-1)
                value_loss = 0.5 * (
                    evaluated["value_ext"] - value_target).square().mean()
                value_cost_target = rollout[
                    "return_int"][start:end, env_ids].reshape(-1)
                value_cost_loss = 0.5 * (
                    evaluated["value_int"] - value_cost_target
                ).square().mean()

                q_a, q_b = model.option_q_values(
                    flat_belief.detach(), flat_response, flat_legal)
                q_target = rollout[
                    "option_target"][start:end, env_ids].reshape(-1)
                decision_float = decision.float()
                if bool(decision.any()):
                    selected_q_a = q_a[decision].gather(
                        -1, flat_option[decision, None]).squeeze(-1)
                    selected_q_b = q_b[decision].gather(
                        -1, flat_option[decision, None]).squeeze(-1)
                    option_q_loss = 0.5 * (
                        (selected_q_a - q_target[decision]).square().mean()
                        + (selected_q_b - q_target[decision]).square().mean())
                else:
                    option_q_loss = q_a.sum() * 0.0

                macro_action = rollout[
                    "macro_action"][start:end, env_ids].reshape(-1, 4)
                macro_evaluated = model.evaluate_conditioned(
                    flat_belief, flat_option, macro_action, flat_obs)
                prior_weight = rollout[
                    "macro_prior_weight"][start:end, env_ids].reshape(-1)
                macro_prior_loss = -(
                    prior_weight * macro_evaluated["log_prob"] / 4.0
                ).sum() / prior_weight.sum().clamp_min(1.0)
                option_entropy = (
                    option_distribution.entropy() * decision_float
                ).sum() / decision_float.sum().clamp_min(1.0)
                low_entropy = evaluated["normalized_entropy"].mean()
                with torch.no_grad():
                    current_option_log_ratio = (
                        new_option_log_prob[decision]
                        - old_option_log_prob[decision]
                        if bool(decision.any()) else torch.zeros(
                            1, device=flat_belief.device))
                    current_option_kl = float((
                        (torch.exp(current_option_log_ratio) - 1.0)
                        - current_option_log_ratio).mean())
                option_update_enabled = not option_kl_stopped
                loss = (
                    low_policy_loss
                    + (args.option_policy_coef * option_policy_loss
                       if option_update_enabled else 0.0)
                    + 0.5 * value_loss + 0.5 * value_cost_loss
                    + args.option_q_coef * option_q_loss
                    + macro_prior_coefficient * macro_prior_loss
                    - args.entropy_coef * low_entropy
                    - (args.option_entropy_coef * option_entropy
                       if option_update_enabled else 0.0))
                with torch.no_grad():
                    low_log_ratio = evaluated["log_prob"] - old_low_log_prob
                    option_log_ratio = (
                        new_option_log_prob[decision]
                        - old_option_log_prob[decision]
                        if bool(decision.any()) else torch.zeros(
                            1, device=flat_belief.device))
                    metrics = (
                        float(low_policy_loss), float(option_policy_loss),
                        float(value_loss),
                        float(value_cost_loss), float(option_q_loss),
                        float(macro_prior_loss),
                        float(low_entropy), float(option_entropy),
                        float(((torch.exp(low_log_ratio) - 1.0)
                               - low_log_ratio).mean()),
                        float(((torch.exp(option_log_ratio) - 1.0)
                               - option_log_ratio).mean()),
                        float((torch.abs(low_ratio - 1.0) > args.clip_ratio)
                              .float().mean()),
                        float((torch.abs(option_ratio - 1.0) > args.clip_ratio)
                              .float().mean()),
                        float(option_update_enabled),
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 2.0)
                optimizer.step()
                if current_option_kl > args.option_target_kl:
                    option_kl_stopped = True
                for key, value in zip(keys[:-1], metrics):
                    totals[key] += value
                totals["grad_norm"] += float(grad_norm)
                updates += 1
    polyak_update(model.option_q_heads, target_q_heads, args.target_tau)
    return {key: value / max(updates, 1) for key, value in totals.items()}


def main():
    args = parse_args()
    if args.rollout_steps % args.sequence_length != 0:
        raise ValueError("rollout-steps must be divisible by sequence-length")
    if args.option_commitment <= 0:
        raise ValueError("option-commitment must be positive")
    if args.progress_window <= 0:
        raise ValueError("progress-window must be positive")
    if args.samples_per_member <= 0 or args.samples_per_member % 2:
        raise ValueError("samples-per-member must be a positive even number")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    initial = torch.load(
        args.rhppo_checkpoint, map_location=device, weights_only=False)
    model = AdaptiveOptionRecurrentActorCritic(
        obs_dim=int(initial["obs_dim"]),
        hidden_size=int(initial["hidden_size"]),
        response_feature_dim=24,
        num_options=NUM_MACRO_OPTIONS,
        option_embedding_size=16,
    ).to(device)
    model.load_from_rhppo(initial["model"])
    target_q_heads = copy.deepcopy(model.option_q_heads).to(device).eval()
    target_q_heads.requires_grad_(False)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, eps=1.0e-5)
    response = load_history_response(
        args.response_checkpoint, args.response_fold, device)
    q_payload, offline_q_folds = load_option_q(
        args.option_q_checkpoint, device)
    history = []
    total_decisions = 0
    start_update = 1
    lagrangian_multiplier = float(args.lagrangian_initial)
    latest_path = args.output_dir / "latest.pt"
    if args.resume and latest_path.exists():
        resumed = torch.load(latest_path, map_location=device, weights_only=False)
        if resumed.get("algorithm") != ALGORITHM:
            raise ValueError("resume checkpoint uses a different algorithm")
        model.load_state_dict(resumed["model"])
        target_q_heads.load_state_dict(resumed["target_q_heads"])
        optimizer.load_state_dict(resumed["optimizer"])
        history = list(resumed.get("history", []))
        total_decisions = int(resumed["training_environment_decisions"])
        start_update = int(resumed["completed_updates"]) + 1
        lagrangian_multiplier = float(resumed.get(
            "lagrangian_multiplier", lagrangian_multiplier))

    env = TorchWVRVecEnv(TorchWVRConfig(
        num_envs=args.num_envs,
        reward_mode="competitive",
        attack_quality_weight=0.0,
        observation_version="v4",
        use_dynamic_launch_envelope=True,
    ), device=device, seed=args.seed, compile_dynamics=True)
    lowlevel = BaselineActor().to(device).eval()
    lowlevel.load_state_dict(torch.load(
        get_root_dir() + "/model/baseline_model.pt",
        map_location=device, weights_only=True))
    lowlevel.requires_grad_(False)
    lowlevel.tpdv = {"dtype": torch.float32, "device": device}

    env_index = torch.arange(args.num_envs, device=device)
    learner_side = env_index.remainder(2)
    obs = env.observations()
    learner_obs = obs[env_index, learner_side]
    hidden = model.initial_hidden(args.num_envs, device=device)
    previous_action = torch.zeros(args.num_envs, 13, device=device)
    reset = torch.ones(args.num_envs, dtype=torch.bool, device=device)
    active_option = torch.zeros(
        args.num_envs, dtype=torch.long, device=device)
    option_age = torch.full(
        (args.num_envs,), args.option_commitment,
        dtype=torch.long, device=device)
    lowlevel_hidden = torch.zeros(
        args.num_envs * 2, 1, 128, device=device)
    response_history = VectorHistoryBuffer(
        args.num_envs, device=device, length=3)
    stagnation_steps = torch.zeros(
        args.num_envs, dtype=torch.long, device=device)

    with torch.inference_mode():
        warm = torch.tensor(
            [1, 2, 1, 0], device=device).expand(args.num_envs, -1)
        warm_all = fixed_mixture_actions(obs, env_index.remainder(3))
        warm_all[env_index, learner_side] = warm
        warm_controls, _ = lowlevel_controls(
            lowlevel, lowlevel_hidden, obs, warm_all,
            torch_speed_adapter=True)
        env.dynamics_runner(
            env.aircraft_state.reshape(-1, env.dynamics.state_size),
            warm_controls, substeps=1)
    torch.cuda.synchronize()
    started = time.perf_counter()

    for update in range(start_update, args.updates + 1):
        option_prior_coefficient = linear_anneal(
            args.option_prior_initial, args.option_prior_final,
            update, args.updates)
        macro_prior_coefficient = linear_anneal(
            args.macro_prior_initial, args.macro_prior_final,
            update, args.updates)
        buffers = {key: [] for key in (
            "obs", "prev_action", "hidden_before", "reset", "belief",
            "response_features", "legal_options", "prior_logits",
            "option", "option_decision", "macro_action",
            "macro_prior_weight", "option_value", "action",
            "low_log_prob", "option_log_prob", "value_ext",
            "value_cost", "next_value_ext", "next_value_cost",
            "reward_ext", "progress_cost", "done")}
        completed = wins = losses = timeouts = launches = hits = 0
        confidence_values = []
        fold_gap_values = []
        decision_values = []
        option_counts = torch.zeros(
            NUM_MACRO_OPTIONS, dtype=torch.long, device=device)
        rollout_started = time.perf_counter()

        for _ in range(args.rollout_steps):
            buffers["obs"].append(learner_obs.clone())
            buffers["prev_action"].append(previous_action.clone())
            buffers["hidden_before"].append(hidden.clone())
            buffers["reset"].append(reset.clone())
            with torch.inference_mode():
                belief, next_hidden = model.belief_step(
                    learner_obs, previous_action, hidden, reset)
                provisional_distributions = model.conditioned_distributions(
                    belief, active_option, learner_obs)
                provisional_action = torch.stack([
                    distribution.sample()
                    for distribution in provisional_distributions
                ], -1)
                preview_history = response_history.preview(
                    learner_obs, provisional_action, reset=reset)
                current_response, _ = extract_response_state(learner_obs)
                candidates = response["model"].sample_candidates(
                    preview_history.context,
                    preview_history.action_input,
                    samples_per_member=args.samples_per_member,
                    variance_scale=response["variance_scale"],
                    target_mean=response["target_mean"],
                    target_scale=response["target_scale"],
                    current_response=current_response)
                features = response_summary(candidates)
                samples = candidates["continuous"].shape[1]
                provisional_one_hot = multi_discrete_to_one_hot(
                    provisional_action)
                raw_context = torch.cat((
                    belief[:, None, :].expand(-1, samples, -1),
                    provisional_one_hot[:, None, :].expand(-1, samples, -1),
                    candidates["continuous"],
                ), -1)
                flat_context = raw_context.reshape(
                    -1, raw_context.shape[-1])
                legal = macro_option_legal_mask(learner_obs)
                flat_legal = legal[:, None, :].expand(
                    -1, samples, -1).reshape(-1, NUM_MACRO_OPTIONS)
                q_a = option_values(
                    offline_q_folds[0], flat_context, flat_legal,
                ).reshape(args.num_envs, samples, NUM_MACRO_OPTIONS)
                q_b = option_values(
                    offline_q_folds[1], flat_context, flat_legal,
                ).reshape(args.num_envs, samples, NUM_MACRO_OPTIONS)
                prior = confidence_gated_option_prior(
                    q_a, q_b, candidates["weights"], legal,
                    gap_temperature=args.gap_temperature)
                decision = reset | (option_age >= args.option_commitment)
                option_logits = model.option_logits(
                    belief, features, legal,
                    prior_logits=prior["prior_logits"],
                    prior_coefficient=option_prior_coefficient)
                option_distribution = torch.distributions.Categorical(
                    logits=option_logits)
                proposed_option = option_distribution.sample()
                active_option = torch.where(
                    decision, proposed_option, active_option)
                option_log_prob = torch.where(
                    decision, option_distribution.log_prob(proposed_option),
                    torch.zeros(args.num_envs, device=device))
                distributions = model.conditioned_distributions(
                    belief, active_option, learner_obs)
                learner_action = torch.stack([
                    distribution.sample() for distribution in distributions
                ], -1)
                low_log_prob = sum(
                    distribution.log_prob(learner_action[:, index])
                    for index, distribution in enumerate(distributions))
                online_q_a, online_q_b = model.option_q_values(
                    belief, features, legal)
                online_q = torch.minimum(online_q_a, online_q_b)
                selected_option_value = online_q.gather(
                    -1, active_option[:, None]).squeeze(-1)
                selected_option_value = torch.where(
                    decision, selected_option_value,
                    torch.zeros_like(selected_option_value))
                macro_action = macro_option_actions(
                    learner_obs, active_option, elapsed=1)
                macro_action[:, 3] = (
                    (active_option == NUM_MACRO_OPTIONS - 1)
                    & decision & (learner_obs[:, 23] >= 0.5)
                ).long()
                legal_count = legal.sum(-1).clamp_min(1)
                legal_mean = (
                    prior["conservative_value"].masked_fill(~legal, 0.0).sum(-1)
                    / legal_count)
                active_value = prior["conservative_value"].gather(
                    -1, active_option[:, None]).squeeze(-1)
                active_advantage = (active_value - legal_mean).clamp_min(0.0)
                macro_prior_weight = prior["confidence"] * torch.tanh(
                    active_advantage / 0.25)
                response_history.append(
                    learner_obs, learner_action, reset=reset)

                executed = fixed_mixture_actions(
                    obs, env_index.remainder(3))
                executed[env_index, learner_side] = learner_action
                controls, next_lowlevel_hidden = lowlevel_controls(
                    lowlevel, lowlevel_hidden, obs, executed,
                    torch_speed_adapter=True)
                next_obs, reward, done, info = env.step(
                    controls.reshape(args.num_envs, 2, 4),
                    executed[..., 3] == 1)
                next_learner_obs = next_obs[env_index, learner_side]
                action_one_hot = multi_discrete_to_one_hot(learner_action)
                next_previous_action = torch.where(
                    done[:, None], torch.zeros_like(action_one_hot),
                    action_one_hot)
                next_belief, _ = model.belief_step(
                    next_learner_obs, next_previous_action,
                    next_hidden, reset=done)
                current_value_ext = model.value_ext(belief).squeeze(-1)
                current_value_cost = model.value_int(belief).squeeze(-1)
                next_value_ext = model.value_ext(next_belief).squeeze(-1)
                next_value_cost = model.value_int(next_belief).squeeze(-1)

                closing_progress = learner_obs[:, 28] > 0.05
                range_progress = (
                    next_learner_obs[:, 13]
                    < learner_obs[:, 13] - args.progress_range_delta)
                launch_progress = (
                    (learner_obs[:, 23] >= 0.5)
                    | (next_learner_obs[:, 23] >= 0.5))
                made_progress = closing_progress | range_progress | launch_progress
                stagnation_steps = torch.where(
                    made_progress | done,
                    torch.zeros_like(stagnation_steps),
                    stagnation_steps + 1)
                progress_cost = progress_violation_events(
                    stagnation_steps, args.progress_window)

            learner_reward = reward[env_index, learner_side]
            stored = (
                ("belief", belief),
                ("response_features", features),
                ("legal_options", legal),
                ("prior_logits", prior["prior_logits"]),
                ("option", active_option),
                ("option_decision", decision),
                ("macro_action", macro_action),
                ("macro_prior_weight", macro_prior_weight),
                ("option_value", selected_option_value),
                ("action", learner_action),
                ("low_log_prob", low_log_prob),
                ("option_log_prob", option_log_prob),
                ("value_ext", current_value_ext),
                ("value_cost", current_value_cost),
                ("next_value_ext", next_value_ext),
                ("next_value_cost", next_value_cost),
                ("reward_ext", learner_reward),
                ("progress_cost", progress_cost),
                ("done", done),
            )
            for key, value in stored:
                buffers[key].append(value.clone())
            outcome = torch.where(
                learner_side == 0,
                info["outcome_red"], -info["outcome_red"])
            completed += int(done.sum())
            wins += int(((outcome > 0) & done).sum())
            losses += int(((outcome < 0) & done).sum())
            timeouts += int(info["timeout"].sum())
            launches += int(info[
                "accepted_launches"][env_index, learner_side].sum())
            hits += int(info["hits"][env_index, learner_side].sum())
            confidence_values.append(prior["confidence"])
            fold_gap_values.append(prior["fold_gap"])
            decision_values.append(decision.float())
            option_counts += torch.bincount(
                active_option, minlength=NUM_MACRO_OPTIONS)
            option_age = torch.where(decision, torch.zeros_like(option_age), option_age)
            option_age = torch.where(done, torch.zeros_like(option_age), option_age + 1)
            active_option = torch.where(
                done, torch.zeros_like(active_option), active_option)
            hidden = next_hidden
            previous_action = next_previous_action
            reset = done
            obs = next_obs
            learner_obs = next_learner_obs
            lowlevel_hidden = torch.where(
                done.repeat_interleave(2)[:, None, None],
                torch.zeros_like(next_lowlevel_hidden),
                next_lowlevel_hidden)

        rollout_s = time.perf_counter() - rollout_started
        rollout = {key: torch.stack(value) for key, value in buffers.items()}
        rollout.update(dual_channel_gae(
            rollout["reward_ext"], rollout["progress_cost"],
            rollout["value_ext"], rollout["value_cost"],
            rollout["next_value_ext"], rollout["next_value_cost"],
            rollout["done"], args.gamma, args.gae_lambda))
        with torch.no_grad():
            target_values = _option_target_values(
                target_q_heads,
                rollout["belief"].reshape(-1, model.hidden_size),
                rollout["response_features"].reshape(
                    -1, model.response_feature_dim),
                rollout["legal_options"].reshape(
                    -1, model.num_options),
            ).reshape(args.rollout_steps, args.num_envs)
            constrained_reward = (
                rollout["reward_ext"]
                - lagrangian_multiplier * rollout["progress_cost"])
            state_bootstrap = torch.cat((
                target_values,
                (rollout["next_value_ext"][-1:]
                 - lagrangian_multiplier
                 * rollout["next_value_cost"][-1:]).clone(),
            ), 0)
            rollout["option_target"] = option_segment_targets(
                constrained_reward, rollout["done"],
                state_bootstrap, args.option_commitment, args.gamma)
            rollout["option_advantage"] = (
                rollout["option_target"] - rollout["option_value"])
        update_started = time.perf_counter()
        metrics = coach_ppo_update(
            model, target_q_heads, optimizer, rollout, args,
            option_prior_coefficient, macro_prior_coefficient,
            lagrangian_multiplier)
        update_s = time.perf_counter() - update_started
        observed_cost = float(rollout["progress_cost"].mean())
        lagrangian_multiplier = min(
            args.lagrangian_max,
            max(0.0, lagrangian_multiplier + args.lagrangian_lr * (
                observed_cost - args.progress_cost_limit)))
        total_decisions += args.num_envs * args.rollout_steps
        confidence = torch.cat(confidence_values)
        fold_gap = torch.cat(fold_gap_values)
        option_decisions = torch.cat(decision_values)
        entry = {
            "update": update,
            "environment_decisions": total_decisions,
            "completed_episodes": completed,
            "learner_wins": wins,
            "learner_losses": losses,
            "timeouts": timeouts,
            "learner_accepted_launches": launches,
            "learner_hits": hits,
            "mean_reward_ext": float(rollout["reward_ext"].mean()),
            "progress_cost_mean": observed_cost,
            "lagrangian_multiplier": lagrangian_multiplier,
            "option_prior_coefficient": option_prior_coefficient,
            "macro_prior_coefficient": macro_prior_coefficient,
            "option_decision_fraction": float(option_decisions.mean()),
            "offline_prior_confidence_mean": float(confidence.mean()),
            "offline_q_fold_gap_mean": float(fold_gap.mean()),
            "macro_prior_weight_mean": float(
                rollout["macro_prior_weight"].mean()),
            "option_counts": option_counts.cpu().tolist(),
            "rollout_s": rollout_s,
            "ppo_update_s": update_s,
            **metrics,
        }
        history.append(entry)
        checkpoint = {
            "algorithm": ALGORITHM,
            "method": METHOD,
            "model": model.state_dict(),
            "target_q_heads": target_q_heads.state_dict(),
            "optimizer": optimizer.state_dict(),
            "obs_dim": env.observation_size,
            "hidden_size": model.hidden_size,
            "response_feature_dim": model.response_feature_dim,
            "num_options": model.num_options,
            "option_embedding_size": model.option_embedding_size,
            "action_dims": [3, 5, 3, 2],
            "seed": args.seed,
            "completed_updates": update,
            "training_environment_decisions": total_decisions,
            "environment_config": asdict(env.config),
            "history": history,
            "response_fold": args.response_fold,
            "response_held_collection_seed": response[
                "held_collection_seed"],
            "offline_option_q_target": q_payload["target"],
            "offline_option_q_train_seeds": q_payload["train_seeds"],
            "option_commitment": args.option_commitment,
            "lagrangian_multiplier": lagrangian_multiplier,
        }
        atomic_torch_save(checkpoint, latest_path)
        _write_json(args.output_dir / "progress.json", entry)
        print(json.dumps(entry), flush=True)

    torch.cuda.synchronize()
    report = {
        "algorithm": ALGORITHM,
        "method": METHOD,
        "scientific_result": False,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "rollout_steps": args.rollout_steps,
        "updates": args.updates,
        "total_environment_decisions": total_decisions,
        "elapsed_s": time.perf_counter() - started,
        "history": history,
    }
    _write_json(args.output_dir / "train.json", report)


if __name__ == "__main__":
    main()
