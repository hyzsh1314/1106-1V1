#!/usr/bin/env python3
"""Evaluate a frozen COACH-RHPPO policy against one fixed opponent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from benchmarks.benchmark_torch6dof_ppo import lowlevel_controls
from benchmarks.drvoi import extract_response_state, multi_discrete_to_one_hot
from benchmarks.drvoi_v7_1.history import VectorHistoryBuffer
from benchmarks.drvoi_v7_1.macro_options import (
    NUM_MACRO_OPTIONS,
    macro_option_legal_mask,
)
from benchmarks.drvoi_v9 import (
    AdaptiveOptionRecurrentActorCritic,
    confidence_gated_option_prior,
)
from benchmarks.evaluate_torch_wvr_fixed_opponents import scripted_actions
from benchmarks.train_torch_wvr_coach_rhppo import response_summary
from benchmarks.train_torch_wvr_mods_rhppo import (
    load_history_response,
    load_option_q,
    option_values,
)
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.utils.utils import get_root_dir
from envs.torch6dof import TorchWVRConfig, TorchWVRVecEnv


MODEL_CLASS = AdaptiveOptionRecurrentActorCritic
OPTION_LEGAL_MASK = macro_option_legal_mask
EXPECTED_ALGORITHM = "coach_rhppo"
EVALUATION_SUITE = "COACH-RHPPO-fixed-opponent"
PROTOCOL_VERSION = "COACH-RHPPO-V9-WVR-1"
ACTION_TRANSFORM = None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--response-checkpoint", type=Path, required=True)
    parser.add_argument("--option-q-checkpoint", type=Path, required=True)
    parser.add_argument("--opponent", choices=("hold", "pursuit", "evasive"),
                        required=True)
    parser.add_argument("--episodes", type=int, default=192)
    parser.add_argument("--num-envs", type=int, default=96)
    parser.add_argument("--samples-per-member", type=int, default=4)
    parser.add_argument("--gap-temperature", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=10085)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if payload.get("algorithm") != EXPECTED_ALGORITHM:
        raise ValueError(
            f"checkpoint is not an {EXPECTED_ALGORITHM} policy")
    model = MODEL_CLASS(
        obs_dim=int(payload["obs_dim"]),
        hidden_size=int(payload["hidden_size"]),
        response_feature_dim=int(payload["response_feature_dim"]),
        num_options=int(payload["num_options"]),
        option_embedding_size=int(payload["option_embedding_size"]),
    ).to(device).eval()
    model.load_state_dict(payload["model"])
    model.requires_grad_(False)
    response = load_history_response(
        args.response_checkpoint, int(payload["response_fold"]), device)
    _, q_folds = load_option_q(args.option_q_checkpoint, device)
    option_prior_coefficient = float(
        payload["history"][-1]["option_prior_coefficient"])
    option_commitment = int(payload["option_commitment"])

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
    hidden = model.initial_hidden(args.num_envs, device=device)
    previous_action = torch.zeros(args.num_envs, 13, device=device)
    reset = torch.ones(args.num_envs, dtype=torch.bool, device=device)
    active_option = torch.zeros(args.num_envs, dtype=torch.long, device=device)
    option_age = torch.full(
        (args.num_envs,), option_commitment, dtype=torch.long, device=device)
    response_history = VectorHistoryBuffer(
        args.num_envs, device=device, length=3)
    lowlevel_hidden = torch.zeros(args.num_envs * 2, 1, 128, device=device)

    episode_launches = torch.zeros(args.num_envs, dtype=torch.long, device=device)
    episode_hits = torch.zeros_like(episode_launches)
    episode_requests = torch.zeros_like(episode_launches)
    episode_rejected = torch.zeros_like(episode_launches)
    episode_legal = torch.zeros_like(episode_launches)
    episode_reward = torch.zeros(args.num_envs, device=device)
    target_red = args.episodes // 2
    target_blue = args.episodes - target_red
    learner_red = learner_blue = 0
    wins = losses = draws = timeouts = crashes = 0
    launches = hits = requests = rejected = legal = 0
    total_reward = 0.0
    lengths = []
    decisions = 0
    confidence_sum = 0.0
    confidence_count = 0
    option_counts = torch.zeros(
        NUM_MACRO_OPTIONS, dtype=torch.long, device=device)
    action_interventions = 0
    action_transform_steps = 0

    with torch.inference_mode():
        while learner_red < target_red or learner_blue < target_blue:
            learner_obs = obs[env_index, learner_side]
            belief, next_hidden = model.belief_step(
                learner_obs, previous_action, hidden, reset)
            provisional = model.conditioned_distributions(
                belief, active_option, learner_obs)
            provisional_action = torch.stack([
                distribution.probs.argmax(-1) for distribution in provisional
            ], -1)
            preview = response_history.preview(
                learner_obs, provisional_action, reset=reset)
            current_response, _ = extract_response_state(learner_obs)
            candidates = response["model"].sample_candidates(
                preview.context,
                preview.action_input,
                samples_per_member=args.samples_per_member,
                variance_scale=response["variance_scale"],
                target_mean=response["target_mean"],
                target_scale=response["target_scale"],
                current_response=current_response)
            features = response_summary(candidates)
            samples = candidates["continuous"].shape[1]
            provisional_one_hot = multi_discrete_to_one_hot(provisional_action)
            raw_context = torch.cat((
                belief[:, None, :].expand(-1, samples, -1),
                provisional_one_hot[:, None, :].expand(-1, samples, -1),
                candidates["continuous"],
            ), -1)
            legal_options = OPTION_LEGAL_MASK(learner_obs)
            flat_legal = legal_options[:, None, :].expand(
                -1, samples, -1).reshape(-1, NUM_MACRO_OPTIONS)
            flat_context = raw_context.reshape(-1, raw_context.shape[-1])
            q_a = option_values(q_folds[0], flat_context, flat_legal).reshape(
                args.num_envs, samples, NUM_MACRO_OPTIONS)
            q_b = option_values(q_folds[1], flat_context, flat_legal).reshape(
                args.num_envs, samples, NUM_MACRO_OPTIONS)
            prior = confidence_gated_option_prior(
                q_a, q_b, candidates["weights"], legal_options,
                gap_temperature=args.gap_temperature)
            decision = reset | (option_age >= option_commitment)
            option_logits = model.option_logits(
                belief, features, legal_options,
                prior_logits=prior["prior_logits"],
                prior_coefficient=option_prior_coefficient)
            proposed = option_logits.argmax(-1)
            active_option = torch.where(decision, proposed, active_option)
            distributions = model.conditioned_distributions(
                belief, active_option, learner_obs)
            learner_action = torch.stack([
                distribution.probs.argmax(-1) for distribution in distributions
            ], -1)
            if ACTION_TRANSFORM is not None:
                untransformed_action = learner_action
                learner_action = ACTION_TRANSFORM(
                    learner_obs, untransformed_action, reset)
                if learner_action.shape != untransformed_action.shape:
                    raise ValueError("ACTION_TRANSFORM changed action shape")
                action_interventions += int(
                    (learner_action != untransformed_action).any(-1).sum())
                action_transform_steps += args.num_envs
            response_history.append(
                learner_obs, learner_action, reset=reset)
            opponent_action = scripted_actions(obs, args.opponent)
            action = opponent_action.clone()
            action[env_index, learner_side] = learner_action
            controls, next_lowlevel_hidden = lowlevel_controls(
                lowlevel, lowlevel_hidden, obs, action,
                torch_speed_adapter=True)
            next_obs, reward, done, info = env.step(
                controls.reshape(args.num_envs, 2, 4), action[..., 3] == 1)
            encoded_action = multi_discrete_to_one_hot(learner_action)
            previous_action = torch.where(
                done[:, None], torch.zeros_like(encoded_action), encoded_action)
            option_counts += torch.bincount(
                active_option, minlength=NUM_MACRO_OPTIONS)
            confidence_sum += float(prior["confidence"].sum())
            confidence_count += args.num_envs
            option_age = torch.where(decision, torch.zeros_like(option_age), option_age)
            option_age = torch.where(done, torch.zeros_like(option_age), option_age + 1)
            active_option = torch.where(
                done, torch.zeros_like(active_option), active_option)
            hidden = next_hidden
            reset = done
            obs = next_obs
            lowlevel_hidden = torch.where(
                done.repeat_interleave(2)[:, None, None],
                torch.zeros_like(next_lowlevel_hidden), next_lowlevel_hidden)
            decisions += args.num_envs
            episode_reward += reward[env_index, learner_side]
            episode_launches += info[
                "accepted_launches"][env_index, learner_side].long()
            episode_hits += info["hits"][env_index, learner_side].long()
            episode_requests += info[
                "fire_requests"][env_index, learner_side].long()
            episode_rejected += info[
                "rejected_fire_requests"][env_index, learner_side].long()
            episode_legal += info[
                "legal_fire_opportunities"][env_index, learner_side].long()
            if not bool(done.any()):
                continue
            for index in torch.nonzero(
                    done, as_tuple=False).flatten().cpu().tolist():
                side = int(learner_side[index])
                if ((side == 0 and learner_red >= target_red)
                        or (side == 1 and learner_blue >= target_blue)):
                    continue
                if side == 0:
                    learner_red += 1
                else:
                    learner_blue += 1
                outcome_red = float(info["outcome_red"][index])
                outcome = outcome_red if side == 0 else -outcome_red
                wins += int(outcome > 0)
                losses += int(outcome < 0)
                draws += int(outcome == 0)
                timeouts += int(info["timeout"][index])
                crashes += int(info["crashes"][index, side])
                launches += int(episode_launches[index])
                hits += int(episode_hits[index])
                requests += int(episode_requests[index])
                rejected += int(episode_rejected[index])
                legal += int(episode_legal[index])
                total_reward += float(episode_reward[index])
                lengths.append(int(info["episode_length"][index]))
            for tensor in (
                    episode_launches, episode_hits, episode_requests,
                    episode_rejected, episode_legal, episode_reward):
                tensor.masked_fill_(done, 0)

    completed = learner_red + learner_blue
    report = {
        "suite": EVALUATION_SUITE,
        "protocol_version": PROTOCOL_VERSION,
        "scientific_result": False,
        "checkpoint": str(args.checkpoint),
        "checkpoint_seed": int(payload["seed"]),
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
        "timeout_rate": timeouts / completed,
        "crash_rate": crashes / completed,
        "mean_episode_reward": total_reward / completed,
        "accepted_launches": launches,
        "hits": hits,
        "hit_per_launch": hits / max(launches, 1),
        "legal_fire_opportunities": legal,
        "fire_requests": requests,
        "rejected_fire_requests": rejected,
        "accepted_per_request": launches / max(requests, 1),
        "legal_request_rate": (requests - rejected) / max(requests, 1),
        "mean_episode_length": sum(lengths) / len(lengths),
        "environment_decisions_executed": decisions,
        "option_prior_coefficient": option_prior_coefficient,
        "offline_prior_confidence_mean": confidence_sum / confidence_count,
        "option_counts": option_counts.cpu().tolist(),
    }
    if ACTION_TRANSFORM is not None:
        report["action_interventions"] = action_interventions
        report["action_intervention_fraction"] = (
            action_interventions / max(action_transform_steps, 1))
    rendered = json.dumps(report, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
