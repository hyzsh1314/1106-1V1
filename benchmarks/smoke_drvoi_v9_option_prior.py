#!/usr/bin/env python3
"""One-batch integration smoke test for the COACH-RHPPO option prior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from benchmarks.drvoi import extract_response_state, multi_discrete_to_one_hot
from benchmarks.drvoi.rhppo import RecurrentMultiDiscreteActorCritic
from benchmarks.drvoi_v7_1.history import VectorHistoryBuffer
from benchmarks.drvoi_v7_1.macro_options import (
    NUM_MACRO_OPTIONS,
    macro_option_legal_mask,
)
from benchmarks.drvoi_v9 import confidence_gated_option_prior
from benchmarks.train_torch_wvr_mods_rhppo import (
    load_history_response,
    load_option_q,
    option_values,
)
from envs.torch6dof import TorchWVRConfig, TorchWVRVecEnv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rhppo-checkpoint", type=Path, required=True)
    parser.add_argument("--response-checkpoint", type=Path, required=True)
    parser.add_argument("--option-q-checkpoint", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=96)
    parser.add_argument("--samples-per-member", type=int, default=4)
    parser.add_argument("--gap-temperature", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=85)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    initial = torch.load(
        args.rhppo_checkpoint, map_location=device, weights_only=False)
    policy = RecurrentMultiDiscreteActorCritic(
        obs_dim=int(initial["obs_dim"]),
        hidden_size=int(initial["hidden_size"]),
    ).to(device).eval()
    policy.load_state_dict(initial["model"])
    policy.requires_grad_(False)
    response = load_history_response(args.response_checkpoint, 0, device)
    _, q_folds = load_option_q(args.option_q_checkpoint, device)
    env = TorchWVRVecEnv(TorchWVRConfig(
        num_envs=args.num_envs,
        reward_mode="competitive",
        attack_quality_weight=0.0,
        observation_version="v4",
        use_dynamic_launch_envelope=True,
    ), device=device, seed=args.seed, compile_dynamics=False)
    env_index = torch.arange(args.num_envs, device=device)
    learner_side = env_index.remainder(2)
    learner_obs = env.observations()[env_index, learner_side]
    hidden = policy.initial_hidden(args.num_envs, device=device)
    previous_action = torch.zeros(args.num_envs, 13, device=device)
    history = VectorHistoryBuffer(args.num_envs, device=device, length=3)
    with torch.inference_mode():
        result = policy.act(
            learner_obs, previous_action, hidden,
            reset=torch.ones(args.num_envs, dtype=torch.bool, device=device))
        action_one_hot = multi_discrete_to_one_hot(result["action"])
        history_batch = history.append(
            learner_obs, result["action"],
            reset=torch.ones(args.num_envs, dtype=torch.bool, device=device))
        current_response, _ = extract_response_state(learner_obs)
        candidates = response["model"].sample_candidates(
            history_batch.context,
            history_batch.action_input,
            samples_per_member=args.samples_per_member,
            variance_scale=response["variance_scale"],
            target_mean=response["target_mean"],
            target_scale=response["target_scale"],
            current_response=current_response,
        )
        samples = candidates["continuous"].shape[1]
        raw_context = torch.cat((
            result["belief"][:, None, :].expand(-1, samples, -1),
            action_one_hot[:, None, :].expand(-1, samples, -1),
            candidates["continuous"],
        ), dim=-1)
        flat_context = raw_context.reshape(-1, raw_context.shape[-1])
        legal = macro_option_legal_mask(learner_obs)
        flat_legal = legal[:, None, :].expand(
            -1, samples, -1).reshape(-1, NUM_MACRO_OPTIONS)
        q_a = option_values(
            q_folds[0], flat_context, flat_legal,
        ).reshape(args.num_envs, samples, NUM_MACRO_OPTIONS)
        q_b = option_values(
            q_folds[1], flat_context, flat_legal,
        ).reshape(args.num_envs, samples, NUM_MACRO_OPTIONS)
        prior = confidence_gated_option_prior(
            q_a, q_b, candidates["weights"], legal,
            gap_temperature=args.gap_temperature)
    counts = torch.bincount(
        prior["selected_option"], minlength=NUM_MACRO_OPTIONS)
    selected_legal = legal.gather(
        -1, prior["selected_option"].unsqueeze(-1)).squeeze(-1)
    report = {
        "batch_size": args.num_envs,
        "candidate_samples": samples,
        "finite": bool(torch.isfinite(prior["confidence"]).all()),
        "illegal_selected": int((~selected_legal).sum()),
        "confidence_mean": float(prior["confidence"].mean()),
        "confidence_p10": float(prior["confidence"].quantile(0.1)),
        "confidence_p90": float(prior["confidence"].quantile(0.9)),
        "fold_gap_mean": float(prior["fold_gap"].mean()),
        "positive_advantage_fraction": float(
            (prior["option_advantage"] > 1.0e-6).float().mean()),
        "selected_option_counts": counts.cpu().tolist(),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
