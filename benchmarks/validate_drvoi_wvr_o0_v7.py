#!/usr/bin/env python3
"""Run the V7 O0 VOI software/signal smoke gate without policy training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.drvoi.gates import evaluate_o0_gate
from benchmarks.drvoi.q_model import ActionEmbeddingQ
from benchmarks.drvoi.response import ContinuousResponseEnsemble
from benchmarks.drvoi.targets import TargetStandardizer, response_state_v7
from benchmarks.validate_drvoi_wvr_response_v7 import (
    flatten_response_data,
    load_episodes,
    stratified_action_permutation,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response-checkpoint", type=Path, required=True)
    parser.add_argument("--q-checkpoint", type=Path, required=True)
    parser.add_argument("--q-gate", type=Path, required=True)
    parser.add_argument("--onpolicy", type=Path, nargs="+", required=True)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/drvoi_rhppo_v7_wvr.json"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--stability-contexts", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=11185)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def rankdata(values: torch.Tensor) -> torch.Tensor:
    _, inverse, counts = torch.unique(
        values, sorted=True, return_inverse=True, return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    return (starts.float() + 0.5 * (counts.float() - 1.0))[inverse]


def spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    x = rankdata(left) - rankdata(left).mean()
    y = rankdata(right) - rankdata(right).mean()
    return float((x * y).sum() / (
        x.square().sum().sqrt() * y.square().sum().sqrt()).clamp_min(1.0e-8))


def voi_from_q_values(q_values: torch.Tensor):
    """EVPI-style VOI from `[B,S,A]` response-conditioned Q values."""
    if q_values.ndim != 3:
        raise ValueError("Q values must have [batch,samples,actions] shape")
    expected_perfect = q_values.max(-1).values.mean(-1)
    best_expected = q_values.mean(1).max(-1).values
    voi = (expected_perfect - best_expected).clamp_min(0.0)
    argmax = q_values.argmax(-1)
    diverse = (argmax != argmax[:, :1]).any(-1)
    mean_best = q_values.mean(1).argmax(-1)
    top_flip = (argmax != mean_best[:, None]).any(-1)
    return voi, diverse, top_flip


def build_models(response_payload, q_payload, device):
    response = ContinuousResponseEnsemble(
        int(response_payload["input_dim"]),
        int(response_payload["response_dim"]),
        members=int(response_payload["members"]),
        hidden_size=int(response_payload["hidden_size"]),
        sigma_min=float(response_payload["sigma_min"]),
        sigma_max=float(response_payload["sigma_max"]),
    ).to(device).eval()
    response.load_state_dict(response_payload["conditioned"])
    response.requires_grad_(False)
    q_models = {}
    for name in ("a", "b"):
        model = ActionEmbeddingQ(int(q_payload["context_dim"])).to(device).eval()
        model.load_state_dict(q_payload["folds"][name]["online"])
        model.requires_grad_(False)
        q_models[name] = model
    return response, q_models


@torch.no_grad()
def score_contexts(response, q_models, q_payload, data, standardizer,
                   variance_scale, samples, batch_size, seed,
                   response_action=None):
    result = []
    diversity = []
    flips = []
    action_input = data.action if response_action is None else response_action
    for start in range(0, data.action.shape[0], batch_size):
        selected = slice(start, start + batch_size)
        belief = data.belief[selected]
        action = data.action[selected]
        current = response_state_v7(data.observation[selected], "rel")
        generator = torch.Generator(device=belief.device).manual_seed(
            seed + start)
        candidates = response.sample_candidates(
            belief,
            action_input[selected],
            samples_per_member=samples // len(response.members),
            generator=generator,
            variance_scale=variance_scale,
            target_mean=standardizer.mean,
            target_scale=standardizer.scale,
            current_response=current,
        )["continuous"]
        batch, count, _ = candidates.shape
        context = torch.cat((
            belief[:, None].expand(-1, count, -1),
            action[:, None].expand(-1, count, -1),
            candidates,
        ), -1)
        q_values = torch.empty(
            batch, count, 90, device=belief.device)
        episode_fold = data.episode[selected] & 1
        # Critic A trained fold 0 is used for fold-1 episodes and vice versa.
        for episode_value, model_name in ((0, "b"), (1, "a")):
            rows = episode_fold == episode_value
            if not bool(rows.any()):
                continue
            mean = q_payload["folds"][model_name]["context_mean"].to(belief.device)
            scale = q_payload["folds"][model_name]["context_scale"].to(belief.device)
            normalized = (context[rows] - mean) / scale
            q_values[rows] = q_models[model_name].all_actions(
                normalized.reshape(-1, normalized.shape[-1])).reshape(
                    int(rows.sum()), count, 90)
        voi, diverse, top_flip = voi_from_q_values(q_values)
        result.append(voi)
        diversity.append(diverse)
        flips.append(top_flip)
    return torch.cat(result), torch.cat(diversity), torch.cat(flips)


def episode_bootstrap_gain(values, episodes, seed):
    unique = torch.unique(episodes)
    means = torch.stack([
        values[episodes == episode].mean() for episode in unique
    ]).cpu().numpy()
    rng = np.random.default_rng(seed)
    bootstrap = [rng.choice(means, len(means), replace=True).mean()
                 for _ in range(1000)]
    return float(means.mean()), np.quantile(bootstrap, [0.025, 0.975]).tolist()


def write_blocked(args, reason):
    report = {
        "suite": "DRVOI-RHPPO-V7-O0",
        "valid": False,
        "reason": reason,
        "intrinsic_reward": 0.0,
        "long_training_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output.parent / "o0_gate.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(2)


def main():
    args = parse_args()
    q_gate = json.loads(args.q_gate.read_text(encoding="utf-8"))
    if not q_gate.get("ready", False):
        write_blocked(args, "q_ready_false")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    response_payload = torch.load(
        args.response_checkpoint, map_location=device, weights_only=False)
    q_payload = torch.load(args.q_checkpoint, map_location=device, weights_only=False)
    response, q_models = build_models(response_payload, q_payload, device)
    episodes = load_episodes(args.onpolicy, "onpolicy")
    data = flatten_response_data(episodes, "test", 3, "rel").to(device)
    standardizer = TargetStandardizer.from_state_dict(
        response_payload["standardizer"])
    standardizer.mean = standardizer.mean.to(device)
    standardizer.scale = standardizer.scale.to(device)
    variance_scale = response_payload["variance_scale"].to(device)
    correct, diverse, top_flip = score_contexts(
        response, q_models, q_payload, data, standardizer, variance_scale,
        20, args.batch_size, args.seed)
    shuffled_action = stratified_action_permutation(
        data.action, data.opponent, data.phase, args.seed + 1)
    shuffled, _, _ = score_contexts(
        response, q_models, q_payload, data, standardizer, variance_scale,
        20, args.batch_size, args.seed, response_action=shuffled_action)
    shuffle_gain, shuffle_ci = episode_bootstrap_gain(
        correct - shuffled, data.episode, args.seed + 2)
    probe_count = min(args.stability_contexts, data.action.shape[0])
    indices = torch.linspace(
        0, data.action.shape[0] - 1, probe_count,
        device=device).long()
    probe = data.select(indices)
    s20, _, _ = score_contexts(
        response, q_models, q_payload, probe, standardizer, variance_scale,
        20, args.batch_size, args.seed + 3)
    s100, _, _ = score_contexts(
        response, q_models, q_payload, probe, standardizer, variance_scale,
        100, max(args.batch_size // 2, 1), args.seed + 3)
    threshold = 1.0e-6
    finite = bool(torch.isfinite(correct).all() and torch.isfinite(s100).all())
    restore_response, restore_q = build_models(response_payload, q_payload, device)
    with torch.no_grad():
        original = response(data.belief[:32], data.action[:32])[0]
        restored = restore_response(data.belief[:32], data.action[:32])[0]
        checkpoint_restore_pass = bool(torch.equal(original, restored))
        for name in ("a", "b"):
            checkpoint_restore_pass &= all(
                torch.equal(left, right) for left, right in zip(
                    q_models[name].parameters(), restore_q[name].parameters()))
    metrics = {
        "valid_voi_contexts": int(correct.numel()),
        "positive_voi_nonzero_fraction": float((correct > threshold).float().mean()),
        "candidate_argmax_diversity_fraction": float(diverse.float().mean()),
        "top_action_flip_fraction": float(top_flip.float().mean()),
        "mean_voi": float(correct.mean()),
        "voi_p50": float(torch.quantile(correct, 0.50)),
        "voi_p90": float(torch.quantile(correct, 0.90)),
        "response_shuffle_gain": shuffle_gain,
        "response_shuffle_gain_ci95": shuffle_ci,
        "response_shuffle_pass": shuffle_ci[0] > 0.0,
        "mc_s20_s100_spearman": spearman(s20, s100),
        "mc_nonzero_sign_agreement": float(
            ((s20 > threshold) == (s100 > threshold)).float().mean()),
        "checkpoint_restore_pass": checkpoint_restore_pass,
        "finite": finite,
    }
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    gate_values = protocol["gates"]
    gate_config = {
        "min_valid_voi_contexts": gate_values["o0_min_valid_voi_contexts"],
        "min_positive_voi_nonzero_fraction": gate_values[
            "o0_min_positive_voi_nonzero_fraction"],
        "min_candidate_argmax_diversity_fraction": gate_values[
            "o0_min_candidate_argmax_diversity_fraction"],
        "mc_s20_s100_spearman_min": gate_values["mc_s20_s100_spearman_min"],
        "mc_nonzero_sign_agreement_min": gate_values[
            "mc_nonzero_sign_agreement_min"],
    }
    gate = evaluate_o0_gate(metrics, gate_config)
    report = {
        "suite": "DRVOI-RHPPO-V7-O0",
        "protocol_version": "DRVOI-RHPPO-V7-WVR-1",
        "scientific_result": False,
        "q_ready": True,
        "candidate_samples": 20,
        "stability_samples": [20, 100],
        "metrics": metrics,
        "o0_gate": gate,
        "intrinsic_reward": 0.0,
        "long_training_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output.parent / "o0_gate.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not gate["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
