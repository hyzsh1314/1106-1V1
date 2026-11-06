#!/usr/bin/env python3
"""Audit V7 main effects by collection seed without changing its decision."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from benchmarks.drvoi.response import (
    ContinuousResponseEnsemble,
    continuous_member_log_likelihood,
)
from benchmarks.drvoi.targets import TargetStandardizer
from benchmarks.validate_drvoi_wvr_response_v7 import (
    flatten_response_data,
    load_episodes,
    stratified_action_permutation,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def nll_by_dimension(model, inputs, actions, targets, batch_size=16384):
    values = []
    for start in range(0, inputs.shape[0], batch_size):
        selected = slice(start, start + batch_size)
        means, variances = model(inputs[selected], actions[selected])
        residual = targets[selected].unsqueeze(0) - means
        log_likelihood = -0.5 * (
            residual.square() / variances + variances.log()
            + math.log(2.0 * math.pi))
        values.append(
            -torch.logsumexp(log_likelihood, 0)
            + math.log(log_likelihood.shape[0]))
    return torch.cat(values)


def episode_equal_mean(values, episode):
    return torch.stack([
        values[episode == episode_id].mean(0)
        for episode_id in torch.unique(episode)]).mean(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path,
                        default=Path("runs/drvoi_rhppo_v7"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/drvoi_rhppo_v7"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the V7 seed audit")
    device = torch.device("cuda")
    artifact = torch.load(
        args.run_root / "p1-response-formal/validation.pt",
        map_location=device, weights_only=False)
    model_kwargs = {
        "input_dim": int(artifact["input_dim"]),
        "response_dim": int(artifact["response_dim"]),
        "members": int(artifact["members"]),
        "hidden_size": int(artifact["hidden_size"]),
        "sigma_min": float(artifact["sigma_min"]),
        "sigma_max": float(artifact["sigma_max"]),
    }
    conditioned = ContinuousResponseEnsemble(**model_kwargs).to(device).eval()
    no_action = ContinuousResponseEnsemble(**model_kwargs).to(device).eval()
    conditioned.load_state_dict(artifact["conditioned"])
    no_action.load_state_dict(artifact["no_action"])
    conditioned.requires_grad_(False)
    no_action.requires_grad_(False)
    standardizer = TargetStandardizer.from_state_dict(artifact["standardizer"])
    standardizer.mean = standardizer.mean.to(device)
    standardizer.scale = standardizer.scale.to(device)

    seed_rows = []
    dimension_rows = []
    opponent_rows = []
    for seed in (6185, 7185, 8185):
        path = args.run_root / f"p1-response-formal/data/onpolicy-seed{seed}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        b0_seed = int(payload["report"]["b0_seed"])
        episodes = load_episodes([path], "onpolicy")
        data = flatten_response_data(episodes, "test", 3, "rel").to(device)
        data.target = standardizer.transform(data.target)
        inputs = data.belief
        zeros = torch.zeros_like(data.action)
        conditioned_nll = nll_by_dimension(
            conditioned, inputs, data.action, data.target)
        no_action_nll = nll_by_dimension(
            no_action, inputs, zeros, data.target)
        delta = no_action_nll - conditioned_nll
        permuted_action = stratified_action_permutation(
            data.action, data.opponent, data.phase, seed + 401)
        permuted_nll = nll_by_dimension(
            conditioned, inputs, permuted_action, data.target)
        permutation = permuted_nll - conditioned_nll
        episode_delta = episode_equal_mean(delta, data.episode)
        episode_permutation = episode_equal_mean(permutation, data.episode)
        seed_rows.append({
            "collection_seed": seed, "b0_seed": b0_seed,
            "test_episodes": int(torch.unique(data.episode).numel()),
            "test_samples": int(data.target.shape[0]),
            "delta_nll_per_dim": float(episode_delta.mean()),
            "permutation_gain_per_dim": float(episode_permutation.mean()),
        })
        for dimension in range(data.target.shape[-1]):
            dimension_rows.append({
                "collection_seed": seed, "b0_seed": b0_seed,
                "dimension": dimension,
                "delta_nll": float(episode_delta[dimension]),
                "permutation_gain": float(episode_permutation[dimension]),
            })
        for opponent in range(3):
            selected = data.opponent == opponent
            opponent_rows.append({
                "collection_seed": seed, "b0_seed": b0_seed,
                "opponent_id": opponent,
                "episodes": int(torch.unique(data.episode[selected]).numel()),
                "delta_nll_per_dim": float(episode_equal_mean(
                    delta[selected], data.episode[selected]).mean()),
                "permutation_gain_per_dim": float(episode_equal_mean(
                    permutation[selected], data.episode[selected]).mean()),
            })
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "main_effect_by_collection_seed.csv", seed_rows)
    write_csv(args.output / "main_effect_by_dimension_and_seed.csv", dimension_rows)
    write_csv(args.output / "main_effect_by_opponent_and_seed.csv", opponent_rows)
    summary = {
        "schema": "drvoi-rhppo-v7-seed-audit-1",
        "historical_test_audit_only": True,
        "not_for_v7_1_selection": True,
        "rows": seed_rows,
    }
    (args.output / "seed_effect_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
