#!/usr/bin/env python3
"""Validate structured HOT Q controls before any WVR Q test is created."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.drvoi.core import action_ids_to_multi_one_hot
from benchmarks.drvoi.q_model import ActionEmbeddingQ
from benchmarks.drvoi_v7_1.q_controls import (
    bootstrap_context_mean,
    irrelevant_perturbation_report,
    normalized_argmax_regret,
    pairwise_ranking_accuracy,
    structured_flip_report,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(
        "configs/drvoi_rhppo_v7_1_p1q.json"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=40185)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def make_contexts(count, rng):
    features, truth, group = [], [], []
    for context_id in range(count):
        prior = rng.dirichlet(np.full(3, 2.0))
        reliability = rng.uniform(0.36, 0.97, 3)
        reward = np.eye(3) * rng.uniform(0.8, 1.2, 3)[None, :]
        for information_action in range(3):
            likelihood = np.full((3, 3), (1 - reliability[information_action]) / 2)
            np.fill_diagonal(likelihood, reliability[information_action])
            predictive = prior @ likelihood
            posterior = prior[:, None] * likelihood / np.clip(
                predictive[None], 1e-12, None)
            for response in range(3):
                for nuisance in (-1.0, 1.0):
                    feature = np.concatenate((
                        prior, reliability, reward.reshape(-1),
                        np.eye(3)[information_action], np.eye(3)[response],
                        [nuisance]))
                    features.append(feature)
                    truth.append(posterior[:, response] @ reward)
                    group.append((context_id, information_action, response,
                                  nuisance))
    return (torch.tensor(np.asarray(features), dtype=torch.float32),
            torch.tensor(np.asarray(truth), dtype=torch.float32), group)


def expanded_supervision(context, truth, device):
    action = action_ids_to_multi_one_hot(torch.arange(3)).to(device)
    return (
        context[:, None].expand(-1, 3, -1).reshape(-1, context.shape[-1]),
        action[None].expand(context.shape[0], -1, -1).reshape(-1, 13),
        truth.reshape(-1),
    )


def train_model(train_context, train_truth, validation_context,
                validation_truth, args, device):
    model = ActionEmbeddingQ(train_context.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    train_x, train_a, train_y = expanded_supervision(
        train_context.to(device), train_truth.to(device), device)
    val_x, val_a, val_y = expanded_supervision(
        validation_context.to(device), validation_truth.to(device), device)
    generator = torch.Generator(device=device).manual_seed(args.seed + 17)
    best, best_state, stale, history = float("inf"), None, 0, []
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(train_y.numel(), device=device, generator=generator)
        losses = []
        for start in range(0, order.numel(), args.batch_size):
            selected = order[start:start + args.batch_size]
            prediction = model(train_x[selected], train_a[selected])
            loss = torch.nn.functional.smooth_l1_loss(prediction, train_y[selected])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            validation_loss = float(torch.nn.functional.mse_loss(
                model(val_x, val_a), val_y))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "validation_mse": validation_loss})
        if validation_loss < best - 1e-7:
            best, best_state, stale = validation_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predicted_values(model, context, device):
    actions = action_ids_to_multi_one_hot(torch.arange(3, device=device))
    x = context.to(device)
    return model(
        x[:, None].expand(-1, 3, -1).reshape(-1, x.shape[-1]),
        actions[None].expand(x.shape[0], -1, -1).reshape(-1, 13)).reshape(-1, 3)


def select_controls(model, prediction, truth, groups, contexts, device,
                    limit=128):
    flip_indices = []
    irrelevant_indices = []
    by_base = {}
    for index, (context_id, information_action, response, nuisance) in enumerate(groups):
        by_base.setdefault((context_id, information_action), []).append(index)
        margin = float(torch.topk(truth[index], 2).values.diff().abs())
        if margin >= 0.20 and nuisance < 0:
            irrelevant_indices.append(index)
    flip_pairs = []
    for indices in by_base.values():
        for left in indices:
            for right in indices:
                if (left < right
                        and groups[left][2] != groups[right][2]
                        and groups[left][3] == groups[right][3]
                        and truth[left].argmax() != truth[right].argmax()):
                    if (torch.topk(truth[left], 2).values.diff().abs() >= 0.20
                            and torch.topk(truth[right], 2).values.diff().abs() >= 0.20):
                        flip_pairs.append((left, right))
    flip_pairs = flip_pairs[:limit]
    irrelevant_indices = irrelevant_indices[:limit]
    before = torch.stack([prediction[left] for left, _ in flip_pairs])
    after = torch.stack([prediction[right] for _, right in flip_pairs])
    expected_before = torch.stack([truth[left].argmax() for left, _ in flip_pairs]).to(device)
    expected_after = torch.stack([truth[right].argmax() for _, right in flip_pairs]).to(device)
    flip = structured_flip_report(before, after, expected_before, expected_after)
    base_context = contexts[irrelevant_indices].clone()
    negative = base_context.clone(); negative[:, -1] = -1.0
    positive = base_context.clone(); positive[:, -1] = 1.0
    negative_q = predicted_values(model, negative, device)
    positive_q = predicted_values(model, positive, device)
    expected = truth[irrelevant_indices].argmax(-1).to(device)
    irrelevant = irrelevant_perturbation_report(
        negative_q, positive_q, expected)
    return flip, irrelevant


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    auth = json.loads(Path(protocol["authorization"]["source"]).read_text(
        encoding="utf-8"))
    if not auth.get("p1q_authorized"):
        raise PermissionError("G3 did not authorize P1Q")
    cfg = protocol["hot_development"]
    rng = np.random.default_rng(args.seed)
    train_x, train_y, _ = make_contexts(cfg["train_contexts"], rng)
    val_x, val_y, _ = make_contexts(cfg["validation_contexts"], rng)
    test_x, test_y, groups = make_contexts(cfg["test_contexts"], rng)
    device = torch.device("cuda")
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    model, history = train_model(train_x, train_y, val_x, val_y, args, device)
    prediction = predicted_values(model, test_x, device)
    truth = test_y.to(device)
    pair_by_context = pairwise_ranking_accuracy(prediction, truth, minimum_gap=0.02)
    pair_mean, pair_ci = bootstrap_context_mean(pair_by_context, args.seed + 101)
    regret = normalized_argmax_regret(prediction, truth)
    random_prediction = torch.rand(
        prediction.shape, device=device,
        generator=torch.Generator(device=device).manual_seed(args.seed + 103))
    random_regret = normalized_argmax_regret(random_prediction, truth)
    flip, irrelevant = select_controls(
        model, prediction, truth, groups, test_x, device)
    gates = {
        "HOT_structured_flip": {**flip, "pass": flip["all_pass"]},
        "HOT_decision_irrelevant": {**irrelevant, "pass": irrelevant["all_pass"]},
        "HOT_pairwise_ranking": {
            "mean": pair_mean, "ci95": pair_ci,
            "pass": pair_mean >= 0.60 and pair_ci[0] > 0.50},
        "HOT_normalized_argmax_regret": {
            "mean": float(regret.mean()),
            "random_mean": float(random_regret.mean()),
            "pass": float(regret.mean()) <= 0.10
                    and float(regret.mean()) < float(random_regret.mean())},
    }
    report = {
        "suite": "DRVOI-RHPPO-V7.1-P1Q-HOT-development",
        "protocol_version": protocol["protocol_version"],
        "scientific_result": False,
        "wvr_q_test_accessed": False,
        "training_history": history,
        "gates": gates,
        "all_hot_gates_pass": all(item["pass"] for item in gates.values()),
        "wvr_q_test_generation_authorized": all(
            item["pass"] for item in gates.values()),
        "o0_authorized": False,
        "long_training_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    torch.save({"model": model.state_dict(), "context_dim": train_x.shape[-1]},
               args.output.with_suffix(".pt"))
    print(json.dumps(report, indent=2))
    if not report["all_hot_gates_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
