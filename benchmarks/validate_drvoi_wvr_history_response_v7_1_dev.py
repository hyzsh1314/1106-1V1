#!/usr/bin/env python3
"""Run V7.1 history-response development gates without confirmation data."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from benchmarks.drvoi.response import (
    ContinuousResponseEnsemble,
    continuous_member_log_likelihood,
)
from benchmarks.drvoi.targets import TargetStandardizer, build_delta_targets


PROTOCOL = "DRVOI-RHPPO-V7.1-history-response-1"
ALLOWED_SPLITS = {"train", "validation"}
ACTION_HEADS = {
    "alt": (0, 3), "heading": (3, 8), "speed": (8, 11), "fire": (11, 13),
}


@dataclass
class HistoryResponseData:
    context: torch.Tensor
    action: torch.Tensor
    wrong_action: torch.Tensor
    target: torch.Tensor
    episode: torch.Tensor
    opponent: torch.Tensor
    phase: torch.Tensor
    collection_seed: torch.Tensor

    def to(self, device: torch.device) -> "HistoryResponseData":
        return HistoryResponseData(**{
            key: value.to(device) for key, value in vars(self).items()})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onpolicy", type=Path, nargs="+", required=True)
    parser.add_argument("--reactive", type=Path, nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--seed", type=int, default=19185)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_development_episodes(paths: list[Path], dataset: str):
    expected_collector = file_sha256(Path(
        "benchmarks/collect_drvoi_v7_1_development_data.py"))
    expected_config = file_sha256(Path("configs/drvoi_rhppo_v7_1_wvr.json"))
    episodes = []
    seeds = set()
    ids = set()
    for path_index, path in enumerate(paths):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        report = payload["report"]
        if report.get("protocol_version") != PROTOCOL:
            raise ValueError(f"{path} has wrong protocol")
        if report.get("dataset") != dataset:
            raise ValueError(f"{path} is not {dataset}")
        if report.get("collector_sha256") != expected_collector:
            raise ValueError(f"{path} was produced by another collector")
        if report.get("protocol_config_sha256") != expected_config:
            raise ValueError(f"{path} was produced under another config")
        if report.get("contains_confirmation_test", True):
            raise ValueError(f"{path} exposes forbidden confirmation data")
        if set(report.get("available_splits", ())) - ALLOWED_SPLITS:
            raise ValueError(f"{path} includes a forbidden split")
        if int(report.get("b0_seed", -1)) != 85:
            raise ValueError(f"{path} does not use fixed B0 seed85")
        seeds.add(int(report["collection_seed"]))
        for episode in payload["episodes"]:
            if str(episode["split"]) not in ALLOWED_SPLITS:
                raise ValueError(f"forbidden episode split in {path}")
            key = (int(episode["collection_seed"]), int(episode["episode_id"]))
            if key in ids:
                raise ValueError(f"duplicate episode {key}")
            ids.add(key)
            copied = dict(episode)
            copied["episode_id"] = path_index * 10**9 + int(episode["episode_id"])
            episodes.append(copied)
    return episodes, sorted(seeds)


def flatten_history_response(episodes, split: str, branch: str,
                             delta: int = 3) -> HistoryResponseData:
    result = {key: [] for key in (
        "context", "action", "wrong_action", "target", "episode",
        "opponent", "phase", "collection_seed")}
    for episode in episodes:
        if episode["split"] != split:
            continue
        observation = episode["observation"]
        done = episode["done"].bool()
        if observation.shape[0] <= delta + 1:
            continue
        targets = build_delta_targets(
            observation, done[:-1], delta=delta, branch=branch)
        valid = targets.valid
        count = targets.delta.shape[0]
        context = torch.cat((
            episode["observation_history"][:count].flatten(1),
            episode["history_valid"][:count].float()), -1)
        result["context"].append(context[valid])
        result["action"].append(
            episode["action_history"][:count].flatten(1)[valid])
        result["wrong_action"].append(
            episode["wrong_delay_action_history"][:count].flatten(1)[valid])
        result["target"].append(targets.delta[valid])
        samples = int(valid.sum())
        result["episode"].append(torch.full(
            (samples,), int(episode["episode_id"]), dtype=torch.long))
        result["opponent"].append(torch.full(
            (samples,), int(episode["opponent"]), dtype=torch.long))
        steps = torch.arange(count)
        phase = torch.clamp(3 * steps // max(int(done.numel()), 1), max=2)
        result["phase"].append(phase[valid])
        result["collection_seed"].append(torch.full(
            (samples,), int(episode["collection_seed"]), dtype=torch.long))
    if not result["target"]:
        raise ValueError(f"no {split} samples for {branch}")
    return HistoryResponseData(**{
        key: torch.cat(values) for key, values in result.items()})


def zero_action_input_weights(model: ContinuousResponseEnsemble) -> None:
    with torch.no_grad():
        for member in model.members:
            member.network[0].weight[:, -model.action_dim:].zero_()


def member_nll(member, context, action, target):
    means, variances = member(context, action)
    return -continuous_member_log_likelihood(
        means[None], variances[None], target)[0].mean()


@torch.no_grad()
def mixture_nll(model, context, action, target, batch_size=16384):
    values = []
    boundary = []
    for start in range(0, context.shape[0], batch_size):
        selected = slice(start, start + batch_size)
        means, variances = model(context[selected], action[selected])
        likelihood = continuous_member_log_likelihood(
            means, variances, target[selected])
        values.append((
            -torch.logsumexp(likelihood, 0) + math.log(likelihood.shape[0])
        ) / target.shape[-1])
        sigma = variances.sqrt()
        boundary.append(torch.stack((
            (sigma <= 1.001e-3).float().mean(),
            (sigma >= 9.999).float().mean())))
    return torch.cat(values), torch.stack(boundary).mean(0)


def train_models(train: HistoryResponseData, validation: HistoryResponseData,
                 response_dim: int, args):
    base = ContinuousResponseEnsemble(
        input_dim=120, response_dim=response_dim, action_dim=39,
        members=args.members, hidden_size=args.hidden_size).to(train.context.device)
    zero_action_input_weights(base)
    models = {name: copy.deepcopy(base) for name in ("H1", "H0")}
    actions = {
        "H1": train.action,
        "H0": torch.zeros_like(train.action),
    }
    val_actions = {
        "H1": validation.action,
        "H0": torch.zeros_like(validation.action),
    }
    with torch.no_grad():
        outputs = [models[name](
            train.context[:64], actions[name][:64])[0] for name in models]
        initial_difference = max(float((outputs[0] - value).abs().max())
                                 for value in outputs[1:])
    optimizers = {name: torch.optim.Adam(model.parameters(), lr=3e-4)
                  for name, model in models.items()}
    unique_episode, inverse = torch.unique(train.episode, return_inverse=True)
    mask_generator = torch.Generator(device=train.context.device).manual_seed(args.seed + 17)
    masks = torch.rand(
        args.members, unique_episode.numel(), device=train.context.device,
        generator=mask_generator) < 0.8
    for member in range(args.members):
        if not bool(masks[member].any()):
            masks[member, member % unique_episode.numel()] = True
    sample_masks = masks[:, inverse]
    generator = torch.Generator(device=train.context.device).manual_seed(args.seed + 29)
    best_score = float("inf")
    best_states = None
    best_epoch = 0
    stale = 0
    gradient_max = 0.0
    gradient_fault = False
    history = []
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(
            train.context.shape[0], device=train.context.device, generator=generator)
        epoch_losses = {name: [] for name in models}
        for start in range(0, order.numel(), args.batch_size):
            indices = order[start:start + args.batch_size]
            for name, model in models.items():
                optimizers[name].zero_grad(set_to_none=True)
                losses = []
                for member_index, member in enumerate(model.members):
                    selected = indices[sample_masks[member_index, indices]]
                    if selected.numel():
                        losses.append(member_nll(
                            member, train.context[selected], actions[name][selected],
                            train.target[selected]))
                loss = torch.stack(losses).mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("non-finite response loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                gradient_max = max(gradient_max, float(norm))
                gradient_fault |= not math.isfinite(float(norm)) or float(norm) > 1000.0
                optimizers[name].step()
                epoch_losses[name].append(float(loss.detach()) / response_dim)
        scores = {}
        for name, model in models.items():
            scores[name] = float(mixture_nll(
                model, validation.context, val_actions[name], validation.target)[0].mean())
        score = sum(scores.values())
        history.append({
            "epoch": epoch,
            **{f"{name}_train_nll_per_dim": float(np.mean(epoch_losses[name]))
               for name in models},
            **{f"{name}_validation_nll_per_dim": value
               for name, value in scores.items()},
        })
        if score < best_score - 1e-5:
            best_score = score
            best_epoch = epoch
            best_states = {name: copy.deepcopy(model.state_dict())
                           for name, model in models.items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_states is None:
        raise RuntimeError("no finite response checkpoint")
    for name, state in best_states.items():
        models[name].load_state_dict(state)
    return models, {
        "initial_function_max_difference": initial_difference,
        "best_epoch": best_epoch,
        "epochs_executed": len(history),
        "raw_gradient_norm_max": gradient_max,
        "gradient_fault_over_1000": gradient_fault,
        "history": history,
    }


def permute_stratified(action, opponent, phase, seed):
    result = action.clone()
    generator = torch.Generator(device=action.device).manual_seed(seed)
    for opponent_id in range(3):
        for phase_id in range(3):
            selected = torch.nonzero(
                (opponent == opponent_id) & (phase == phase_id),
                as_tuple=False).flatten()
            if selected.numel() > 1:
                order = torch.randperm(
                    selected.numel(), device=action.device, generator=generator)
                result[selected] = action[selected[order]]
    return result


def episode_equal_mean(values, episode, selected=None):
    if selected is not None:
        values, episode = values[selected], episode[selected]
    unique = torch.unique(episode)
    return float(torch.stack([values[episode == item].mean() for item in unique]).mean())


def by_collection_seed(values, data):
    return {str(int(seed)): episode_equal_mean(
        values, data.episode, data.collection_seed == seed)
        for seed in torch.unique(data.collection_seed).cpu().tolist()}


def ablate_head(action, head):
    result = action.clone()
    start, end = ACTION_HEADS[head]
    for history_step in range(3):
        offset = history_step * 13
        result[:, offset + start:offset + end] = 0.0
    return result


def evaluate_dataset(episodes, branch, args, reactive=False):
    device = torch.device("cuda")
    train = flatten_history_response(episodes, "train", branch).to(device)
    validation = flatten_history_response(episodes, "validation", branch).to(device)
    standardizer = TargetStandardizer.fit(train.target)
    train.target = standardizer.transform(train.target)
    validation.target = standardizer.transform(validation.target)
    models, training = train_models(
        train, validation, train.target.shape[-1], args)
    zeros = torch.zeros_like(validation.action)
    h1, boundary = mixture_nll(
        models["H1"], validation.context, validation.action, validation.target)
    h0 = mixture_nll(
        models["H0"], validation.context, zeros, validation.target)[0]
    # Wrong-delay is an intervention on the fitted H1, not a separately
    # trained model that could learn to reinterpret an intentionally shifted
    # time index (or exploit older low-level controller memory).
    wrong = mixture_nll(
        models["H1"], validation.context,
        validation.wrong_action, validation.target)[0]
    shuffled = permute_stratified(
        validation.action, validation.opponent, validation.phase, args.seed + 211)
    shuffled_nll = mixture_nll(
        models["H1"], validation.context, shuffled, validation.target)[0]
    delta = h0 - h1
    wrong_gain = wrong - h1
    permutation_gain = shuffled_nll - h1
    report = {
        "branch": branch,
        "train_samples": int(train.target.shape[0]),
        "validation_samples": int(validation.target.shape[0]),
        "validation_collection_seeds": sorted(
            int(value) for value in torch.unique(validation.collection_seed)),
        "mean_delta_nll_per_dimension": episode_equal_mean(
            delta, validation.episode),
        "delta_nll_by_collection_seed": by_collection_seed(delta, validation),
        "permutation_gain_per_dimension": episode_equal_mean(
            permutation_gain, validation.episode),
        "permutation_gain_by_collection_seed": by_collection_seed(
            permutation_gain, validation),
        "correct_minus_wrong_delay_gain_per_dimension": episode_equal_mean(
            wrong_gain, validation.episode),
        "correct_minus_wrong_delay_by_collection_seed": by_collection_seed(
            wrong_gain, validation),
        "wrong_delay_control": "same_fitted_H1_with_nonoverlapping_older_actions",
        "sigma_boundary_fraction": {
            "lower": float(boundary[0]), "upper": float(boundary[1])},
        "training": training,
    }
    if reactive:
        head_diagnostics = {}
        for head in ACTION_HEADS:
            ablated_nll = mixture_nll(
                models["H1"], validation.context,
                ablate_head(validation.action, head), validation.target)[0]
            head_diagnostics[head] = episode_equal_mean(
                ablated_nll - h1, validation.episode)
        report["single_head_ablation_gain_per_dimension"] = head_diagnostics
    artifacts = {
        "models": {name: model.state_dict() for name, model in models.items()},
        "standardizer": standardizer.state_dict(),
        "input_dim": 120,
        "action_dim": 39,
        "response_dim": int(train.target.shape[-1]),
        "hidden_size": args.hidden_size,
        "members": args.members,
    }
    return report, artifacts


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    onpolicy, onpolicy_seeds = load_development_episodes(args.onpolicy, "onpolicy")
    reactive, reactive_seeds = load_development_episodes(args.reactive, "reactive")
    if onpolicy_seeds != reactive_seeds:
        raise ValueError("onpolicy/reactive collection seeds differ")
    primary, artifacts = evaluate_dataset(onpolicy, "rel", args)
    positive, reactive_artifacts = evaluate_dataset(
        reactive, "opp", args, reactive=True)

    primary_seed_values = list(primary["delta_nll_by_collection_seed"].values())
    reactive_seed_values = list(positive["delta_nll_by_collection_seed"].values())
    wrong_seed_values = list(
        positive["correct_minus_wrong_delay_by_collection_seed"].values())
    smoke = len(onpolicy_seeds) < 5
    gates = {
        "history_initialization_paired": {
            "pass": primary["training"]["initial_function_max_difference"] <= 1e-7
                    and positive["training"]["initial_function_max_difference"] <= 1e-7},
        "onpolicy_validation_positive": {
            "pass": primary["mean_delta_nll_per_dimension"] > 0
                    and sum(value > 0 for value in primary_seed_values)
                    >= max(1, len(primary_seed_values) - 1)},
        "reactive_correct_history": {
            "pass": positive["mean_delta_nll_per_dimension"] > 0
                    and positive["correct_minus_wrong_delay_gain_per_dimension"] > 0
                    and sum(value > 0 for value in reactive_seed_values)
                    >= max(1, len(reactive_seed_values) - 1)
                    and sum(value > 0 for value in wrong_seed_values)
                    >= max(1, len(wrong_seed_values) - 1)},
        "reactive_single_head_diagnostic": {
            "pass": sum(value > 0 for value in positive[
                "single_head_ablation_gain_per_dimension"].values()) >= 1},
        "numerical_health": {
            "pass": not primary["training"]["gradient_fault_over_1000"]
                    and not positive["training"]["gradient_fault_over_1000"]
                    and primary["sigma_boundary_fraction"]["lower"] == 0
                    and primary["sigma_boundary_fraction"]["upper"] == 0},
    }
    report = {
        "suite": "DRVOI-RHPPO-V7.1-P1D-history-response-development",
        "protocol_version": PROTOCOL,
        "scientific_result": False,
        "development_only": True,
        "confirmation_test_accessed": False,
        "collection_seeds": onpolicy_seeds,
        "smoke_mode": smoke,
        "primary": primary,
        "reactive_positive_control": positive,
        "gates": gates,
        "all_development_gates_pass": all(item["pass"] for item in gates.values()),
        "confirmation_data_generation_authorized": (
            not smoke and all(item["pass"] for item in gates.values())),
        "p1q_authorized": False,
        "o0_authorized": False,
        "long_training_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    torch.save({
        "primary": artifacts, "reactive": reactive_artifacts,
        "protocol_version": PROTOCOL,
        "confirmation_test_accessed": False,
    }, args.output.with_suffix(".pt"))
    print(json.dumps(report, indent=2))
    if not report["all_development_gates_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
