#!/usr/bin/env python3
"""Run the once-only V7.1 five-fold confirmation response gates."""

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

from benchmarks.drvoi.calibration import (
    calibrated_member_parameters,
    fit_nll_variance_scale,
    interval_coverage,
    mixture_moments,
)
from benchmarks.drvoi.response import (
    ContinuousResponseEnsemble,
    continuous_member_log_likelihood,
)
from benchmarks.drvoi.targets import TargetStandardizer, build_delta_targets
from benchmarks.export_drvoi_v7_1_development_audit import t_interval


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

    def select(self, selected: torch.Tensor) -> "HistoryResponseData":
        return HistoryResponseData(**{
            key: value[selected] for key, value in vars(self).items()})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-onpolicy", type=Path, nargs="+", required=True)
    parser.add_argument("--development-reactive", type=Path, nargs="+", required=True)
    parser.add_argument("--confirmation-onpolicy", type=Path, nargs="+", required=True)
    parser.add_argument("--confirmation-reactive", type=Path, nargs="+", required=True)
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


def load_confirmation_episodes(paths: list[Path], dataset: str):
    expected_collector = file_sha256(Path(
        "benchmarks/collect_drvoi_v7_1_confirmation_data.py"))
    expected_config = file_sha256(Path("configs/drvoi_rhppo_v7_1_wvr.json"))
    expected_authorization = file_sha256(Path(
        "results/drvoi_rhppo_v7_1/w2/development_audit.json"))
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
            raise ValueError(f"{path} was produced by another confirmation collector")
        if report.get("protocol_config_sha256") != expected_config:
            raise ValueError(f"{path} was produced under another config")
        if report.get("development_authorization_sha256") != expected_authorization:
            raise ValueError(f"{path} has a different development authorization")
        if not report.get("contains_confirmation_test"):
            raise ValueError(f"{path} does not contain confirmation test")
        if set(report.get("available_splits", ())) != {"calibration", "test"}:
            raise ValueError(f"{path} has wrong confirmation splits")
        if int(report.get("b0_seed", -1)) != 85:
            raise ValueError(f"{path} does not use fixed B0 seed85")
        seeds.add(int(report["collection_seed"]))
        for episode in payload["episodes"]:
            if str(episode["split"]) not in {"calibration", "test"}:
                raise ValueError(f"forbidden episode split in {path}")
            key = (int(episode["collection_seed"]), int(episode["episode_id"]),
                   str(episode["split"]))
            if key in ids:
                raise ValueError(f"duplicate confirmation episode {key}")
            ids.add(key)
            copied = dict(episode)
            copied["episode_id"] = (path_index + 10) * 10**9 + int(episode["episode_id"])
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


@torch.no_grad()
def calibrate_and_evaluate(model, calibration, test):
    cal_means, cal_variances = model(calibration.context, calibration.action)
    variance_scale = fit_nll_variance_scale(
        cal_means, cal_variances, calibration.target)
    means, variances = model(test.context, test.action)
    means, variances = calibrated_member_parameters(
        means, variances, variance_scale)
    mean, variance = mixture_moments(means, variances)
    residual = test.target - mean
    log_likelihood_dim = -0.5 * (
        (test.target[None] - means).square() / variances
        + variances.log() + math.log(2.0 * math.pi))
    nll_dimension_samples = (
        -torch.logsumexp(log_likelihood_dim, 0) + math.log(means.shape[0]))
    pit = (0.5 * (1.0 + torch.erf(
        (test.target[None] - means) / (2.0 * variances).sqrt()))).mean(0)
    return {
        "variance_scale": variance_scale,
        "mean": mean,
        "variance": variance,
        "target": test.target,
        "residual": residual,
        "nll_dimension_samples": nll_dimension_samples,
        "pit": pit,
        "opponent": test.opponent,
        "phase": test.phase,
    }


def aggregate_calibration(parts):
    merged = {key: torch.cat([part[key] for part in parts]) for key in (
        "mean", "variance", "target", "residual", "nll_dimension_samples",
        "pit", "opponent", "phase")}
    standardized = merged["residual"] / merged["variance"].sqrt()
    coverage50 = interval_coverage(
        merged["mean"], merged["variance"], merged["target"], 0.6744897501960817)
    coverage90 = interval_coverage(
        merged["mean"], merged["variance"], merged["target"], 1.6448536269514722)
    coverage95 = interval_coverage(
        merged["mean"], merged["variance"], merged["target"], 1.959963984540054)
    strata = {}
    for opponent in range(3):
        strata[str(opponent)] = {}
        for phase in range(3):
            selected = ((merged["opponent"] == opponent)
                        & (merged["phase"] == phase))
            strata[str(opponent)][str(phase)] = (
                float((standardized[selected].abs() <= 1.6448536269514722)
                      .float().mean()) if bool(selected.any()) else None)
    pit_histograms = []
    pit_ks = []
    for dimension in range(merged["target"].shape[-1]):
        values = merged["pit"][:, dimension]
        pit_histograms.append(torch.histc(
            values, bins=10, min=0, max=1).long().cpu().tolist())
        ordered = values.sort().values
        empirical = torch.arange(
            1, ordered.numel() + 1, device=ordered.device) / ordered.numel()
        pit_ks.append(float((empirical - ordered).abs().max()))
    overall = float(coverage90.mean())
    valid_strata = [value for phases in strata.values()
                    for value in phases.values() if value is not None]
    passed = (0.85 <= overall <= 0.95
              and bool(((coverage90 >= 0.75) & (coverage90 <= 0.98)).all())
              and min(valid_strata) >= 0.75)
    return {
        "variance_scale_by_fold": [part["variance_scale"].cpu().tolist()
                                   for part in parts],
        "nll_per_dimension": merged["nll_dimension_samples"].mean(0).cpu().tolist(),
        "rmse_per_dimension": merged["residual"].square().mean(0).sqrt().cpu().tolist(),
        "coverage_50_by_dimension": coverage50.cpu().tolist(),
        "coverage_90_by_dimension": coverage90.cpu().tolist(),
        "coverage_95_by_dimension": coverage95.cpu().tolist(),
        "coverage_90_overall": overall,
        "coverage_90_by_opponent_phase": strata,
        "standardized_residual_mean": standardized.mean(0).cpu().tolist(),
        "standardized_residual_std": standardized.std(0, unbiased=False).cpu().tolist(),
        "pit_histogram_10_by_dimension": pit_histograms,
        "pit_ks_by_dimension": pit_ks,
        "sigma_boundary_fraction": {
            "lower": float((merged["variance"].sqrt() <= 1.001e-3).float().mean()),
            "upper": float((merged["variance"].sqrt() >= 9.999).float().mean()),
        },
        "pass": passed,
    }


def confirmation_fold(all_data, held_seed, branch, args, fold_index,
                      with_calibration=False):
    train = all_data["train"].select(
        all_data["train"].collection_seed != held_seed)
    validation = all_data["validation"].select(
        all_data["validation"].collection_seed != held_seed)
    calibration = all_data["calibration"].select(
        all_data["calibration"].collection_seed != held_seed)
    test = all_data["test"].select(
        all_data["test"].collection_seed == held_seed)
    standardizer = TargetStandardizer.fit(train.target)
    for value in (train, validation, calibration, test):
        value.target = standardizer.transform(value.target)
    fold_args = copy.copy(args)
    fold_args.seed = args.seed + fold_index * 1009
    models, training = train_models(
        train, validation, train.target.shape[-1], fold_args)
    h1, boundary = mixture_nll(
        models["H1"], test.context, test.action, test.target)
    h0 = mixture_nll(
        models["H0"], test.context, torch.zeros_like(test.action), test.target)[0]
    wrong = mixture_nll(
        models["H1"], test.context, test.wrong_action, test.target)[0]
    shuffled = permute_stratified(
        test.action, test.opponent, test.phase, fold_args.seed + 211)
    shuffled_nll = mixture_nll(
        models["H1"], test.context, shuffled, test.target)[0]
    result = {
        "held_collection_seed": held_seed,
        "branch": branch,
        "train_collection_seeds": sorted(
            int(seed) for seed in torch.unique(train.collection_seed)),
        "test_collection_seeds": sorted(
            int(seed) for seed in torch.unique(test.collection_seed)),
        "train_samples": int(train.target.shape[0]),
        "test_samples": int(test.target.shape[0]),
        "test_episodes": int(torch.unique(test.episode).numel()),
        "delta_nll_per_dimension": episode_equal_mean(h0 - h1, test.episode),
        "permutation_gain_per_dimension": episode_equal_mean(
            shuffled_nll - h1, test.episode),
        "correct_vs_wrong_delay_per_dimension": episode_equal_mean(
            wrong - h1, test.episode),
        "sigma_boundary_fraction": {
            "lower": float(boundary[0]), "upper": float(boundary[1])},
        "training": training,
    }
    calibration_part = None
    if with_calibration:
        calibration_part = calibrate_and_evaluate(
            models["H1"], calibration, test)
    artifacts = {
        "held_collection_seed": held_seed,
        "models": {name: model.state_dict() for name, model in models.items()},
        "standardizer": standardizer.state_dict(),
        "variance_scale": (None if calibration_part is None
                           else calibration_part["variance_scale"]),
    }
    return result, calibration_part, artifacts


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dev_onpolicy, dev_seeds = load_development_episodes(
        args.development_onpolicy, "onpolicy")
    dev_reactive, dev_reactive_seeds = load_development_episodes(
        args.development_reactive, "reactive")
    conf_onpolicy, conf_seeds = load_confirmation_episodes(
        args.confirmation_onpolicy, "onpolicy")
    conf_reactive, conf_reactive_seeds = load_confirmation_episodes(
        args.confirmation_reactive, "reactive")
    if not (dev_seeds == dev_reactive_seeds == conf_seeds == conf_reactive_seeds):
        raise ValueError("development/confirmation collection seeds differ")
    if len(dev_seeds) != 5:
        raise ValueError("confirmation protocol requires exactly five seeds")
    device = torch.device("cuda")
    onpolicy_episodes = dev_onpolicy + conf_onpolicy
    reactive_episodes = dev_reactive + conf_reactive
    onpolicy = {split: flatten_history_response(
        onpolicy_episodes, split, "rel").to(device)
        for split in ("train", "validation", "calibration", "test")}
    reactive = {split: flatten_history_response(
        reactive_episodes, split, "opp").to(device)
        for split in ("train", "validation", "calibration", "test")}

    primary_folds = []
    reactive_folds = []
    calibration_parts = []
    artifacts = {"primary": [], "reactive": []}
    for fold_index, held_seed in enumerate(dev_seeds):
        primary, calibration, primary_artifact = confirmation_fold(
            onpolicy, held_seed, "rel", args, fold_index,
            with_calibration=True)
        positive, _, reactive_artifact = confirmation_fold(
            reactive, held_seed, "opp", args, fold_index + 10)
        primary_folds.append(primary)
        reactive_folds.append(positive)
        calibration_parts.append(calibration)
        artifacts["primary"].append(primary_artifact)
        artifacts["reactive"].append(reactive_artifact)

    primary_values = [fold["delta_nll_per_dimension"] for fold in primary_folds]
    permutation_values = [fold["permutation_gain_per_dimension"]
                          for fold in primary_folds]
    reactive_values = [fold["delta_nll_per_dimension"] for fold in reactive_folds]
    wrong_values = [fold["correct_vs_wrong_delay_per_dimension"]
                    for fold in reactive_folds]
    primary_mean, primary_ci = t_interval(primary_values)
    permutation_mean, permutation_ci = t_interval(permutation_values)
    reactive_mean, reactive_ci = t_interval(reactive_values)
    wrong_mean, wrong_ci = t_interval(wrong_values)
    calibration = aggregate_calibration(calibration_parts)
    numerical_pass = all(
        not fold["training"]["gradient_fault_over_1000"]
        and fold["sigma_boundary_fraction"]["lower"] == 0
        and fold["sigma_boundary_fraction"]["upper"] == 0
        for fold in primary_folds + reactive_folds)
    gates = {
        "G0_numerical_health": {"pass": numerical_pass},
        "G1_primary_history_response": {"pass": (
            primary_mean >= 0.05 and primary_ci[0] > 0
            and sum(value > 0 for value in primary_values) >= 4)},
        "G1_history_permutation": {"pass": (
            permutation_mean > 0 and permutation_ci[0] > 0)},
        "G1_reactive_positive_control": {"pass": (
            reactive_mean > 0 and reactive_ci[0] > 0
            and wrong_mean > 0 and wrong_ci[0] > 0)},
        "G2_probability_calibration": {"pass": calibration["pass"]},
    }
    all_pass = all(item["pass"] for item in gates.values())
    report = {
        "suite": "DRVOI-RHPPO-V7.1-five-fold-confirmation",
        "protocol_version": PROTOCOL,
        "scientific_result": True,
        "confirmation_test_accessed": True,
        "test_access_count": 1,
        "outer_protocol": "leave-one-collection-seed-out",
        "collection_seeds": dev_seeds,
        "primary": {
            "branch": "rel", "delta": 3, "dataset": "onpolicy",
            "folds": primary_folds,
            "mean_delta_nll_per_dimension": primary_mean,
            "delta_nll_small_sample_t_ci95": primary_ci,
            "positive_folds": sum(value > 0 for value in primary_values),
            "mean_permutation_gain_per_dimension": permutation_mean,
            "permutation_small_sample_t_ci95": permutation_ci,
        },
        "reactive_positive_control": {
            "branch": "opp", "delta": 3, "folds": reactive_folds,
            "mean_delta_nll_per_dimension": reactive_mean,
            "delta_nll_small_sample_t_ci95": reactive_ci,
            "mean_correct_vs_wrong_delay_per_dimension": wrong_mean,
            "wrong_delay_small_sample_t_ci95": wrong_ci,
        },
        "calibration": calibration,
        "gates": gates,
        "all_confirmation_gates_pass": all_pass,
        "g3_b0_robustness_required": all_pass,
        "p1q_authorized": False,
        "o0_authorized": False,
        "long_training_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    torch.save({
        **artifacts, "protocol_version": PROTOCOL,
        "confirmation_test_accessed": True,
    }, args.output.with_suffix(".pt"))
    print(json.dumps(report, indent=2))
    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
