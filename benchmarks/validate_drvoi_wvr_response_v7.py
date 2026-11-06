#!/usr/bin/env python3
"""Run the pre-registered V7 P1R/P1C response identification gates."""

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

from benchmarks.drvoi import multi_discrete_to_one_hot
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


@dataclass
class FlatResponseData:
    observation: torch.Tensor
    belief: torch.Tensor
    action: torch.Tensor
    target: torch.Tensor
    episode: torch.Tensor
    opponent: torch.Tensor
    phase: torch.Tensor

    def to(self, device: torch.device) -> "FlatResponseData":
        return FlatResponseData(**{
            key: value.to(device) for key, value in vars(self).items()})

    def select(self, indices: torch.Tensor) -> "FlatResponseData":
        return FlatResponseData(**{
            key: value[indices] for key, value in vars(self).items()})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onpolicy", type=Path, nargs="+", required=True)
    parser.add_argument("--cover", type=Path, nargs="+", required=True)
    parser.add_argument("--reactive", type=Path, nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=9185)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_episodes(paths: list[Path], expected_dataset: str) -> list[dict[str, object]]:
    episodes: list[dict[str, object]] = []
    seen_ids: set[tuple[int, int]] = set()
    expected_collector_hash = hashlib.sha256(Path(
        "benchmarks/collect_drvoi_v7_response_data.py").read_bytes()).hexdigest()
    expected_protocol_hash = hashlib.sha256(Path(
        "configs/drvoi_rhppo_v7_wvr.json").read_bytes()).hexdigest()
    for path_index, path in enumerate(paths):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["report"]["dataset"] != expected_dataset:
            raise ValueError(f"{path} is not {expected_dataset}")
        if not payload["report"]["complete_episode_invariants"]["all_end_done"]:
            raise ValueError(f"{path} contains incomplete episodes")
        if payload["report"].get("collector_sha256") != expected_collector_hash:
            raise ValueError(f"{path} was produced by a different collector")
        if payload["report"].get("protocol_config_sha256") != expected_protocol_hash:
            raise ValueError(f"{path} was produced under a different protocol config")
        for episode in payload["episodes"]:
            key = (path_index, int(episode["episode_id"]))
            if key in seen_ids:
                raise ValueError(f"duplicate episode in {path}")
            seen_ids.add(key)
            copied = dict(episode)
            # Collection seeds already make ids globally unique in the frozen
            # protocol; this extra prefix guards custom smoke datasets.
            copied["episode_id"] = path_index * 10**9 + int(episode["episode_id"])
            episodes.append(copied)
    return episodes


def flatten_response_data(episodes: list[dict[str, object]], split: str,
                          delta: int, branch: str) -> FlatResponseData:
    result = {key: [] for key in (
        "observation", "belief", "action", "target", "episode",
        "opponent", "phase")}
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
        steps = torch.arange(count)
        result["observation"].append(observation[:count][valid])
        result["belief"].append(episode["belief"][:count][valid])
        result["action"].append(multi_discrete_to_one_hot(
            episode["action"][:count])[valid])
        result["target"].append(targets.delta[valid])
        samples = int(valid.sum())
        result["episode"].append(torch.full(
            (samples,), int(episode["episode_id"]), dtype=torch.long))
        result["opponent"].append(torch.full(
            (samples,), int(episode["opponent"]), dtype=torch.long))
        phase = torch.clamp(3 * steps // max(int(done.numel()), 1), max=2)
        result["phase"].append(phase[valid])
    if not result["target"]:
        raise ValueError(f"no {split} samples for delta={delta}, branch={branch}")
    return FlatResponseData(**{
        key: torch.cat(values) for key, values in result.items()})


def zero_action_input_weights(model: ContinuousResponseEnsemble) -> None:
    with torch.no_grad():
        for member in model.members:
            member.network[0].weight[:, -13:].zero_()


def _member_nll(member, inputs, actions, target):
    means, variances = member(inputs, actions)
    return -continuous_member_log_likelihood(
        means[None], variances[None], target)[0].mean()


@torch.no_grad()
def mixture_nll_batched(model, inputs, actions, target,
                        batch_size=16384) -> torch.Tensor:
    values = []
    for start in range(0, inputs.shape[0], batch_size):
        selected = slice(start, start + batch_size)
        means, variances = model(inputs[selected], actions[selected])
        likelihood = continuous_member_log_likelihood(
            means, variances, target[selected])
        values.append((
            -torch.logsumexp(likelihood, 0) + math.log(likelihood.shape[0])
        ) / target.shape[-1])
    return torch.cat(values)


def train_paired_ensembles(train: FlatResponseData, validation: FlatResponseData,
                           input_kind: str, response_dim: int, args):
    inputs = train.belief if input_kind == "belief" else train.observation
    val_inputs = validation.belief if input_kind == "belief" else validation.observation
    conditioned = ContinuousResponseEnsemble(
        inputs.shape[-1], response_dim, members=5,
        hidden_size=args.hidden_size).to(inputs.device)
    zero_action_input_weights(conditioned)
    no_action = copy.deepcopy(conditioned)
    zeros = torch.zeros_like(train.action)
    val_zeros = torch.zeros_like(validation.action)
    with torch.no_grad():
        initial_difference = max(float((left - right).abs().max()) for left, right in zip(
            conditioned(inputs[:64], train.action[:64]),
            no_action(inputs[:64], zeros[:64])))
    optimizers = (
        torch.optim.Adam(conditioned.parameters(), lr=3.0e-4),
        torch.optim.Adam(no_action.parameters(), lr=3.0e-4),
    )
    unique_episodes, inverse = torch.unique(train.episode, return_inverse=True)
    mask_generator = torch.Generator(device=inputs.device).manual_seed(args.seed + 17)
    masks = torch.rand(
        5, unique_episodes.numel(), device=inputs.device,
        generator=mask_generator) < 0.8
    for member in range(5):
        if not bool(masks[member].any()):
            masks[member, member % unique_episodes.numel()] = True
    sample_masks = masks[:, inverse]
    generator = torch.Generator(device=inputs.device).manual_seed(args.seed + 29)
    best_score = float("inf")
    best_epoch = 0
    best_states = None
    stale = 0
    raw_gradient_max = 0.0
    gradient_fault = False
    history = []
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(inputs.shape[0], device=inputs.device, generator=generator)
        losses = [[], []]
        for start in range(0, order.numel(), args.batch_size):
            indices = order[start:start + args.batch_size]
            for model_index, (model, optimizer, action) in enumerate(zip(
                    (conditioned, no_action), optimizers, (train.action, zeros))):
                optimizer.zero_grad(set_to_none=True)
                member_losses = []
                for member_index, member in enumerate(model.members):
                    selected = indices[sample_masks[member_index, indices]]
                    if selected.numel():
                        member_losses.append(_member_nll(
                            member, inputs[selected], action[selected],
                            train.target[selected]))
                loss = torch.stack(member_losses).mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("non-finite response loss")
                loss.backward()
                raw_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                raw_gradient_max = max(raw_gradient_max, float(raw_norm))
                if not math.isfinite(float(raw_norm)):
                    raise FloatingPointError("non-finite response gradient")
                if float(raw_norm) > 1000.0:
                    # Offline fitting continues with the pre-registered clip
                    # so a complete diagnostic report can be written, but the
                    # numerical-health gate remains irreversibly failed.
                    gradient_fault = True
                optimizer.step()
                losses[model_index].append(float(loss.detach()))
        conditioned_val = float(mixture_nll_batched(
            conditioned, val_inputs, validation.action, validation.target).mean())
        no_action_val = float(mixture_nll_batched(
            no_action, val_inputs, val_zeros, validation.target).mean())
        score = conditioned_val + no_action_val
        history.append({
            "epoch": epoch,
            "conditioned_train_nll": float(np.mean(losses[0])) / response_dim,
            "no_action_train_nll": float(np.mean(losses[1])) / response_dim,
            "conditioned_validation_nll_per_dim": conditioned_val,
            "no_action_validation_nll_per_dim": no_action_val,
        })
        if score < best_score - 1.0e-5:
            best_score = score
            best_epoch = epoch
            best_states = (
                copy.deepcopy(conditioned.state_dict()),
                copy.deepcopy(no_action.state_dict()),
            )
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    conditioned.load_state_dict(best_states[0])
    no_action.load_state_dict(best_states[1])
    return conditioned, no_action, {
        "initial_function_max_difference": initial_difference,
        "best_epoch": best_epoch,
        "epochs_executed": len(history),
        "raw_gradient_norm_max": raw_gradient_max,
        "gradient_fault_over_1000": gradient_fault,
        "history": history,
    }


def stratified_action_permutation(action, opponent, phase, seed):
    generator = torch.Generator(device=action.device).manual_seed(seed)
    result = action.clone()
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


def stratified_episode_bootstrap(values, episode, opponent, seed, samples=2000):
    unique = torch.unique(episode)
    episode_values = torch.stack([values[episode == item].mean() for item in unique])
    episode_opponents = torch.stack([
        opponent[episode == item][0] for item in unique])
    values_np = episode_values.detach().cpu().numpy()
    opponents_np = episode_opponents.detach().cpu().numpy()
    means = [values_np[opponents_np == index].mean() for index in range(3)]
    macro = float(np.mean(means))
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(samples):
        strata = []
        for index in range(3):
            selected = values_np[opponents_np == index]
            strata.append(rng.choice(selected, len(selected), replace=True).mean())
        bootstrap.append(np.mean(strata))
    return macro, np.quantile(bootstrap, [0.025, 0.975]).tolist(), {
        str(index): float(means[index]) for index in range(3)}


@torch.no_grad()
def calibration_report(model, calibration, test, input_kind):
    calibration_input = (
        calibration.belief if input_kind == "belief" else calibration.observation)
    test_input = test.belief if input_kind == "belief" else test.observation
    cal_means, cal_variances = model(calibration_input, calibration.action)
    variance_scale = fit_nll_variance_scale(
        cal_means, cal_variances, calibration.target)
    means, variances = model(test_input, test.action)
    means, variances = calibrated_member_parameters(
        means, variances, variance_scale)
    mean, variance = mixture_moments(means, variances)
    residual = test.target - mean
    log_likelihood_dim = -0.5 * (
        (test.target[None] - means).square() / variances
        + variances.log() + math.log(2.0 * math.pi))
    nll_dimension = (-torch.logsumexp(log_likelihood_dim, 0)
                     + math.log(means.shape[0])).mean(0)
    coverage = {
        "50": interval_coverage(mean, variance, test.target, 0.6744897501960817),
        "90": interval_coverage(mean, variance, test.target, 1.6448536269514722),
        "95": interval_coverage(mean, variance, test.target, 1.959963984540054),
    }
    standardized = residual / variance.sqrt()
    pit = (0.5 * (1.0 + torch.erf(
        (test.target[None] - means) / (2.0 * variances).sqrt()))).mean(0)
    pit_histogram = []
    pit_ks = []
    for dimension in range(test.target.shape[-1]):
        values = pit[:, dimension]
        pit_histogram.append(torch.histc(
            values, bins=10, min=0.0, max=1.0).long().cpu().tolist())
        ordered = values.sort().values
        empirical = torch.arange(
            1, ordered.numel() + 1, device=ordered.device) / ordered.numel()
        pit_ks.append(float(torch.max(torch.abs(empirical - ordered))))
    stratified_90 = {}
    for opponent in range(3):
        stratified_90[str(opponent)] = {}
        for phase in range(3):
            selected = (test.opponent == opponent) & (test.phase == phase)
            stratified_90[str(opponent)][str(phase)] = (
                float((standardized[selected].abs() <= 1.6448536269514722).float().mean())
                if bool(selected.any()) else None)
    per_dim_90 = coverage["90"]
    overall_90 = float(per_dim_90.mean())
    main_strata = [value for phases in stratified_90.values()
                   for value in phases.values() if value is not None]
    gate_pass = (
        0.85 <= overall_90 <= 0.95
        and bool(((per_dim_90 >= 0.75) & (per_dim_90 <= 0.98)).all())
        and min(main_strata) >= 0.75)
    return {
        "variance_scale_by_dimension": variance_scale.cpu().tolist(),
        "nll_per_dimension": nll_dimension.cpu().tolist(),
        "rmse_per_dimension": residual.square().mean(0).sqrt().cpu().tolist(),
        "coverage_50_by_dimension": coverage["50"].cpu().tolist(),
        "coverage_90_by_dimension": per_dim_90.cpu().tolist(),
        "coverage_95_by_dimension": coverage["95"].cpu().tolist(),
        "coverage_90_overall": overall_90,
        "coverage_90_by_opponent_phase": stratified_90,
        "standardized_residual_mean": standardized.mean(0).cpu().tolist(),
        "standardized_residual_std": standardized.std(0, unbiased=False).cpu().tolist(),
        "pit_histogram_10_by_dimension": pit_histogram,
        "pit_ks_by_dimension": pit_ks,
        "sigma_boundary_fraction": {
            "lower": float((variances.sqrt() <= 1.001e-3).float().mean()),
            "upper": float((variances.sqrt() >= 9.999).float().mean()),
        },
        "pass": gate_pass,
    }, variance_scale


def evaluate_pair(episodes, dataset, delta, branch, input_kind, args,
                  full_calibration=False):
    device = torch.device("cuda")
    flat = {
        split: flatten_response_data(episodes, split, delta, branch).to(device)
        for split in ("train", "validation", "calibration", "test")}
    standardizer = TargetStandardizer.fit(flat["train"].target)
    for value in flat.values():
        value.target = standardizer.transform(value.target)
    conditioned, no_action, training = train_paired_ensembles(
        flat["train"], flat["validation"], input_kind,
        flat["train"].target.shape[-1], args)
    test = flat["test"]
    inputs = test.belief if input_kind == "belief" else test.observation
    zeros = torch.zeros_like(test.action)
    conditioned_nll = mixture_nll_batched(
        conditioned, inputs, test.action, test.target)
    no_action_nll = mixture_nll_batched(
        no_action, inputs, zeros, test.target)
    delta_nll = no_action_nll - conditioned_nll
    delta_mean, delta_ci, delta_strata = stratified_episode_bootstrap(
        delta_nll, test.episode, test.opponent, args.seed + delta * 101)
    permuted = stratified_action_permutation(
        test.action, test.opponent, test.phase, args.seed + delta * 103)
    permuted_nll = mixture_nll_batched(
        conditioned, inputs, permuted, test.target)
    permutation_gain = permuted_nll - conditioned_nll
    permutation_mean, permutation_ci, permutation_strata = (
        stratified_episode_bootstrap(
            permutation_gain, test.episode, test.opponent,
            args.seed + delta * 107))
    result = {
        "dataset": dataset,
        "branch": branch,
        "delta": delta,
        "context": input_kind,
        "train_samples": int(flat["train"].target.shape[0]),
        "test_samples": int(test.target.shape[0]),
        "test_episodes": int(torch.unique(test.episode).numel()),
        "mean_delta_nll_per_dimension": delta_mean,
        "delta_nll_ci95": delta_ci,
        "delta_nll_by_opponent": delta_strata,
        "permutation_gain_per_dimension": permutation_mean,
        "permutation_gain_ci95": permutation_ci,
        "permutation_gain_by_opponent": permutation_strata,
        "training": training,
    }
    calibration = None
    variance_scale = None
    if full_calibration:
        calibration, variance_scale = calibration_report(
            conditioned, flat["calibration"], test, input_kind)
        result["calibration"] = calibration
    artifacts = {
        "conditioned": conditioned.state_dict(),
        "no_action": no_action.state_dict(),
        "standardizer": standardizer.state_dict(),
        "variance_scale": variance_scale,
        "input_dim": int(inputs.shape[-1]),
        "response_dim": int(test.target.shape[-1]),
        "hidden_size": int(args.hidden_size),
        "members": 5,
        "sigma_min": 1.0e-3,
        "sigma_max": 10.0,
    }
    return result, artifacts


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    datasets = {
        "onpolicy": load_episodes(args.onpolicy, "onpolicy"),
        "cover": load_episodes(args.cover, "cover"),
        "reactive": load_episodes(args.reactive, "reactive"),
    }
    sensitivity = []
    main_artifacts = None
    for dataset in ("cover", "onpolicy"):
        for delta in (1, 3, 5):
            for context in ("direct", "belief"):
                is_main = dataset == "onpolicy" and delta == 3 and context == "belief"
                result, artifacts = evaluate_pair(
                    datasets[dataset], dataset, delta, "rel", context,
                    args, full_calibration=is_main)
                sensitivity.append(result)
                if is_main:
                    main_artifacts = artifacts
    reactive_positive = []
    for context in ("direct", "belief"):
        result, _ = evaluate_pair(
            datasets["reactive"], "reactive", 3, "opp", context, args)
        reactive_positive.append(result)
    main_result = next(item for item in sensitivity if (
        item["dataset"] == "onpolicy" and item["delta"] == 3
        and item["context"] == "belief"))
    g1_pass = (
        main_result["mean_delta_nll_per_dimension"] >= 0.05
        and main_result["delta_nll_ci95"][0] > 0.0
        and main_result["permutation_gain_per_dimension"] > 0.0
        and main_result["permutation_gain_ci95"][0] > 0.0)
    reactive_pass = all(
        item["mean_delta_nll_per_dimension"] > 0.0
        and item["delta_nll_ci95"][0] > 0.0
        for item in reactive_positive)
    g2_pass = bool(main_result["calibration"]["pass"])
    numerical_pass = all(
        not item["training"]["gradient_fault_over_1000"]
        for item in sensitivity + reactive_positive)
    report = {
        "suite": "DRVOI-RHPPO-V7-P1R-P1C",
        "protocol_version": "DRVOI-RHPPO-V7-WVR-1",
        "scientific_result": False,
        "models": {
            "M0": "direct/no-action",
            "M1": "direct/action-conditioned",
            "M2": "mature-frozen-B0-belief/no-action",
            "M3": "mature-frozen-B0-belief/action-conditioned",
            "M4": "M3 with opponent-phase-stratified shuffled action",
            "M5": "known two-step reactive positive control",
        },
        "primary": main_result,
        "sensitivity": sensitivity,
        "reactive_positive_control": reactive_positive,
        "gates": {
            "G1_action_identifiability": {"pass": g1_pass},
            "G1_reactive_positive_control": {"pass": reactive_pass},
            "G2_probability_calibration": {"pass": g2_pass},
            "G0_response_numerical_health": {"pass": numerical_pass},
        },
        "all_gates_pass": (
            g1_pass and reactive_pass and g2_pass and numerical_pass),
        "long_training_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    torch.save(main_artifacts, args.output.with_suffix(".pt"))
    for name, payload in (
        ("response_gate.json", report["gates"]["G1_action_identifiability"]),
        ("calibration_gate.json", report["gates"]["G2_probability_calibration"]),
    ):
        (args.output.parent / name).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["all_gates_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
