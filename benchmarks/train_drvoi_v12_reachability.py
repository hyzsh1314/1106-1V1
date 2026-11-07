#!/usr/bin/env python3
"""Train and calibrate the V12 reachability ensemble from paired branches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from benchmarks.drvoi.checkpointing import atomic_torch_save
from benchmarks.drvoi_v12 import (
    EXECUTION_FEATURE_DIM,
    ReachabilityEnsemble,
    aggregate_hypothesis_targets,
    write_json_atomic,
)


PROTOCOL_VERSION = "MRAT-RHPPO-V12-REACH-MODEL-1"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--softmin-temperature", type=float, default=0.25)
    parser.add_argument("--censor-horizon", type=float, default=30.0)
    parser.add_argument(
        "--score-target", choices=(
            "reachability", "discounted_return",
            "discounted_return_behavior"),
        default="reachability")
    parser.add_argument("--seed", type=int, default=85)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def _group_bucket(seed: int, env_index: int, episode_id: int) -> int:
    key = f"{seed}:{env_index}:{episode_id}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % 10_000


def _split_masks(env_index: torch.Tensor, episode_id: torch.Tensor,
                 seed: int) -> dict[str, torch.Tensor]:
    buckets = torch.tensor([
        _group_bucket(seed, int(env), int(episode))
        for env, episode in zip(env_index.tolist(), episode_id.tolist())
    ])
    masks = {
        "train": buckets < 7000,
        "validation": (buckets >= 7000) & (buckets < 8500),
        "calibration": (buckets >= 8500) & (buckets < 9250),
        "test": buckets >= 9250,
    }
    if any(not bool(mask.any()) for mask in masks.values()):
        raise ValueError(
            "grouped split produced an empty partition; collect more complete episodes")
    return masks


def _load_dataset(data_dir: Path, temperature: float,
                  score_target: str = "reachability"):
    paths = sorted((data_dir / "chunks").glob("reachability-*.pt"))
    if not paths:
        raise FileNotFoundError(f"no reachability chunks in {data_dir}")
    chunks = [torch.load(path, map_location="cpu", weights_only=False)
              for path in paths]
    protocols = {chunk.get("protocol_version") for chunk in chunks}
    if len(protocols) != 1 or not protocols.issubset({
            "MRAT-RHPPO-V12-REACH-DATA-2",
            "MRAT-RHPPO-V12-REACH-DATA-3",
            "MRAT-RHPPO-V12-REACH-DATA-4"}):
        raise ValueError("dataset chunk protocol mismatch")
    features = torch.cat([chunk["features"] for chunk in chunks])
    if features.shape[1] != EXECUTION_FEATURE_DIM:
        raise ValueError("dataset feature dimension mismatch")
    score_field = (
        "target_score" if score_target == "reachability"
        else "target_discounted_return")
    if any(score_field not in chunk for chunk in chunks):
        raise ValueError(f"dataset does not contain {score_field}")
    raw_score = torch.cat([chunk[score_field] for chunk in chunks])
    raw_legal = torch.cat([chunk["target_legal"] for chunk in chunks])
    raw_time = torch.cat([chunk["target_first_legal"] for chunk in chunks])
    aggregated = aggregate_hypothesis_targets(
        raw_score, raw_legal, raw_time, temperature=temperature)
    env_index = torch.cat([chunk["env_index"] for chunk in chunks])
    if score_target == "discounted_return_behavior":
        category = env_index.remainder(3)
        evasive_direction = 2 + torch.div(
            env_index, 3, rounding_mode="floor").remainder(2)
        behavior_hypothesis = torch.where(
            category == 2, evasive_direction, category)
        score = raw_score.gather(
            2, behavior_hypothesis[:, None, None].expand(-1, 2, 1)
        ).squeeze(-1)
    else:
        score = aggregated["score"]
    num_states = features.shape[0]
    expert = torch.arange(2).repeat(num_states)
    return {
        "paths": paths,
        "score_target": score_target,
        "num_states": num_states,
        "features": features.repeat_interleave(2, 0),
        "expert": expert,
        "target_score": score.reshape(-1),
        "target_probability": aggregated["probability"].reshape(-1),
        "target_time": aggregated["time"].reshape(-1),
        "event_observed": aggregated["event_observed"].reshape(-1),
        "env_index": env_index.repeat_interleave(2),
        "episode_id": torch.cat([
            chunk["episode_id"] for chunk in chunks]).repeat_interleave(2),
    }


def _member_loss(member, features, expert, target_score,
                 target_probability, target_time, event_observed,
                 censor_horizon):
    score, probability, time_to_event = member(features, expert)
    score_loss = F.smooth_l1_loss(score, target_score)
    probability_loss = F.binary_cross_entropy(
        probability, target_probability)
    if bool(event_observed.any()):
        time_loss = F.smooth_l1_loss(
            time_to_event[event_observed], target_time[event_observed])
    else:
        time_loss = time_to_event.sum() * 0.0
    censored = ~event_observed
    if bool(censored.any()):
        censored_loss = F.relu(
            censor_horizon - time_to_event[censored]).square().mean()
    else:
        censored_loss = time_to_event.sum() * 0.0
    total = score_loss + probability_loss + 0.25 * time_loss + 0.25 * censored_loss
    return total, {
        "score_loss": float(score_loss.detach()),
        "probability_loss": float(probability_loss.detach()),
        "time_loss": float(time_loss.detach()),
        "censored_loss": float(censored_loss.detach()),
    }


def _binary_auc(score: torch.Tensor, label: torch.Tensor) -> float:
    label = label.bool()
    positives = int(label.sum())
    negatives = len(label) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = torch.argsort(score)
    sorted_score = score[order]
    _, counts = torch.unique_consecutive(sorted_score, return_counts=True)
    sorted_ranks = torch.empty(len(score), dtype=torch.float64)
    offset = 0
    for count in counts.tolist():
        first, last = offset + 1, offset + count
        sorted_ranks[offset:offset + count] = (first + last) / 2.0
        offset += count
    ranks = torch.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    rank_sum = ranks[label].sum()
    return float((rank_sum - positives * (positives + 1) / 2) /
                 (positives * negatives))


def _average_precision(score: torch.Tensor, label: torch.Tensor) -> float:
    label = label.bool()
    positives = int(label.sum())
    if positives == 0:
        return float("nan")
    ordered = label[torch.argsort(score, descending=True)].float()
    precision = ordered.cumsum(0) / torch.arange(1, len(ordered) + 1)
    return float((precision * ordered).sum() / positives)


def _spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    if len(left) < 2:
        return float("nan")
    left_rank = torch.argsort(torch.argsort(left)).float()
    right_rank = torch.argsort(torch.argsort(right)).float()
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = left_rank.norm() * right_rank.norm()
    return float((left_rank * right_rank).sum() / denominator.clamp_min(1e-8))


def _ece(probability: torch.Tensor, soft_target: torch.Tensor,
         bins: int = 10) -> float:
    total = len(probability)
    error = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = (probability >= lower) & (
            probability < upper if index + 1 < bins else probability <= upper)
        if bool(selected.any()):
            error += (int(selected.sum()) / total) * abs(
                float(probability[selected].mean() - soft_target[selected].mean()))
    return error


def _paired_advantage_metrics(
        prediction, target_score: torch.Tensor, lcb_z: float,
        state_mask: torch.Tensor | None = None) -> dict:
    if len(target_score) % 2:
        raise ValueError("paired expert rows must have even length")
    predicted = prediction.score_mean.reshape(-1, 2)
    target = target_score.reshape(-1, 2)
    lcb = prediction.lower_confidence_bound(lcb_z).reshape(-1, 2)
    predicted_advantage = predicted[:, 1] - predicted[:, 0]
    target_advantage = target[:, 1] - target[:, 0]
    conservative_advantage = lcb[:, 1] - lcb[:, 0]
    if state_mask is not None:
        state_mask = state_mask.to(device=target.device, dtype=torch.bool)
        predicted_advantage = predicted_advantage[state_mask]
        target_advantage = target_advantage[state_mask]
        conservative_advantage = conservative_advantage[state_mask]
    quantiles = torch.tensor(
        [0.05, 0.25, 0.50, 0.75, 0.95], device=target.device)
    return {
        "states": int(len(target_advantage)),
        "target_intercept_better_fraction": float(
            (target_advantage > 0).float().mean()),
        "predicted_intercept_better_fraction": float(
            (predicted_advantage > 0).float().mean()),
        "conservative_gate_allowed_fraction": float(
            (conservative_advantage >= 0).float().mean()),
        "advantage_sign_accuracy": float(
            ((predicted_advantage >= 0)
             == (target_advantage >= 0)).float().mean()),
        "advantage_mae": float(
            (predicted_advantage - target_advantage).abs().mean()),
        "advantage_spearman": _spearman(
            predicted_advantage.cpu(), target_advantage.cpu()),
        "target_advantage_quantiles": torch.quantile(
            target_advantage, quantiles).tolist(),
        "predicted_advantage_quantiles": torch.quantile(
            predicted_advantage, quantiles).tolist(),
        "conservative_advantage_quantiles": torch.quantile(
            conservative_advantage, quantiles).tolist(),
    }


@torch.no_grad()
def _evaluate(model, data, indices, mean, scale, device):
    features = ((data["features"][indices].to(device) - mean) / scale)
    expert = data["expert"][indices].to(device)
    prediction = model(features, expert)
    target_score = data["target_score"][indices].to(device)
    target_probability = data["target_probability"][indices].to(device)
    target_time = data["target_time"][indices].to(device)
    event = data["event_observed"][indices].to(device)
    metrics = {
        "samples": int(len(indices)),
        "event_rate": float(event.float().mean()),
        "soft_probability_base_rate": float(target_probability.mean()),
        "score_mae": float((prediction.score_mean - target_score).abs().mean()),
        "score_spearman": _spearman(
            prediction.score_mean.cpu(), target_score.cpu()),
        "probability_brier": float(
            (prediction.probability_mean - target_probability).square().mean()),
        "probability_auroc": _binary_auc(
            prediction.probability_mean.cpu(), event.cpu()),
        "probability_auprc": _average_precision(
            prediction.probability_mean.cpu(), event.cpu()),
        "probability_ece": _ece(
            prediction.probability_mean.cpu(), target_probability.cpu()),
        "time_mae_observed": float(
            (prediction.time_mean[event] - target_time[event]).abs().mean()
        ) if bool(event.any()) else float("nan"),
    }
    observation = data["features"][indices, :39].to(device)
    old_guard = (
        (observation[:, 13] * 10000.0 >= 6000.0)
        & (observation[:, 28] * 340.0 <= 17.0)
    ).float()
    intercept = expert == 1
    if bool(intercept.any()):
        metrics["old_guard_intercept_brier"] = float(
            (old_guard[intercept] - target_probability[intercept]).square().mean())
        metrics["old_guard_intercept_auroc"] = _binary_auc(
            old_guard[intercept].cpu(), event[intercept].cpu())
    return metrics, prediction, target_score


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    data = _load_dataset(
        args.data_dir, args.softmin_temperature, args.score_target)
    dataset_completed = json.loads(
        (args.data_dir / "COMPLETED.json").read_text(encoding="utf-8"))
    base_masks = _split_masks(
        data["env_index"][::2], data["episode_id"][::2], args.seed)
    masks = {name: mask.repeat_interleave(2)
             for name, mask in base_masks.items()}
    indices = {name: torch.nonzero(mask, as_tuple=False).flatten()
               for name, mask in masks.items()}
    train_features = data["features"][indices["train"]]
    feature_mean = train_features.mean(0).to(device)
    feature_scale = train_features.std(0, unbiased=False).clamp_min(1.0e-4).to(device)
    model = ReachabilityEnsemble(
        feature_dim=EXECUTION_FEATURE_DIM, members=args.members,
        hidden_size=args.hidden_size).to(device)
    # Start censored-time predictions at the horizon. Otherwise the initial
    # squared censor penalty (~H^2) overwhelms score/probability supervision.
    time_bias = math.log(math.expm1(args.censor_horizon))
    with torch.no_grad():
        for member in model.members:
            member.time_head.weight.zero_()
            member.time_head.bias.fill_(time_bias)
    optimizers = [torch.optim.AdamW(
        member.parameters(), lr=args.learning_rate, weight_decay=1.0e-5)
        for member in model.members]
    generator = torch.Generator().manual_seed(args.seed)
    history = []
    started = time.perf_counter()
    train_index = indices["train"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        member_records = []
        for member_index, (member, optimizer) in enumerate(
                zip(model.members, optimizers)):
            bootstrap = train_index[torch.randint(
                len(train_index), (len(train_index),), generator=generator)]
            bootstrap = bootstrap[torch.randperm(
                len(bootstrap), generator=generator)]
            totals = {key: 0.0 for key in (
                "loss", "score_loss", "probability_loss", "time_loss",
                "censored_loss")}
            batches = 0
            for start in range(0, len(bootstrap), args.batch_size):
                batch = bootstrap[start:start + args.batch_size]
                features = (
                    data["features"][batch].to(device) - feature_mean
                ) / feature_scale
                event = data["event_observed"][batch].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss, components = _member_loss(
                    member, features, data["expert"][batch].to(device),
                    data["target_score"][batch].to(device),
                    data["target_probability"][batch].to(device),
                    data["target_time"][batch].to(device), event,
                    args.censor_horizon)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    member.parameters(), 5.0)
                optimizer.step()
                totals["loss"] += float(loss.detach())
                for key, value in components.items():
                    totals[key] += value
                batches += 1
            record = {key: value / max(batches, 1)
                      for key, value in totals.items()}
            record["member"] = member_index
            record["grad_norm_last"] = float(grad_norm)
            member_records.append(record)
        validation, _, _ = _evaluate(
            model, data, indices["validation"], feature_mean, feature_scale,
            device)
        epoch_record = {
            "epoch": epoch,
            "members": member_records,
            "validation": validation,
            "wall_time_s": time.perf_counter() - started,
        }
        history.append(epoch_record)
        with metrics_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({
                "metric_schema": "MRAT-RHPPO-V12-REACH-TRAIN-METRICS-1",
                **epoch_record,
            }, sort_keys=True) + "\n")
        write_json_atomic(args.output_dir / "progress.json", {
            "protocol_version": PROTOCOL_VERSION,
            "epoch": epoch,
            "epochs": args.epochs,
            "progress_fraction": epoch / args.epochs,
            "validation": validation,
            "wall_time_s": epoch_record["wall_time_s"],
        })
        atomic_torch_save({
            "protocol_version": PROTOCOL_VERSION,
            "epoch": epoch,
            "feature_dim": EXECUTION_FEATURE_DIM,
            "members": args.members,
            "hidden_size": args.hidden_size,
            "feature_mean": feature_mean.cpu(),
            "feature_scale": feature_scale.cpu(),
            "model": model.state_dict(),
            "optimizers": [optimizer.state_dict() for optimizer in optimizers],
            "history": history,
            "seed": args.seed,
        }, args.output_dir / "latest.pt")
        print(json.dumps(epoch_record, ensure_ascii=False), flush=True)

    calibration, calibration_prediction, calibration_target = _evaluate(
        model, data, indices["calibration"], feature_mean, feature_scale,
        device)
    candidates = (1.0, 1.64)
    coverages = {}
    selected_z = candidates[-1]
    for z_value in candidates:
        lcb = calibration_prediction.lower_confidence_bound(z_value)
        coverage = float((calibration_target >= lcb).float().mean())
        coverages[str(z_value)] = coverage
        if coverage >= 0.90:
            selected_z = z_value
            break
    test, test_prediction, test_target = _evaluate(
        model, data, indices["test"], feature_mean, feature_scale, device)
    test_pair_advantage = _paired_advantage_metrics(
        test_prediction, test_target, selected_z)
    test_state_indices = torch.div(
        indices["test"][::2], 2, rounding_mode="floor")
    test_categories = data["env_index"][::2][test_state_indices].remainder(3)
    test_pair_advantage_by_behavior = {
        name: _paired_advantage_metrics(
            test_prediction, test_target, selected_z,
            test_categories == category)
        for category, name in enumerate(("hold", "pursuit", "evasive"))
    }
    split_counts = {name: int(len(value)) for name, value in indices.items()}
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "scientific_result": False,
        "dataset_states": data["num_states"],
        "dataset_chunks": len(data["paths"]),
        "dataset_compute": {
            "online_environment_decisions": dataset_completed[
                "online_environment_decisions"],
            "teacher_model_decisions": dataset_completed[
                "teacher_model_decisions"],
            "wall_time_seconds": dataset_completed["wall_time_seconds"],
        },
        "split_counts_expert_rows": split_counts,
        "split_rule": "SHA256(seed,env_index,episode_id): 70/15/7.5/7.5",
        "softmin_temperature": args.softmin_temperature,
        "score_target": args.score_target,
        "censor_horizon": args.censor_horizon,
        "calibration": calibration,
        "lcb_coverage": coverages,
        "selected_lcb_z": selected_z,
        "test": test,
        "test_pair_advantage": test_pair_advantage,
        "test_pair_advantage_by_behavior": (
            test_pair_advantage_by_behavior),
        "history": history,
        "wall_time_s": time.perf_counter() - started,
    }
    checkpoint = {
        "protocol_version": PROTOCOL_VERSION,
        "algorithm": "mrat_reachability_ensemble",
        "score_target": args.score_target,
        "feature_dim": EXECUTION_FEATURE_DIM,
        "members": args.members,
        "hidden_size": args.hidden_size,
        "softmin_temperature": args.softmin_temperature,
        "censor_horizon": args.censor_horizon,
        "selected_lcb_z": selected_z,
        "feature_mean": feature_mean.cpu(),
        "feature_scale": feature_scale.cpu(),
        "model": model.state_dict(),
        "seed": args.seed,
        "dataset_manifest": json.loads(
            (args.data_dir / "dataset-manifest.json").read_text(encoding="utf-8")),
        "dataset_completed": dataset_completed,
    }
    atomic_torch_save(checkpoint, args.output_dir / "reachability.pt")
    write_json_atomic(args.output_dir / "report.json", report)
    write_json_atomic(args.output_dir / "COMPLETED.json", {
        "protocol_version": PROTOCOL_VERSION,
        "selected_lcb_z": selected_z,
        "test": test,
        "wall_time_s": report["wall_time_s"],
    })
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
