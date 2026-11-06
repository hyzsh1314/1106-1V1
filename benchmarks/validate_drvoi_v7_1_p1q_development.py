#!/usr/bin/env python3
"""Select V7.1 Q target-update cadence on train/validation only."""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from benchmarks.drvoi import multi_discrete_to_one_hot
from benchmarks.drvoi.q_model import (
    ActionEmbeddingQ,
    FactorizedDuelingQ,
    JointDuelingQ,
    double_q_target,
)
from benchmarks.drvoi.targets import response_state_v7
from benchmarks.drvoi_v7_1.q_controls import bootstrap_context_mean
from benchmarks.validate_drvoi_wvr_q_v7 import spearman


@dataclass
class QDevelopmentData:
    context: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_context: torch.Tensor
    done: torch.Tensor
    monte_carlo_return: torch.Tensor
    episode: torch.Tensor
    collection_seed: torch.Tensor

    def to(self, device):
        return QDevelopmentData(**{
            key: value.to(device) for key, value in vars(self).items()})

    def select(self, index):
        return QDevelopmentData(**{
            key: value[index] for key, value in vars(self).items()})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onpolicy", type=Path, nargs="+", required=True)
    parser.add_argument("--config", type=Path, default=Path(
        "configs/drvoi_rhppo_v7_1_p1q.json"))
    parser.add_argument("--seed", type=int, default=41185)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_development(paths):
    episodes, seeds = [], set()
    for path_index, path in enumerate(paths):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        report = payload["report"]
        if report.get("dataset") != "onpolicy" or report.get(
                "contains_confirmation_test", True):
            raise ValueError(f"{path} is not development-only onpolicy data")
        if set(report.get("available_splits", ())) != {"train", "validation"}:
            raise ValueError(f"{path} contains a forbidden split")
        seeds.add(int(report["collection_seed"]))
        for episode in payload["episodes"]:
            copied = dict(episode)
            copied["episode_id"] = path_index * 10**9 + int(episode["episode_id"])
            episodes.append(copied)
    if len(seeds) != 5:
        raise ValueError("Q development requires five collection seeds")
    return episodes, sorted(seeds)


def build_q_data(episodes, split, delta=3, gamma=0.99):
    values = {key: [] for key in (
        "context", "action", "reward", "next_context", "done",
        "monte_carlo_return", "episode", "collection_seed")}
    for episode in episodes:
        if episode["split"] != split:
            continue
        observation = episode["observation"]
        belief = episode["belief"]
        action = multi_discrete_to_one_hot(episode["action"])
        reward = episode["reward"]
        length = int(reward.numel())
        response = response_state_v7(observation, "rel")
        bases = torch.arange(0, length, delta)
        decisions = bases + delta
        selected = decisions < length
        bases, decisions = bases[selected], decisions[selected]
        return_end = (decisions + delta).clamp_max(length)
        terminal = return_end == length
        context = torch.cat((
            belief[bases], action[bases], response[decisions]), -1)
        offsets = torch.arange(delta)
        reward_index = decisions[:, None] + offsets[None]
        valid_reward = reward_index < length
        safe_reward_index = reward_index.clamp_max(length - 1)
        discount = torch.as_tensor(gamma).pow(offsets)
        discounted = (
            reward[safe_reward_index] * discount[None]
            * valid_reward.float()).sum(-1)
        safe_next = return_end.clamp_max(length - 1)
        next_context = torch.cat((
            belief[decisions], action[decisions], response[safe_next]), -1)
        next_context[terminal] = 0
        # G_i = gamma^-i * sum_{k=i}^{T-1} gamma^k r_k.  Episode lengths
        # are short enough that this vectorized form remains well conditioned
        # and exactly preserves the terminal no-bootstrap semantics.
        powers = torch.as_tensor(gamma).pow(torch.arange(length))
        discounted_suffix = torch.flip(torch.cumsum(torch.flip(
            reward * powers, (0,)), 0), (0,)) / powers
        batch = decisions.numel()
        for key, value in (
            ("context", context), ("action", action[decisions]),
            ("reward", discounted), ("next_context", next_context),
            ("done", terminal),
            ("monte_carlo_return", discounted_suffix[decisions]),
            ("episode", torch.full(
                (batch,), int(episode["episode_id"]), dtype=torch.long)),
            ("collection_seed", torch.full(
                (batch,), int(episode["collection_seed"]), dtype=torch.long)),
        ):
            values[key].append(value)
    return QDevelopmentData(**{
        key: torch.cat(items) for key, items in values.items()})


def stratified_subsample(data, maximum, seed):
    if data.context.shape[0] <= maximum:
        return data
    generator = torch.Generator().manual_seed(seed)
    per_seed = maximum // len(torch.unique(data.collection_seed))
    selected = []
    for collection_seed in torch.unique(data.collection_seed):
        index = torch.nonzero(data.collection_seed == collection_seed).flatten()
        selected.append(index[torch.randperm(index.numel(), generator=generator)[:per_seed]])
    return data.select(torch.cat(selected))


def standardize(train, validation):
    mean = train.context.mean(0)
    scale = train.context.std(0, unbiased=False).clamp_min(1e-5)
    for data in (train, validation):
        data.context = (data.context - mean) / scale
        data.next_context = (data.next_context - mean) / scale
        data.next_context[data.done] = 0
    return mean, scale


@torch.no_grad()
def validation_metrics(model, data, seed):
    prediction = model(data.context, data.action)
    mse = float(torch.nn.functional.mse_loss(
        prediction, data.monte_carlo_return))
    unique = torch.unique(data.episode)
    predicted_episode = torch.stack([
        prediction[data.episode == episode].mean() for episode in unique])
    true_episode = torch.stack([
        data.monte_carlo_return[data.episode == episode].mean()
        for episode in unique])
    rho = spearman(predicted_episode, true_episode)
    seed_rhos = {}
    for collection_seed in torch.unique(data.collection_seed):
        selected = data.collection_seed == collection_seed
        seed_rhos[str(int(collection_seed))] = spearman(
            prediction[selected], data.monte_carlo_return[selected])
    return {"mc_mse": mse, "episode_spearman": rho,
            "tuple_spearman_by_collection_seed": seed_rhos}


def build_q_model(context_dim, architecture):
    if architecture == "action_embedding_v1":
        return ActionEmbeddingQ(context_dim)
    if architecture == "factorized_dueling_pairwise_v1":
        return FactorizedDuelingQ(context_dim)
    if architecture == "joint_dueling_v1":
        return JointDuelingQ(context_dim)
    raise ValueError(f"unknown Q architecture: {architecture}")


def train_candidate(train, validation, period, cfg, seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = build_q_model(
        train.context.shape[-1], cfg["architecture"]).to(train.context.device)
    target = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    generator = torch.Generator(device=train.context.device).manual_seed(seed + 17)
    best_score, best_state, stale, history = float("inf"), None, 0, []
    for epoch in range(1, cfg["epochs"] + 1):
        order = torch.randperm(train.context.shape[0], device=train.context.device,
                               generator=generator)
        losses = []
        for start in range(0, order.numel(), cfg["batch_size"]):
            selected = order[start:start + cfg["batch_size"]]
            with torch.no_grad():
                target_value = double_q_target(
                    train.reward[selected], train.done[selected],
                    train.next_context[selected], model, target,
                    gamma=0.99, discount_steps=3)
            prediction = model(train.context[selected], train.action[selected])
            loss = torch.nn.functional.smooth_l1_loss(prediction, target_value)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            if not math.isfinite(float(norm)):
                raise FloatingPointError("non-finite Q gradient")
            optimizer.step(); losses.append(float(loss.detach()))
        if epoch % period == 0:
            target.load_state_dict(model.state_dict())
        metrics = validation_metrics(model, validation, seed + epoch)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **metrics})
        if metrics["mc_mse"] < best_score - 1e-6:
            best_score = metrics["mc_mse"]
            best_state = copy.deepcopy(model.state_dict()); stale = 0
        else:
            stale += 1
            if stale >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return model, history, validation_metrics(model, validation, seed + 999)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    hot = json.loads(Path(
        "results/drvoi_rhppo_v7_1/p1q/hot_development_validation.json"
    ).read_text(encoding="utf-8"))
    if not hot.get("wvr_q_test_generation_authorized"):
        raise PermissionError("HOT did not authorize WVR Q development")
    episodes, seeds = load_development(args.onpolicy)
    cfg = protocol["q_development"]
    train = stratified_subsample(
        build_q_data(episodes, "train"), cfg["max_train_tuples"], args.seed)
    validation = stratified_subsample(
        build_q_data(episodes, "validation"), cfg["max_validation_tuples"],
        args.seed + 1)
    mean, scale = standardize(train, validation)
    device = torch.device("cuda")
    train, validation = train.to(device), validation.to(device)
    candidates, artifacts = {}, {}
    for period in protocol["target_network_candidates"]:
        model, history, metrics = train_candidate(
            train, validation, period, cfg, args.seed)
        candidates[str(period)] = {"target_update_epochs": period,
                                   "history": history, "validation": metrics}
        artifacts[str(period)] = model.state_dict()
    selected = min(candidates, key=lambda key: candidates[key][
        "validation"]["mc_mse"])
    selected_metrics = candidates[selected]["validation"]
    report = {
        "suite": "DRVOI-RHPPO-V7.1-P1Q-WVR-development",
        "protocol_version": protocol["protocol_version"],
        "scientific_result": False,
        "wvr_q_test_accessed": False,
        "collection_seeds": seeds,
        "architecture": cfg["architecture"],
        "train_tuples": int(train.context.shape[0]),
        "validation_tuples": int(validation.context.shape[0]),
        "candidates": candidates,
        "selected_target_update_epochs": int(selected),
        "selected_validation": selected_metrics,
        "development_semantics_pass": (
            selected_metrics["episode_spearman"] >= protocol[
                "wvr_value_gate"]["episode_spearman_min_auxiliary"]),
        "mc_probe_required": True,
        "formal_q_test_authorized": False,
        "o0_authorized": False,
        "long_training_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    torch.save({
        "selected": selected, "models": artifacts,
        "architecture": cfg["architecture"],
        "context_mean": mean, "context_scale": scale,
        "context_dim": train.context.shape[-1],
    }, args.output.with_suffix(".pt"))
    print(json.dumps(report, indent=2))
    if not report["development_semantics_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
