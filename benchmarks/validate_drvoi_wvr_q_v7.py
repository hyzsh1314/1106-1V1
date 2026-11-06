#!/usr/bin/env python3
"""Train and gate the cross-fitted, cross-delta extrinsic Q proxy for V7."""

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
from benchmarks.drvoi.checkpointing import atomic_torch_save
from benchmarks.drvoi.gates import evaluate_q_ready
from benchmarks.drvoi.q_model import ActionEmbeddingQ, double_q_target
from benchmarks.drvoi.targets import response_state_v7


@dataclass
class QData:
    context: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_context: torch.Tensor
    done: torch.Tensor
    monte_carlo_return: torch.Tensor
    episode: torch.Tensor

    def to(self, device):
        return QData(**{key: value.to(device) for key, value in vars(self).items()})

    def select(self, selected):
        return QData(**{key: value[selected] for key, value in vars(self).items()})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onpolicy", type=Path, nargs="+", required=True)
    parser.add_argument("--delta", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=10185)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/drvoi_rhppo_v7_wvr.json"))
    return parser.parse_args()


def load_episodes(paths):
    result = []
    for path_index, path in enumerate(paths):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["report"]["dataset"] != "onpolicy":
            raise ValueError("Q validation only accepts D-onpolicy")
        for episode in payload["episodes"]:
            copied = dict(episode)
            copied["episode_id"] = path_index * 10**9 + int(episode["episode_id"])
            result.append(copied)
    return result


def build_q_data(episodes, split: str, delta=3, gamma=0.99) -> QData:
    values = {key: [] for key in (
        "context", "action", "reward", "next_context", "done",
        "monte_carlo_return", "episode")}
    for episode in episodes:
        if episode["split"] != split:
            continue
        observation = episode["observation"]
        belief = episode["belief"]
        action = multi_discrete_to_one_hot(episode["action"])
        reward = episode["reward"]
        length = int(reward.numel())
        response = response_state_v7(observation, "rel")
        for base in range(0, length, delta):
            return_end = base + 2 * delta - 1
            if return_end >= length:
                break
            decision = base + delta
            terminal = return_end == length - 1
            context = torch.cat((belief[base], action[base], response[decision]))
            discounted = sum(
                reward[decision + offset] * (gamma ** offset)
                for offset in range(delta))
            if terminal:
                next_context = torch.zeros_like(context)
            else:
                next_context = torch.cat((
                    belief[decision], action[decision],
                    response[base + 2 * delta]))
            mc_return = sum(
                reward[index] * (gamma ** (index - decision))
                for index in range(decision, length))
            values["context"].append(context)
            values["action"].append(action[decision])
            values["reward"].append(discounted)
            values["next_context"].append(next_context)
            values["done"].append(torch.tensor(terminal))
            values["monte_carlo_return"].append(mc_return)
            values["episode"].append(torch.tensor(
                int(episode["episode_id"]), dtype=torch.long))
    if not values["context"]:
        raise ValueError(f"no Q tuples for split {split}")
    return QData(**{key: torch.stack(items) for key, items in values.items()})


def standardize_context(train: QData, *others: QData):
    mean = train.context.mean(0)
    scale = train.context.std(0, unbiased=False).clamp_min(1.0e-5)
    train.context = (train.context - mean) / scale
    train.next_context = (train.next_context - mean) / scale
    train.next_context[train.done] = 0.0
    for value in others:
        value.context = (value.context - mean) / scale
        value.next_context = (value.next_context - mean) / scale
        value.next_context[value.done] = 0.0
    return mean, scale


def train_q(train: QData, context_dim, args, seed):
    online = ActionEmbeddingQ(context_dim).to(train.context.device)
    target = copy.deepcopy(online).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(online.parameters(), lr=3.0e-4)
    generator = torch.Generator(device=train.context.device).manual_seed(seed)
    history = []
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(
            train.context.shape[0], device=train.context.device,
            generator=generator)
        losses = []
        td_values = []
        for start in range(0, order.numel(), args.batch_size):
            selected = order[start:start + args.batch_size]
            with torch.no_grad():
                targets = double_q_target(
                    train.reward[selected], train.done[selected],
                    train.next_context[selected], online, target,
                    gamma=args.gamma, discount_steps=args.delta)
            prediction = online(train.context[selected], train.action[selected])
            td = targets - prediction
            loss = torch.nn.functional.smooth_l1_loss(prediction, targets)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite Q loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
            if not math.isfinite(float(norm)):
                raise FloatingPointError("non-finite Q gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
            td_values.append(td.detach().abs())
        target.load_state_dict(online.state_dict())
        td = torch.cat(td_values)
        history.append({
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "td_p90": float(torch.quantile(td, 0.9)),
        })
    return online, target, history


def rankdata(values: torch.Tensor) -> torch.Tensor:
    _, inverse, counts = torch.unique(
        values, sorted=True, return_inverse=True, return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    average_ranks = starts.float() + 0.5 * (counts.float() - 1.0)
    return average_ranks[inverse]


def spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() < 3:
        return float("nan")
    x = rankdata(left)
    y = rankdata(right)
    x = x - x.mean()
    y = y - y.mean()
    return float((x * y).sum() / (
        x.square().sum().sqrt() * y.square().sum().sqrt()).clamp_min(1.0e-8))


@torch.no_grad()
def heldout_metrics(model, heldout: QData, seed: int):
    prediction = model(heldout.context, heldout.action)
    episodes = torch.unique(heldout.episode)
    predicted_episode = torch.stack([
        prediction[heldout.episode == episode].mean() for episode in episodes])
    return_episode = torch.stack([
        heldout.monte_carlo_return[heldout.episode == episode].mean()
        for episode in episodes])
    rho = spearman(predicted_episode, return_episode)
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(1000):
        selected = torch.as_tensor(
            rng.integers(0, episodes.numel(), episodes.numel()),
            device=prediction.device)
        bootstrap.append(spearman(
            predicted_episode[selected], return_episode[selected]))
    ci = np.nanquantile(bootstrap, [0.025, 0.975]).tolist()
    # A real response input must affect Q; shuffle only the registered 12-D
    # response suffix and compare predictions at fixed state/action.
    shuffled = heldout.context.clone()
    shuffled[:, -12:] = shuffled[torch.randperm(
        shuffled.shape[0], device=shuffled.device), -12:]
    shifted = model(shuffled, heldout.action)
    sensitivity = float((prediction - shifted).abs().mean())
    q_std = float(prediction.std(unbiased=False))
    return {
        "rho": rho,
        "ci95": ci,
        "prediction": prediction,
        "response_sensitivity_mean_abs": sensitivity,
        "response_sensitivity_relative": sensitivity / max(q_std, 1.0e-8),
        "response_sensitivity_pass": sensitivity > max(1.0e-4, 0.01 * q_std),
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.delta != 3:
        raise ValueError("the primary V7 Q gate is locked to delta=3")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    episodes = load_episodes(args.onpolicy)
    train_all = build_q_data(episodes, "train", args.delta, args.gamma)
    heldout_all = build_q_data(episodes, "validation", args.delta, args.gamma)
    device = torch.device("cuda")
    fold_reports = {}
    artifacts = {}
    td_errors = []
    sensitivity_pass = []
    metrics = {"finite": True}
    for fold, name in ((0, "a"), (1, "b")):
        train = train_all.select((train_all.episode & 1) == fold).to(device)
        # Cross-fit semantics: each critic is evaluated on the opposite fold.
        heldout = heldout_all.select((heldout_all.episode & 1) != fold).to(device)
        mean, scale = standardize_context(train, heldout)
        model, target, history = train_q(
            train, train.context.shape[-1], args, args.seed + fold)
        heldout_report = heldout_metrics(model, heldout, args.seed + 100 + fold)
        with torch.no_grad():
            td_target = double_q_target(
                train.reward, train.done, train.next_context,
                model, target, gamma=args.gamma, discount_steps=args.delta)
            td = (td_target - model(train.context, train.action)).abs()
        td_errors.append(td)
        sensitivity_pass.append(heldout_report["response_sensitivity_pass"])
        episode_ids = torch.unique(train.episode)
        nonzero_episodes = sum(bool(
            train.reward[train.episode == episode].abs().max() > 1.0e-8)
            for episode in episode_ids)
        metrics[f"fold_{name}_episodes"] = int(episode_ids.numel())
        metrics[f"fold_{name}_tuples"] = int(train.context.shape[0])
        metrics[f"fold_{name}_nonzero_return_episodes"] = int(nonzero_episodes)
        metrics[f"fold_{name}_heldout_return_spearman"] = heldout_report["rho"]
        metrics[f"fold_{name}_heldout_return_spearman_ci_low"] = (
            heldout_report["ci95"][0])
        fold_reports[name] = {
            "training_history": history,
            "heldout_return_spearman": heldout_report["rho"],
            "heldout_return_spearman_ci95": heldout_report["ci95"],
            "response_sensitivity_mean_abs": heldout_report[
                "response_sensitivity_mean_abs"],
            "response_sensitivity_relative": heldout_report[
                "response_sensitivity_relative"],
            "response_sensitivity_pass": heldout_report[
                "response_sensitivity_pass"],
        }
        artifacts[name] = {
            "online": model.state_dict(),
            "target": target.state_dict(),
            "context_mean": mean,
            "context_scale": scale,
        }
    td = torch.cat(td_errors)
    metrics.update({
        "response_sensitivity_control": all(sensitivity_pass),
        "td_p50": float(torch.quantile(td, 0.50)),
        "td_p90": float(torch.quantile(td, 0.90)),
        "td_p99": float(torch.quantile(td, 0.99)),
    })
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    q_config = protocol["q_model"]
    gate_config = {
        "min_episodes_per_fold": q_config["ready_min_episodes_per_fold"],
        "min_tuples_per_fold": q_config["ready_min_tuples_per_fold"],
        "min_nonzero_return_episodes_per_fold": q_config[
            "ready_min_nonzero_return_episodes_per_fold"],
        "min_return_spearman": q_config["ready_min_return_spearman"],
    }
    gate = evaluate_q_ready(metrics, gate_config)
    report = {
        "suite": "DRVOI-RHPPO-V7-P1Q",
        "protocol_version": "DRVOI-RHPPO-V7-WVR-1",
        "scientific_result": False,
        "delta": args.delta,
        "q_stride": args.delta,
        "gamma": args.gamma,
        "extrinsic_only": True,
        "folds": fold_reports,
        "q_ready_gate": gate,
        "long_training_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output.parent / "q_ready_gate.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    atomic_torch_save({
        "protocol_version": "DRVOI-RHPPO-V7-WVR-1",
        "delta": args.delta,
        "gamma": args.gamma,
        "context_dim": int(train_all.context.shape[-1]),
        "folds": artifacts,
    }, args.output.with_suffix(".pt"))
    print(json.dumps(report, indent=2))
    if not gate["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
