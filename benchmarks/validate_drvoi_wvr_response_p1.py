#!/usr/bin/env python3
"""Validate G1/G2 on legal WVR observations before online DRVOI training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from benchmarks.benchmark_torch6dof_ppo import lowlevel_controls
from benchmarks.drvoi import extract_response_state, multi_discrete_to_one_hot
from benchmarks.drvoi.response import (
    ResponseEnsemble,
    member_log_likelihood,
    mixture_nll,
)
from benchmarks.drvoi.rhppo import RecurrentMultiDiscreteActorCritic
from benchmarks.train_torch_wvr_ppo import fixed_mixture_actions
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.utils.utils import get_root_dir
from envs.torch6dof import TorchWVRConfig, TorchWVRVecEnv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rhppo-checkpoint", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=96)
    parser.add_argument("--collection-steps", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=6185)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def ece(probabilities: torch.Tensor, targets: torch.Tensor, bins: int = 15) -> float:
    confidence, prediction = probabilities.max(-1)
    correct = (prediction == targets).float()
    result = torch.zeros((), device=probabilities.device)
    boundaries = torch.linspace(0.0, 1.0, bins + 1, device=probabilities.device)
    for index in range(bins):
        selected = (confidence > boundaries[index]) & (
            confidence <= boundaries[index + 1])
        if bool(selected.any()):
            result += selected.float().mean() * torch.abs(
                correct[selected].mean() - confidence[selected].mean())
    return float(result)


def train_ensemble(model: ResponseEnsemble, belief: torch.Tensor,
                   action: torch.Tensor, delta: torch.Tensor,
                   side: torch.Tensor, episode: torch.Tensor,
                   epochs: int, batch_size: int, seed: int) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=3.0e-4)
    generator = torch.Generator(device=belief.device).manual_seed(seed)
    losses = []
    unique_episodes = torch.unique(episode)
    episode_samples = [torch.nonzero(
        episode == episode_id, as_tuple=False).squeeze(-1)
        for episode_id in unique_episodes]
    for _ in range(epochs):
        epoch_losses = []
        # V6 requires an 80% bootstrap over complete episodes, not individual
        # transitions.  A sampled episode may occur more than once.
        sampled_episode_count = max(int(0.8 * len(episode_samples)), 1)
        member_indices = []
        for _ in model.members:
            sampled = torch.randint(
                len(episode_samples), (sampled_episode_count,),
                device=belief.device, generator=generator)
            member_indices.append(torch.cat([
                episode_samples[int(index)] for index in sampled]))
        batches = math.ceil(member_indices[0].numel() / batch_size)
        for batch in range(batches):
            optimizer.zero_grad(set_to_none=True)
            member_losses = []
            for member, indices in zip(model.members, member_indices):
                selected = indices[batch * batch_size:(batch + 1) * batch_size]
                if selected.numel() == 0:
                    continue
                mean, variance, logits = member(belief[selected], action[selected])
                log_likelihood = member_log_likelihood(
                    mean[None], variance[None], logits[None],
                    delta[selected], side[selected])[0]
                member_losses.append(-log_likelihood.mean())
            loss = torch.stack(member_losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_losses)))
    return losses


def stratified_episode_split(episode: torch.Tensor, opponent: torch.Tensor,
                             seed: int):
    """Return train/calibration/test indices with no episode leakage."""
    generator = torch.Generator(device=episode.device).manual_seed(seed)
    split_episodes = {key: [] for key in ("train", "calibration", "test")}
    for opponent_id in torch.unique(opponent).tolist():
        selected = torch.unique(episode[opponent == opponent_id])
        selected = selected[torch.randperm(
            selected.numel(), device=episode.device, generator=generator)]
        count = selected.numel()
        train_count = max(1, int(0.70 * count))
        calibration_count = max(1, int(0.15 * count))
        if train_count + calibration_count >= count:
            train_count = max(1, count - 2)
            calibration_count = 1
        split_episodes["train"].append(selected[:train_count])
        split_episodes["calibration"].append(
            selected[train_count:train_count + calibration_count])
        split_episodes["test"].append(
            selected[train_count + calibration_count:])
    result = {}
    for key, values in split_episodes.items():
        selected_episodes = torch.cat(values)
        result[key] = torch.nonzero(
            torch.isin(episode, selected_episodes), as_tuple=False).squeeze(-1)
    return result, {key: torch.cat(value) for key, value in split_episodes.items()}


def phase_bins(episode: torch.Tensor, episode_step: torch.Tensor) -> torch.Tensor:
    """Discretize each episode into early/middle/late thirds."""
    result = torch.zeros_like(episode_step)
    for episode_id in torch.unique(episode):
        selected = episode == episode_id
        length = episode_step[selected].max() + 1
        result[selected] = torch.clamp(
            (3 * episode_step[selected]) // length.clamp_min(1), max=2)
    return result


def stratified_action_permutation(action: torch.Tensor, opponent: torch.Tensor,
                                  phase: torch.Tensor, seed: int):
    """Permute actions only within the same opponent and episode phase."""
    generator = torch.Generator(device=action.device).manual_seed(seed)
    result = action.clone()
    for opponent_id in torch.unique(opponent):
        for phase_id in torch.unique(phase):
            selected = torch.nonzero(
                (opponent == opponent_id) & (phase == phase_id),
                as_tuple=False).squeeze(-1)
            if selected.numel() > 1:
                permutation = torch.randperm(
                    selected.numel(), device=action.device, generator=generator)
                result[selected] = action[selected[permutation]]
    return result


def episode_bootstrap_ci(values: torch.Tensor, episode: torch.Tensor,
                         seed: int, samples: int = 2000):
    """Paired, equal-episode-weight bootstrap interval and mean."""
    unique_episodes = torch.unique(episode)
    episode_means = torch.stack([
        values[episode == episode_id].mean() for episode_id in unique_episodes])
    array = episode_means.detach().cpu().numpy()
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray([
        rng.choice(array, size=len(array), replace=True).mean()
        for _ in range(samples)])
    return float(array.mean()), np.quantile(bootstrap, [0.025, 0.975]).tolist()


def zero_action_input_weights(model: ResponseEnsemble):
    """Make action/no-action models functionally identical at initialization."""
    with torch.no_grad():
        for member in model.members:
            member.network[0].weight[:, -13:].zero_()


@torch.no_grad()
def predictions(model: ResponseEnsemble, belief: torch.Tensor,
                action: torch.Tensor, variance_scale: torch.Tensor | None = None,
                temperature: float = 1.0):
    means, variances, logits = model(belief, action)
    mixture_mean = means.mean(0)
    total_variance = (variances + means.square()).mean(0) - mixture_mean.square()
    if variance_scale is not None:
        total_variance = total_variance * variance_scale
    side_probabilities = (logits / temperature).softmax(-1).mean(0)
    return mixture_mean, total_variance.clamp_min(1.0e-8), side_probabilities


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    payload = torch.load(args.rhppo_checkpoint, map_location=device, weights_only=False)
    policy = RecurrentMultiDiscreteActorCritic(
        obs_dim=int(payload["obs_dim"]), hidden_size=int(payload["hidden_size"])).to(device)
    policy.load_state_dict(payload["model"])
    policy.eval().requires_grad_(False)
    env = TorchWVRVecEnv(
        TorchWVRConfig(num_envs=args.num_envs, reward_mode="competitive",
                       attack_quality_weight=0.0),
        device=device, seed=args.seed, compile_dynamics=True)
    lowlevel = BaselineActor().to(device).eval()
    lowlevel.load_state_dict(torch.load(
        get_root_dir() + "/model/baseline_model.pt",
        map_location=device, weights_only=True))
    lowlevel.requires_grad_(False)
    lowlevel.tpdv = {"dtype": torch.float32, "device": device}
    env_index = torch.arange(args.num_envs, device=device)
    learner_side = env_index.remainder(2)
    obs = env.observations()
    hidden = policy.initial_hidden(args.num_envs, device=device)
    previous_action = torch.zeros(args.num_envs, 13, device=device)
    reset = torch.ones(args.num_envs, dtype=torch.bool, device=device)
    lowlevel_hidden = torch.zeros(args.num_envs * 2, 1, 128, device=device)
    collected = {key: [] for key in (
        "belief", "action", "delta", "side", "episode", "opponent",
        "episode_step")}
    episode_generation = torch.zeros(
        args.num_envs, dtype=torch.long, device=device)
    episode_step = torch.zeros(args.num_envs, dtype=torch.long, device=device)
    with torch.inference_mode():
        for _ in range(args.collection_steps):
            learner_obs = obs[env_index, learner_side]
            result = policy.act(learner_obs, previous_action, hidden, reset=reset)
            learner_action = result["action"]
            executed = fixed_mixture_actions(obs, env_index.remainder(3))
            executed[env_index, learner_side] = learner_action
            controls, next_lowlevel_hidden = lowlevel_controls(
                lowlevel, lowlevel_hidden, obs, executed,
                torch_speed_adapter=True)
            next_obs, _, done, _ = env.step(
                controls.reshape(args.num_envs, 2, 4), executed[..., 3] == 1)
            next_learner_obs = next_obs[env_index, learner_side]
            current_response, _ = extract_response_state(learner_obs)
            next_response, next_side = extract_response_state(next_learner_obs)
            valid = ~done
            collected["belief"].append(result["belief"][valid].clone())
            collected["action"].append(
                multi_discrete_to_one_hot(learner_action)[valid])
            collected["delta"].append(
                (next_response - current_response)[valid])
            collected["side"].append(next_side[valid])
            episode_id = episode_generation * args.num_envs + env_index
            collected["episode"].append(episode_id[valid].clone())
            collected["opponent"].append(env_index.remainder(3)[valid].clone())
            collected["episode_step"].append(episode_step[valid].clone())
            hidden = result["hidden"]
            previous_action = multi_discrete_to_one_hot(learner_action)
            reset = done
            obs = next_obs
            lowlevel_hidden = torch.where(
                done.repeat_interleave(2)[:, None, None],
                torch.zeros_like(next_lowlevel_hidden), next_lowlevel_hidden)
            episode_generation = episode_generation + done.long()
            episode_step = torch.where(done, torch.zeros_like(episode_step),
                                       episode_step + 1)
    data = {key: torch.cat(value) for key, value in collected.items()}
    count = data["belief"].shape[0]
    data["phase"] = phase_bins(data["episode"], data["episode_step"])
    splits, split_episodes = stratified_episode_split(
        data["episode"], data["opponent"], args.seed + 1)
    conditioned = ResponseEnsemble(
        belief_dim=policy.hidden_size, members=5, hidden_size=256).to(device)
    zero_action_input_weights(conditioned)
    no_action = ResponseEnsemble(
        belief_dim=policy.hidden_size, members=5, hidden_size=256).to(device)
    no_action.load_state_dict(conditioned.state_dict())
    with torch.no_grad():
        initial_conditioned = conditioned(
            data["belief"][:64], data["action"][:64])
        initial_no_action = no_action(
            data["belief"][:64], torch.zeros_like(data["action"][:64]))
        initial_max_difference = max(float((left - right).abs().max())
                                     for left, right in zip(
                                         initial_conditioned, initial_no_action))
    train_index = splits["train"]
    conditioned_losses = train_ensemble(
        conditioned, data["belief"][train_index], data["action"][train_index],
        data["delta"][train_index], data["side"][train_index],
        data["episode"][train_index],
        args.epochs, args.batch_size, args.seed)
    no_action_losses = train_ensemble(
        no_action, data["belief"][train_index],
        torch.zeros_like(data["action"][train_index]),
        data["delta"][train_index], data["side"][train_index],
        data["episode"][train_index],
        args.epochs, args.batch_size, args.seed)

    calibration = splits["calibration"]
    calibration_mean, calibration_variance, _ = predictions(
        conditioned, data["belief"][calibration], data["action"][calibration])
    standardized_error = (
        (data["delta"][calibration] - calibration_mean).abs()
        / calibration_variance.sqrt())
    calibration_quantile = torch.quantile(
        standardized_error, 0.90, dim=0)
    variance_scale = (
        calibration_quantile / 1.6448536269514722).square().clamp(0.05, 20.0)
    best_temperature = 1.0
    best_side_nll = float("inf")
    best_side_ece = float("inf")
    for temperature in torch.logspace(
            math.log10(0.01), math.log10(20.0), 240).tolist():
        _, _, probabilities = predictions(
            conditioned, data["belief"][calibration], data["action"][calibration],
            variance_scale, temperature)
        nll = float(torch.nn.functional.nll_loss(
            probabilities.clamp_min(1.0e-8).log(), data["side"][calibration]))
        calibration_ece = ece(probabilities, data["side"][calibration])
        if (calibration_ece < best_side_ece
                or (calibration_ece == best_side_ece and nll < best_side_nll)):
            best_side_ece = calibration_ece
            best_side_nll = nll
            best_temperature = float(temperature)

    test = splits["test"]
    conditioned_nll = mixture_nll(
        conditioned, data["belief"][test], data["action"][test],
        data["delta"][test], data["side"][test])
    no_action_nll = mixture_nll(
        no_action, data["belief"][test], torch.zeros_like(data["action"][test]),
        data["delta"][test], data["side"][test])
    delta_nll = no_action_nll - conditioned_nll
    mean_delta_nll, delta_ci = episode_bootstrap_ci(
        delta_nll, data["episode"][test], args.seed + 2)
    permuted_action = stratified_action_permutation(
        data["action"][test], data["opponent"][test],
        data["phase"][test], args.seed + 3)
    permuted_nll = mixture_nll(
        conditioned, data["belief"][test], permuted_action,
        data["delta"][test], data["side"][test])
    permutation_gain = permuted_nll - conditioned_nll
    mean_permutation_gain, permutation_ci = episode_bootstrap_ci(
        permutation_gain, data["episode"][test], args.seed + 4)
    mean, variance, side_probs = predictions(
        conditioned, data["belief"][test], data["action"][test],
        variance_scale, best_temperature)
    residual = data["delta"][test] - mean
    coverage90 = float((
        residual.abs() <= 1.6448536269514722 * variance.sqrt()).float().mean())
    rmse = float(residual.square().mean().sqrt())
    side_ece = ece(side_probs, data["side"][test])
    side_one_hot = torch.nn.functional.one_hot(data["side"][test], 3).float()
    side_brier = float((side_probs - side_one_hot).square().sum(-1).mean())
    opponent_names = {0: "hold", 1: "pursuit", 2: "evasive"}
    phase_names = {0: "early", 1: "middle", 2: "late"}
    stratified_delta_nll = {}
    for opponent_id, opponent_name in opponent_names.items():
        stratified_delta_nll[opponent_name] = {}
        for phase_id, phase_name in phase_names.items():
            selected = ((data["opponent"][test] == opponent_id)
                        & (data["phase"][test] == phase_id))
            stratified_delta_nll[opponent_name][phase_name] = (
                float(delta_nll[selected].mean()) if bool(selected.any()) else None)
    side_accuracy = float((side_probs.argmax(-1) == data["side"][test]).float().mean())
    side_mean_confidence = float(side_probs.max(-1).values.mean())
    side_prediction = side_probs.argmax(-1)
    f1_values = []
    for side_class in range(3):
        true_positive = ((side_prediction == side_class)
                         & (data["side"][test] == side_class)).sum().float()
        false_positive = ((side_prediction == side_class)
                          & (data["side"][test] != side_class)).sum().float()
        false_negative = ((side_prediction != side_class)
                          & (data["side"][test] == side_class)).sum().float()
        denominator = 2.0 * true_positive + false_positive + false_negative
        f1_values.append(float(
            2.0 * true_positive / denominator.clamp_min(1.0)))
    side_class_frequency = torch.bincount(
        data["side"][test], minlength=3).float()
    side_class_frequency = (
        side_class_frequency / side_class_frequency.sum()).cpu().tolist()
    gates = {
        "G1_action_conditioning": {
            "test_samples": int(test.numel()),
            "test_episodes": int(torch.unique(data["episode"][test]).numel()),
            "mean_delta_nll": mean_delta_nll,
            "bootstrap_ci95": delta_ci,
            "bootstrap_unit": "episode",
            "stratified_delta_nll": stratified_delta_nll,
            "action_permutation_gain": mean_permutation_gain,
            "action_permutation_ci95": permutation_ci,
            "action_permutation_negative_control_pass": (
                mean_permutation_gain > 0.0 and permutation_ci[0] > 0.0),
            "pass": int(test.numel()) >= 1000 and mean_delta_nll >= 0.05
                    and delta_ci[0] > 0.0 and mean_permutation_gain > 0.0
                    and permutation_ci[0] > 0.0,
        },
        "G2_probability_calibration": {
            "continuous_90_coverage": coverage90,
            "continuous_rmse": rmse,
            "side_ece": side_ece,
            "side_brier": side_brier,
            "side_accuracy": side_accuracy,
            "side_mean_confidence": side_mean_confidence,
            "side_class_frequency": side_class_frequency,
            "side_f1_by_class": f1_values,
            "side_macro_f1": float(np.mean(f1_values)),
            "variance_scale_by_dimension": variance_scale.cpu().tolist(),
            "side_temperature": best_temperature,
            "calibration_side_ece": best_side_ece,
            "calibration_side_nll": best_side_nll,
            "pass": 0.85 <= coverage90 <= 0.95 and side_ece <= 0.05,
        },
    }
    report = {
        "suite": "DRVOI-RHPPO-V6-WVR-response-P1",
        "scientific_result": False,
        "context": "frozen RHPPO belief state",
        "checkpoint": str(args.rhppo_checkpoint),
        "seed": args.seed, "samples": count,
        "split_sizes": {key: int(value.numel()) for key, value in splits.items()},
        "split_episode_counts": {
            key: int(value.numel()) for key, value in split_episodes.items()},
        "training": {
            "epochs": args.epochs,
            "bootstrap_unit": "episode",
            "initial_function_max_difference": initial_max_difference,
            "conditioned_loss_initial_final": [
                conditioned_losses[0], conditioned_losses[-1]],
            "no_action_loss_initial_final": [
                no_action_losses[0], no_action_losses[-1]],
        },
        "gates": gates,
        "all_gates_pass": all(item["pass"] for item in gates.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    torch.save({
        "conditioned": conditioned.state_dict(),
        "no_action": no_action.state_dict(),
        "variance_scale": variance_scale,
        "side_temperature": best_temperature,
        "belief_dim": policy.hidden_size,
    }, args.output.with_suffix(".pt"))
    print(json.dumps(report, indent=2))
    if not report["all_gates_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
