#!/usr/bin/env python3
"""Online VOI-only DRVOI-RHPPO (O0) for the frozen Torch WVR platform."""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch

from benchmarks.benchmark_torch6dof_ppo import lowlevel_controls
from benchmarks.drvoi import (
    cross_fitted_voi,
    extract_response_state,
    multi_discrete_to_one_hot,
)
from benchmarks.drvoi.q_model import ActionEmbeddingQ, double_q_target, q_huber_loss
from benchmarks.drvoi.response import ResponseEnsemble, member_log_likelihood, mixture_nll
from benchmarks.drvoi.rhppo import RecurrentMultiDiscreteActorCritic, dual_channel_gae
from benchmarks.train_torch_wvr_ppo import fixed_mixture_actions
from benchmarks.train_torch_wvr_rhppo import recurrent_ppo_update
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.utils.utils import get_root_dir
from envs.torch6dof import TorchWVRConfig, TorchWVRVecEnv


ALGORITHM = "drvoi_rhppo"
METHOD = "O0-VOI-only"


class RunningRMS:
    def __init__(self, device):
        self.count = torch.zeros((), device=device)
        self.sum_squares = torch.zeros((), device=device)

    def normalize(self, values: torch.Tensor, update: bool = True):
        if update:
            self.count += values.numel()
            self.sum_squares += values.square().sum()
        rms = torch.sqrt(self.sum_squares / self.count.clamp_min(1.0)).clamp_min(1.0e-6)
        return (values / rms).clamp(0.0, 5.0) / 5.0, rms


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rhppo-checkpoint", type=Path, required=True)
    parser.add_argument("--response-checkpoint", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=96)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--burn-in", type=int, default=20)
    parser.add_argument("--chunks-per-minibatch", type=int, default=16)
    parser.add_argument("--q-warmup-updates", type=int, default=2)
    parser.add_argument("--q-gradient-updates", type=int, default=8)
    parser.add_argument("--q-batch-size", type=int, default=2048)
    parser.add_argument("--q-replay-capacity", type=int, default=100000)
    parser.add_argument("--response-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--intrinsic-beta", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=85)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def append_replay(replay: dict[str, torch.Tensor] | None,
                  batch: dict[str, torch.Tensor], capacity: int):
    if replay is None:
        replay = {key: value.detach() for key, value in batch.items()}
    else:
        replay = {key: torch.cat((replay[key], value.detach()))[-capacity:]
                  for key, value in batch.items()}
    return replay


def update_response(model, optimizer, belief, action, delta, side):
    optimizer.zero_grad(set_to_none=True)
    losses = []
    count = belief.shape[0]
    for member in model.members:
        indices = torch.randint(
            count, (max(int(0.8 * count), 1),), device=belief.device)
        mean, variance, logits = member(belief[indices], action[indices])
        log_likelihood = member_log_likelihood(
            mean[None], variance[None], logits[None],
            delta[indices], side[indices])[0]
        losses.append(-log_likelihood.mean())
    loss = torch.stack(losses).mean()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    return float(loss.detach()), float(grad_norm)


def update_q(model, target, optimizer, replay, batch_size, updates, gamma=0.99):
    if replay is None or replay["context"].shape[0] < batch_size:
        return {"loss": float("nan"), "abs_td": float("nan")}
    losses = []
    td_values = []
    count = replay["context"].shape[0]
    for _ in range(updates):
        indices = torch.randint(count, (batch_size,), device=replay["context"].device)
        with torch.no_grad():
            target_value = double_q_target(
                replay["reward"][indices], replay["done"][indices],
                replay["next_context"][indices], model, target, gamma)
        prediction = model(
            replay["context"][indices], replay["next_action"][indices])
        loss = q_huber_loss(
            model, replay["context"][indices], replay["next_action"][indices],
            target_value)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        td_values.append(float((target_value - prediction).abs().mean().detach()))
    return {"loss": sum(losses) / len(losses),
            "abs_td": sum(td_values) / len(td_values)}


def main():
    args = parse_args()
    if args.rollout_steps % args.sequence_length != 0:
        raise ValueError("rollout-steps must be divisible by sequence-length")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rhppo_payload = torch.load(
        args.rhppo_checkpoint, map_location=device, weights_only=False)
    policy = RecurrentMultiDiscreteActorCritic(
        obs_dim=int(rhppo_payload["obs_dim"]),
        hidden_size=int(rhppo_payload["hidden_size"])).to(device)
    policy.load_state_dict(rhppo_payload["model"])
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4, eps=1.0e-5)
    response_payload = torch.load(
        args.response_checkpoint, map_location=device, weights_only=False)
    response = ResponseEnsemble(
        belief_dim=int(response_payload["belief_dim"]), members=5,
        hidden_size=256).to(device)
    response.load_state_dict(response_payload["conditioned"])
    delayed_response = response.delayed_snapshot()
    response_optimizer = torch.optim.Adam(
        response.parameters(), lr=args.response_learning_rate)
    variance_scale = response_payload["variance_scale"].to(device)
    side_temperature = float(response_payload["side_temperature"])
    context_dim = policy.hidden_size + 13 + 12 + 3
    q_a = ActionEmbeddingQ(context_dim).to(device)
    q_b = ActionEmbeddingQ(context_dim).to(device)
    target_a = copy.deepcopy(q_a).eval().requires_grad_(False)
    target_b = copy.deepcopy(q_b).eval().requires_grad_(False)
    optimizer_a = torch.optim.Adam(q_a.parameters(), lr=3.0e-4)
    optimizer_b = torch.optim.Adam(q_b.parameters(), lr=3.0e-4)
    replay_a = replay_b = None
    voi_rms = RunningRMS(device)

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
    learner_obs = obs[env_index, learner_side]
    hidden = policy.initial_hidden(args.num_envs, device=device)
    previous_action = torch.zeros(args.num_envs, 13, device=device)
    reset = torch.ones(args.num_envs, dtype=torch.bool, device=device)
    lowlevel_hidden = torch.zeros(args.num_envs * 2, 1, 128, device=device)
    pending_context = torch.zeros(args.num_envs, context_dim, device=device)
    pending_valid = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    episode_id = torch.arange(args.num_envs, device=device, dtype=torch.long)
    pending_episode = episode_id.clone()
    next_episode_id = args.num_envs
    history = []
    total_decisions = 0
    started = time.perf_counter()

    ppo_args = SimpleNamespace(
        sequence_length=args.sequence_length, burn_in=args.burn_in,
        chunks_per_minibatch=args.chunks_per_minibatch,
        epochs=args.ppo_epochs, clip_ratio=0.2, entropy_coef=0.01,
        intrinsic_beta=0.0)

    for update in range(1, args.updates + 1):
        buffers = {key: [] for key in (
            "obs", "prev_action", "hidden_before", "reset", "action",
            "log_prob", "value_ext", "value_int", "next_value_ext",
            "next_value_int", "reward_ext", "reward_int", "done")}
        response_batch = {key: [] for key in ("belief", "action", "delta", "side")}
        q_batches = {0: {key: [] for key in (
            "context", "next_action", "reward", "next_context", "done")},
                     1: {key: [] for key in (
            "context", "next_action", "reward", "next_context", "done")}}
        signed_voi_values = []
        positive_voi_values = []
        q_candidate_diversity = []
        q_fold_gaps = []
        response_nll_values = []
        completed = wins = losses = timeouts = launches = hits = 0
        rollout_started = time.perf_counter()
        for _ in range(args.rollout_steps):
            buffers["obs"].append(learner_obs.clone())
            buffers["prev_action"].append(previous_action.clone())
            buffers["hidden_before"].append(hidden.clone())
            buffers["reset"].append(reset.clone())
            with torch.inference_mode():
                result = policy.act(
                    learner_obs, previous_action, hidden, reset=reset)
                learner_action = result["action"]
                action_one_hot = multi_discrete_to_one_hot(learner_action)
                current_response, _ = extract_response_state(learner_obs)
                if update > args.q_warmup_updates:
                    candidates = response.sample_candidates(
                        result["belief"], action_one_hot, current_response,
                        samples_per_member=4,
                        variance_scale=variance_scale,
                        side_temperature=side_temperature)
                    candidate_side = torch.nn.functional.one_hot(
                        candidates["side"], 3).float()
                    candidate_context = torch.cat((
                        result["belief"][:, None].expand(-1, 20, -1),
                        action_one_hot[:, None].expand(-1, 20, -1),
                        candidates["continuous"], candidate_side), dim=-1)
                    flat_context = candidate_context.reshape(-1, context_dim)
                    values_a = q_a.all_actions(flat_context).reshape(
                        args.num_envs, 20, 90)
                    values_b = q_b.all_actions(flat_context).reshape(
                        args.num_envs, 20, 90)
                    q_candidate_diversity.append(
                        (values_a.argmax(-1).amax(-1)
                         != values_a.argmax(-1).amin(-1)).float())
                    q_fold_gaps.append((values_a - values_b).abs().mean((1, 2)))
                    voi = cross_fitted_voi(
                        values_a, values_b, candidates["weights"])
                    reward_int, rms = voi_rms.normalize(voi["positive_voi"])
                    signed_voi_values.append(voi["signed_voi"])
                    positive_voi_values.append(voi["positive_voi"])
                else:
                    reward_int = torch.zeros(args.num_envs, device=device)
                    rms = torch.ones((), device=device)
                executed = fixed_mixture_actions(obs, env_index.remainder(3))
                executed[env_index, learner_side] = learner_action
                controls, next_lowlevel_hidden = lowlevel_controls(
                    lowlevel, lowlevel_hidden, obs, executed,
                    torch_speed_adapter=True)
                next_obs, reward, done, info = env.step(
                    controls.reshape(args.num_envs, 2, 4),
                    executed[..., 3] == 1)
                next_learner_obs = next_obs[env_index, learner_side]
                next_response, next_side = extract_response_state(next_learner_obs)
                delta_response = next_response - current_response
                valid_response = ~done
                if bool(valid_response.any()):
                    response_nll_values.append(mixture_nll(
                        response, result["belief"][valid_response],
                        action_one_hot[valid_response],
                        delta_response[valid_response], next_side[valid_response]))
                actual_context = torch.cat((
                    result["belief"], action_one_hot, next_response,
                    torch.nn.functional.one_hot(next_side, 3).float()), dim=-1)
                next_previous_action = action_one_hot
                preview = policy.act(
                    next_learner_obs, next_previous_action,
                    result["hidden"], reset=done, deterministic=True)
            learner_reward = reward[env_index, learner_side]
            valid_indices = torch.nonzero(pending_valid, as_tuple=False).flatten()
            if valid_indices.numel() > 0:
                folds = pending_episode[valid_indices].remainder(2)
                for fold in (0, 1):
                    selected = valid_indices[folds == fold]
                    if selected.numel() == 0:
                        continue
                    q_batches[fold]["context"].append(pending_context[selected])
                    q_batches[fold]["next_action"].append(action_one_hot[selected])
                    q_batches[fold]["reward"].append(learner_reward[selected])
                    q_batches[fold]["next_context"].append(actual_context[selected])
                    q_batches[fold]["done"].append(done[selected])
            if bool(valid_response.any()):
                response_batch["belief"].append(
                    result["belief"][valid_response].detach())
                response_batch["action"].append(action_one_hot[valid_response])
                response_batch["delta"].append(delta_response[valid_response])
                response_batch["side"].append(next_side[valid_response])
            for key, value in (
                    ("action", learner_action), ("log_prob", result["log_prob"]),
                    ("value_ext", result["value_ext"]),
                    ("value_int", result["value_int"]),
                    ("next_value_ext", preview["value_ext"]),
                    ("next_value_int", preview["value_int"]),
                    ("reward_ext", learner_reward), ("reward_int", reward_int),
                    ("done", done)):
                buffers[key].append(value.clone())
            outcome = torch.where(
                learner_side == 0, info["outcome_red"], -info["outcome_red"])
            completed += int(done.sum())
            wins += int(((outcome > 0) & done).sum())
            losses += int(((outcome < 0) & done).sum())
            timeouts += int(info["timeout"].sum())
            launches += int(info["accepted_launches"][env_index, learner_side].sum())
            hits += int(info["hits"][env_index, learner_side].sum())
            pending_context = actual_context.detach()
            pending_episode = episode_id.clone()
            pending_valid = ~done
            if bool(done.any()):
                done_indices = torch.nonzero(done, as_tuple=False).flatten()
                episode_id[done_indices] = torch.arange(
                    next_episode_id, next_episode_id + done_indices.numel(),
                    device=device)
                next_episode_id += done_indices.numel()
            hidden = result["hidden"]
            previous_action = next_previous_action
            reset = done
            obs = next_obs
            learner_obs = next_learner_obs
            lowlevel_hidden = torch.where(
                done.repeat_interleave(2)[:, None, None],
                torch.zeros_like(next_lowlevel_hidden), next_lowlevel_hidden)
        rollout_s = time.perf_counter() - rollout_started
        rollout = {key: torch.stack(value) for key, value in buffers.items()}
        gae = dual_channel_gae(
            rollout["reward_ext"], rollout["reward_int"],
            rollout["value_ext"], rollout["value_int"],
            rollout["next_value_ext"], rollout["next_value_int"],
            rollout["done"], 0.99, 0.95)
        rollout.update(gae)
        ppo_args.intrinsic_beta = (
            0.0 if update <= args.q_warmup_updates else args.intrinsic_beta)
        update_started = time.perf_counter()
        ppo_metrics = recurrent_ppo_update(
            policy, policy_optimizer, rollout, ppo_args)
        response_flat = {key: torch.cat(value) for key, value in response_batch.items()}
        response_loss, response_grad = update_response(
            response, response_optimizer, response_flat["belief"],
            response_flat["action"], response_flat["delta"],
            response_flat["side"])
        if update % 5 == 0:
            delayed_response = response.delayed_snapshot()
        for fold in (0, 1):
            if q_batches[fold]["context"]:
                batch = {key: torch.cat(value) for key, value in q_batches[fold].items()}
                if fold == 0:
                    replay_a = append_replay(replay_a, batch, args.q_replay_capacity)
                else:
                    replay_b = append_replay(replay_b, batch, args.q_replay_capacity)
        q_metrics_a = update_q(
            q_a, target_a, optimizer_a, replay_a, args.q_batch_size,
            args.q_gradient_updates)
        q_metrics_b = update_q(
            q_b, target_b, optimizer_b, replay_b, args.q_batch_size,
            args.q_gradient_updates)
        if update % 10 == 0:
            target_a.load_state_dict(q_a.state_dict())
            target_b.load_state_dict(q_b.state_dict())
        update_s = time.perf_counter() - update_started
        decisions = args.num_envs * args.rollout_steps
        total_decisions += decisions
        signed = torch.cat(signed_voi_values) if signed_voi_values else torch.zeros(1)
        positive = torch.cat(positive_voi_values) if positive_voi_values else torch.zeros(1)
        diversity = torch.cat(q_candidate_diversity) if q_candidate_diversity else torch.zeros(1)
        fold_gap = torch.cat(q_fold_gaps) if q_fold_gaps else torch.zeros(1)
        entry = {
            "update": update, "environment_decisions": total_decisions,
            "completed_episodes": completed, "learner_wins": wins,
            "learner_losses": losses, "timeouts": timeouts,
            "accepted_launches": launches, "hits": hits,
            "mean_reward_ext": float(rollout["reward_ext"].mean()),
            "mean_reward_int": float(rollout["reward_int"].mean()),
            "intrinsic_beta": ppo_args.intrinsic_beta,
            "voi_valid_fraction": float(update > args.q_warmup_updates),
            "signed_voi_mean": float(signed.mean()),
            "signed_voi_negative_fraction": float((signed < 0).float().mean()),
            "positive_voi_mean": float(positive.mean()),
            "positive_voi_nonzero_fraction": float((positive > 1.0e-6).float().mean()),
            "q_candidate_action_diversity_fraction": float(diversity.mean()),
            "q_fold_mean_absolute_gap": float(fold_gap.mean()),
            "voi_rms": float(rms),
            "prequential_response_nll": float(torch.cat(response_nll_values).mean()),
            "response_loss": response_loss, "response_grad_norm": response_grad,
            "q_a_loss": q_metrics_a["loss"], "q_b_loss": q_metrics_b["loss"],
            "q_a_abs_td": q_metrics_a["abs_td"], "q_b_abs_td": q_metrics_b["abs_td"],
            "q_a_samples": 0 if replay_a is None else int(replay_a["context"].shape[0]),
            "q_b_samples": 0 if replay_b is None else int(replay_b["context"].shape[0]),
            "rollout_s": rollout_s, "update_s": update_s, **ppo_metrics,
        }
        history.append(entry)
        print(json.dumps(entry), flush=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    report = {
        "algorithm": ALGORITHM, "method": METHOD, "scientific_result": False,
        "environment_frozen": True, "causal_ordering": {
            "voi_pre_step": True, "response_prequential": True,
            "q_extrinsic_only": True, "beta_applied_once": True},
        "seed": args.seed, "num_envs": args.num_envs,
        "updates": args.updates, "rollout_steps": args.rollout_steps,
        "total_environment_decisions": total_decisions,
        "elapsed_s": elapsed, "environment_decisions_per_s": total_decisions / elapsed,
        "environment_config": asdict(env.config), "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "algorithm": ALGORITHM, "method": METHOD,
            "policy": policy.state_dict(), "response": response.state_dict(),
            "delayed_response": delayed_response.state_dict(),
            "q_a": q_a.state_dict(), "q_b": q_b.state_dict(),
            "target_a": target_a.state_dict(), "target_b": target_b.state_dict(),
            "variance_scale": variance_scale,
            "side_temperature": side_temperature,
            "obs_dim": env.observation_size, "hidden_size": policy.hidden_size,
            "training_environment_decisions": total_decisions,
            "environment_config": asdict(env.config),
        }, args.checkpoint)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
