#!/usr/bin/env python3
"""Isolated recurrent PPO backbone for DRVOI comparison experiments."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from benchmarks.benchmark_torch6dof_ppo import lowlevel_controls
from benchmarks.drvoi import multi_discrete_to_one_hot
from benchmarks.drvoi.rhppo import (
    RecurrentMultiDiscreteActorCritic,
    dual_channel_gae,
)
from benchmarks.train_torch_wvr_ppo import fixed_mixture_actions
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.utils.utils import get_root_dir
from envs.torch6dof import TorchWVRConfig, TorchWVRVecEnv


ALGORITHM = "rhppo"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=96)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--burn-in", type=int, default=20)
    parser.add_argument("--chunks-per-minibatch", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=85)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def recurrent_ppo_update(model, optimizer, rollout, args):
    advantage_ext = rollout["advantage_ext"]
    advantage_ext = (advantage_ext - advantage_ext.mean()) / advantage_ext.std(
        unbiased=False).clamp_min(1.0e-8)
    intrinsic_beta = float(getattr(args, "intrinsic_beta", 0.0))
    if intrinsic_beta > 0.0:
        advantage_int = rollout["advantage_int"]
        advantage_int = (advantage_int - advantage_int.mean()) / advantage_int.std(
            unbiased=False).clamp_min(1.0e-8)
        advantages = advantage_ext + intrinsic_beta * advantage_int
    else:
        advantages = advantage_ext
    time_steps, num_envs = advantages.shape
    starts = list(range(0, time_steps, args.sequence_length))
    totals = {key: 0.0 for key in (
        "policy_loss", "value_ext_loss", "value_int_loss", "entropy", "approx_kl",
        "clip_fraction", "grad_norm")}
    updates = 0
    for _ in range(args.epochs):
        start_order = torch.randperm(len(starts)).tolist()
        for start_index in start_order:
            start = starts[start_index]
            end = min(start + args.sequence_length, time_steps)
            env_order = torch.randperm(num_envs, device=advantages.device)
            for offset in range(0, num_envs, args.chunks_per_minibatch):
                env_ids = env_order[offset:offset + args.chunks_per_minibatch]
                warm_start = max(0, start - args.burn_in)
                hidden = rollout["hidden_before"][warm_start, env_ids]
                if warm_start < start:
                    with torch.no_grad():
                        _, hidden = model.sequence_beliefs(
                            rollout["obs"][warm_start:start, env_ids],
                            rollout["prev_action"][warm_start:start, env_ids],
                            hidden,
                            rollout["reset"][warm_start:start, env_ids])
                beliefs, _ = model.sequence_beliefs(
                    rollout["obs"][start:end, env_ids],
                    rollout["prev_action"][start:end, env_ids],
                    hidden.detach(), rollout["reset"][start:end, env_ids])
                flat_belief = beliefs.reshape(-1, model.hidden_size)
                flat_obs = rollout["obs"][start:end, env_ids].reshape(-1, 39)
                action = rollout["action"][start:end, env_ids].reshape(-1, 4)
                evaluated = model.evaluate_actions(flat_belief, action, flat_obs)
                old_log_prob = rollout["log_prob"][start:end, env_ids].reshape(-1)
                advantage = advantages[start:end, env_ids].reshape(-1)
                ratio = torch.exp(evaluated["log_prob"] - old_log_prob)
                surrogate = torch.minimum(
                    ratio * advantage,
                    ratio.clamp(1.0 - args.clip_ratio, 1.0 + args.clip_ratio)
                    * advantage)
                value_target = rollout["return_ext"][start:end, env_ids].reshape(-1)
                value_int_target = rollout["return_int"][start:end, env_ids].reshape(-1)
                policy_loss = -surrogate.mean()
                value_loss = 0.5 * (evaluated["value_ext"] - value_target).square().mean()
                value_int_loss = 0.5 * (
                    evaluated["value_int"] - value_int_target).square().mean()
                entropy = evaluated["normalized_entropy"].mean()
                loss = (policy_loss + 0.5 * value_loss
                        + 0.5 * value_int_loss - args.entropy_coef * entropy)
                with torch.no_grad():
                    log_ratio = evaluated["log_prob"] - old_log_prob
                    metric = (
                        float(policy_loss), float(value_loss), float(value_int_loss),
                        float(entropy),
                        float(((torch.exp(log_ratio) - 1.0) - log_ratio).mean()),
                        float((torch.abs(ratio - 1.0) > args.clip_ratio).float().mean()))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                for key, value in zip(
                        ("policy_loss", "value_ext_loss", "value_int_loss", "entropy",
                         "approx_kl", "clip_fraction"), metric):
                    totals[key] += value
                totals["grad_norm"] += float(grad_norm)
                updates += 1
    return {key: value / max(updates, 1) for key, value in totals.items()}


def main():
    args = parse_args()
    if args.rollout_steps % args.sequence_length != 0:
        raise ValueError("rollout-steps must be divisible by sequence-length")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    env = TorchWVRVecEnv(
        TorchWVRConfig(num_envs=args.num_envs, reward_mode="competitive",
                       attack_quality_weight=0.0),
        device=device, seed=args.seed, compile_dynamics=True)
    model = RecurrentMultiDiscreteActorCritic(
        obs_dim=env.observation_size, hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, eps=1.0e-5)
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
    hidden = model.initial_hidden(args.num_envs, device=device)
    previous_action = torch.zeros(args.num_envs, 13, device=device)
    reset = torch.ones(args.num_envs, dtype=torch.bool, device=device)
    lowlevel_hidden = torch.zeros(args.num_envs * 2, 1, 128, device=device)

    # Compile the environment before timing.
    with torch.inference_mode():
        warm = torch.zeros(args.num_envs, 4, dtype=torch.long, device=device)
        warm[:, 0] = 1
        warm[:, 1] = 2
        warm[:, 2] = 1
        all_warm = fixed_mixture_actions(obs, env_index.remainder(3))
        all_warm[env_index, learner_side] = warm
        controls, _ = lowlevel_controls(
            lowlevel, lowlevel_hidden, obs, all_warm, torch_speed_adapter=True)
        env.dynamics_runner(
            env.aircraft_state.reshape(-1, env.dynamics.state_size),
            controls, substeps=1)
    torch.cuda.synchronize()

    history = []
    total_decisions = 0
    started = time.perf_counter()
    for update in range(1, args.updates + 1):
        buffers = {key: [] for key in (
            "obs", "prev_action", "hidden_before", "reset", "action",
            "log_prob", "value_ext", "value_int", "next_value_ext",
            "next_value_int", "reward_ext", "reward_int", "done")}
        completed = wins = losses = timeouts = launches = hits = 0
        rollout_started = time.perf_counter()
        for _ in range(args.rollout_steps):
            buffers["obs"].append(learner_obs.clone())
            buffers["prev_action"].append(previous_action.clone())
            buffers["hidden_before"].append(hidden.clone())
            buffers["reset"].append(reset.clone())
            with torch.inference_mode():
                result = model.act(
                    learner_obs, previous_action, hidden, reset=reset)
                learner_action = result["action"]
                executed = fixed_mixture_actions(obs, env_index.remainder(3))
                executed[env_index, learner_side] = learner_action
                controls, next_lowlevel_hidden = lowlevel_controls(
                    lowlevel, lowlevel_hidden, obs, executed,
                    torch_speed_adapter=True)
                next_obs, reward, done, info = env.step(
                    controls.reshape(args.num_envs, 2, 4),
                    executed[..., 3] == 1)
                next_learner_obs = next_obs[env_index, learner_side]
                next_previous_action = multi_discrete_to_one_hot(learner_action)
                preview = model.act(
                    next_learner_obs, next_previous_action,
                    result["hidden"], reset=done, deterministic=True)
            learner_reward = reward[env_index, learner_side]
            for key, value in (
                    ("action", learner_action), ("log_prob", result["log_prob"]),
                    ("value_ext", result["value_ext"]),
                    ("value_int", result["value_int"]),
                    ("next_value_ext", preview["value_ext"]),
                    ("next_value_int", preview["value_int"]),
                    ("reward_ext", learner_reward),
                    ("reward_int", torch.zeros_like(learner_reward)),
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
            rollout["done"], args.gamma, args.gae_lambda)
        rollout.update(gae)
        update_started = time.perf_counter()
        metrics = recurrent_ppo_update(model, optimizer, rollout, args)
        update_s = time.perf_counter() - update_started
        decisions = args.num_envs * args.rollout_steps
        total_decisions += decisions
        entry = {
            "update": update, "environment_decisions": total_decisions,
            "completed_episodes": completed, "learner_wins": wins,
            "learner_losses": losses, "timeouts": timeouts,
            "learner_accepted_launches": launches, "learner_hits": hits,
            "mean_reward_ext": float(rollout["reward_ext"].mean()),
            "rollout_s": rollout_s, "ppo_update_s": update_s, **metrics,
        }
        history.append(entry)
        print(json.dumps(entry), flush=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    report = {
        "algorithm": ALGORITHM, "scientific_result": False,
        "environment_frozen": True, "reward_variant": "competitive",
        "training_opponents": "fixed-mix", "seed": args.seed,
        "num_envs": args.num_envs, "rollout_steps": args.rollout_steps,
        "updates": args.updates, "total_environment_decisions": total_decisions,
        "environment_decisions_per_s": total_decisions / elapsed,
        "elapsed_s": elapsed, "sequence_length": args.sequence_length,
        "burn_in": args.burn_in, "environment_config": asdict(env.config),
        "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "algorithm": ALGORITHM, "model": model.state_dict(),
            "obs_dim": env.observation_size, "hidden_size": args.hidden_size,
            "action_dims": [3, 5, 3, 2], "seed": args.seed,
            "training_environment_decisions": total_decisions,
            "environment_config": asdict(env.config),
        }, args.checkpoint)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
