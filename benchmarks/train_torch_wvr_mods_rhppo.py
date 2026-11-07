#!/usr/bin/env python3
"""Online masked macro-option soft-VOI RHPPO development trainer."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from benchmarks.benchmark_torch6dof_ppo import lowlevel_controls
from benchmarks.drvoi import (
    extract_response_state,
    masked_soft_cross_fitted_voi,
    multi_discrete_to_one_hot,
)
from benchmarks.drvoi.checkpointing import atomic_torch_save
from benchmarks.drvoi.q_model import MacroOptionDuelingQ
from benchmarks.drvoi.response import ContinuousResponseEnsemble
from benchmarks.drvoi.rhppo import (
    RecurrentMultiDiscreteActorCritic,
    dual_channel_gae,
)
from benchmarks.drvoi_v7_1.history import VectorHistoryBuffer
from benchmarks.drvoi_v7_1.macro_options import macro_option_legal_mask
from benchmarks.train_torch_wvr_drvoi_rhppo import RunningRMS
from benchmarks.train_torch_wvr_ppo import fixed_mixture_actions
from benchmarks.train_torch_wvr_rhppo import recurrent_ppo_update
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.utils.utils import get_root_dir
from envs.torch6dof import TorchWVRConfig, TorchWVRVecEnv


ALGORITHM = "mods_drvoi_rhppo"
METHOD = "masked_macro_option_soft_voi"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rhppo-checkpoint", type=Path, required=True)
    parser.add_argument("--response-checkpoint", type=Path, required=True)
    parser.add_argument("--option-q-checkpoint", type=Path, required=True)
    parser.add_argument("--response-fold", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=96)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--burn-in", type=int, default=20)
    parser.add_argument("--chunks-per-minibatch", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--samples-per-member", type=int, default=4)
    parser.add_argument("--soft-voi-temperature", type=float, default=0.05)
    parser.add_argument("--soft-voi-top-k", type=int, default=3)
    parser.add_argument("--intrinsic-beta", type=float, default=0.05)
    parser.add_argument("--intrinsic-beta-final", type=float, default=0.005)
    parser.add_argument("--intrinsic-hold-fraction", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=85)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def intrinsic_beta_schedule(initial, final, update, updates, hold_fraction):
    if not 0.0 <= hold_fraction <= 1.0:
        raise ValueError("hold_fraction must lie in [0,1]")
    if updates <= 1:
        return float(initial)
    progress = (update - 1) / (updates - 1)
    if progress <= hold_fraction or hold_fraction == 1.0:
        return float(initial)
    anneal = (progress - hold_fraction) / (1.0 - hold_fraction)
    cosine = 0.5 * (1.0 + math.cos(math.pi * anneal))
    return float(final + (initial - final) * cosine)


def _write_json(path: Path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_history_response(path: Path, fold_index: int, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    folds = payload.get("primary")
    if not isinstance(folds, list) or not 0 <= fold_index < len(folds):
        raise ValueError("response checkpoint lacks the requested primary fold")
    fold = folds[fold_index]
    model = ContinuousResponseEnsemble(
        input_dim=120, response_dim=12, action_dim=39,
        members=5, hidden_size=256).to(device).eval()
    model.load_state_dict(fold["models"]["H1"])
    model.requires_grad_(False)
    standardizer = fold["standardizer"]
    return {
        "model": model,
        "target_mean": standardizer["mean"].to(device),
        "target_scale": standardizer["scale"].to(device),
        "variance_scale": fold["variance_scale"].to(device),
        "held_collection_seed": int(fold["held_collection_seed"]),
    }


def load_option_q(path: Path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if (payload.get("target") != "return_30"
            or payload.get("representation") != "registered_v7_1"):
        raise ValueError("online option Q must use registered_v7_1@return_30")
    if len(payload.get("folds", [])) != 2:
        raise ValueError("option Q checkpoint must contain two cross-fitted folds")
    loaded = []
    for fold in payload["folds"]:
        model = MacroOptionDuelingQ(
            payload["context_dim"], payload["num_options"],
            payload["hidden_size"]).to(device).eval()
        model.load_state_dict(fold["model"])
        model.requires_grad_(False)
        loaded.append({
            "model": model,
            "mean": fold["context_mean"].to(device),
            "scale": fold["context_scale"].to(device),
            "collection_seed": int(fold["collection_seed"]),
        })
    return payload, loaded


def option_values(fold, raw_context, legal):
    standardized = (raw_context - fold["mean"]) / fold["scale"]
    return fold["model"].all_options(standardized, legal)


def main():
    args = parse_args()
    if args.rollout_steps % args.sequence_length != 0:
        raise ValueError("rollout-steps must be divisible by sequence-length")
    if args.samples_per_member <= 0 or args.samples_per_member % 2:
        raise ValueError("samples-per-member must be a positive even number")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    initial = torch.load(
        args.rhppo_checkpoint, map_location=device, weights_only=False)
    policy = RecurrentMultiDiscreteActorCritic(
        obs_dim=int(initial["obs_dim"]),
        hidden_size=int(initial["hidden_size"])).to(device)
    policy.load_state_dict(initial["model"])
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=args.learning_rate, eps=1.0e-5)
    response = load_history_response(
        args.response_checkpoint, args.response_fold, device)
    q_payload, q_folds = load_option_q(args.option_q_checkpoint, device)

    env = TorchWVRVecEnv(TorchWVRConfig(
        num_envs=args.num_envs, reward_mode="competitive",
        attack_quality_weight=0.0, observation_version="v4",
        use_dynamic_launch_envelope=True),
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
    response_history = VectorHistoryBuffer(
        args.num_envs, device=device, length=3)
    voi_rms = RunningRMS(device)
    history = []
    total_decisions = 0

    # Compile dynamics before timing without changing the environment state.
    with torch.inference_mode():
        warm = torch.tensor(
            [1, 2, 1, 0], device=device).expand(args.num_envs, -1)
        warm_all = fixed_mixture_actions(obs, env_index.remainder(3))
        warm_all[env_index, learner_side] = warm
        warm_controls, _ = lowlevel_controls(
            lowlevel, lowlevel_hidden, obs, warm_all,
            torch_speed_adapter=True)
        env.dynamics_runner(
            env.aircraft_state.reshape(-1, env.dynamics.state_size),
            warm_controls, substeps=1)
    torch.cuda.synchronize()
    started = time.perf_counter()

    initial_intrinsic_beta = args.intrinsic_beta
    for update in range(1, args.updates + 1):
        buffers = {key: [] for key in (
            "obs", "prev_action", "hidden_before", "reset", "action",
            "log_prob", "value_ext", "value_int", "next_value_ext",
            "next_value_int", "reward_ext", "reward_int", "done")}
        completed = wins = losses = timeouts = launches = hits = 0
        signed_values = []
        positive_values = []
        diversity_values = []
        fold_gap_values = []
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
                history_batch = response_history.append(
                    learner_obs, learner_action, reset=reset)
                current_response, _ = extract_response_state(learner_obs)
                candidates = response["model"].sample_candidates(
                    history_batch.context, history_batch.action_input,
                    samples_per_member=args.samples_per_member,
                    variance_scale=response["variance_scale"],
                    target_mean=response["target_mean"],
                    target_scale=response["target_scale"],
                    current_response=current_response)
                samples = candidates["continuous"].shape[1]
                raw_context = torch.cat((
                    result["belief"][:, None, :].expand(-1, samples, -1),
                    action_one_hot[:, None, :].expand(-1, samples, -1),
                    candidates["continuous"]), -1)
                flat_context = raw_context.reshape(-1, raw_context.shape[-1])
                legal = macro_option_legal_mask(learner_obs)
                flat_legal = legal[:, None, :].expand(
                    -1, samples, -1).reshape(-1, legal.shape[-1])
                q_a = option_values(q_folds[0], flat_context, flat_legal).reshape(
                    args.num_envs, samples, -1)
                q_b = option_values(q_folds[1], flat_context, flat_legal).reshape(
                    args.num_envs, samples, -1)
                voi = masked_soft_cross_fitted_voi(
                    q_a, q_b, candidates["weights"], legal,
                    temperature=args.soft_voi_temperature,
                    top_k=args.soft_voi_top_k)
                reward_int, current_rms = voi_rms.normalize(voi["positive_voi"])
                signed_values.append(voi["signed_voi"])
                positive_values.append(voi["positive_voi"])
                diversity_values.append(
                    (q_a.argmax(-1).amax(-1)
                     != q_a.argmax(-1).amin(-1)).float())
                fold_gap_values.append((q_a - q_b).abs().mean((1, 2)))

                executed = fixed_mixture_actions(obs, env_index.remainder(3))
                executed[env_index, learner_side] = learner_action
                controls, next_lowlevel_hidden = lowlevel_controls(
                    lowlevel, lowlevel_hidden, obs, executed,
                    torch_speed_adapter=True)
                next_obs, reward, done, info = env.step(
                    controls.reshape(args.num_envs, 2, 4),
                    executed[..., 3] == 1)
                next_learner_obs = next_obs[env_index, learner_side]
                next_previous_action = torch.where(
                    done[:, None], torch.zeros_like(action_one_hot),
                    action_one_hot)
                preview = policy.act(
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
                    ("reward_int", reward_int), ("done", done)):
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
        rollout.update(dual_channel_gae(
            rollout["reward_ext"], rollout["reward_int"],
            rollout["value_ext"], rollout["value_int"],
            rollout["next_value_ext"], rollout["next_value_int"],
            rollout["done"], args.gamma, args.gae_lambda))
        args.intrinsic_beta = intrinsic_beta_schedule(
            initial_intrinsic_beta, args.intrinsic_beta_final,
            update, args.updates, args.intrinsic_hold_fraction)
        update_started = time.perf_counter()
        metrics = recurrent_ppo_update(policy, optimizer, rollout, args)
        update_s = time.perf_counter() - update_started
        total_decisions += args.num_envs * args.rollout_steps
        signed = torch.cat(signed_values)
        positive = torch.cat(positive_values)
        diversity = torch.cat(diversity_values)
        fold_gap = torch.cat(fold_gap_values)
        entry = {
            "update": update,
            "environment_decisions": total_decisions,
            "completed_episodes": completed,
            "learner_wins": wins,
            "learner_losses": losses,
            "timeouts": timeouts,
            "learner_accepted_launches": launches,
            "learner_hits": hits,
            "mean_reward_ext": float(rollout["reward_ext"].mean()),
            "mean_reward_int": float(rollout["reward_int"].mean()),
            "intrinsic_beta": args.intrinsic_beta,
            "signed_voi_mean": float(signed.mean()),
            "signed_voi_negative_fraction": float((signed < 0).float().mean()),
            "positive_voi_mean": float(positive.mean()),
            "positive_voi_nonzero_fraction": float(
                (positive > 1.0e-6).float().mean()),
            "candidate_option_diversity_fraction": float(diversity.mean()),
            "q_fold_mean_absolute_gap": float(fold_gap.mean()),
            "voi_rms": float(current_rms),
            "rollout_s": rollout_s,
            "ppo_update_s": update_s,
            **metrics,
        }
        history.append(entry)
        checkpoint = {
            "algorithm": ALGORITHM,
            "method": METHOD,
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "obs_dim": env.observation_size,
            "hidden_size": policy.hidden_size,
            "action_dims": [3, 5, 3, 2],
            "seed": args.seed,
            "completed_updates": update,
            "training_environment_decisions": total_decisions,
            "environment_config": asdict(env.config),
            "history": history,
            "response_fold": args.response_fold,
            "response_held_collection_seed": response[
                "held_collection_seed"],
            "option_q_target": q_payload["target"],
            "option_q_train_seeds": q_payload["train_seeds"],
        }
        atomic_torch_save(checkpoint, args.output_dir / "latest.pt")
        _write_json(args.output_dir / "progress.json", entry)
        print(json.dumps(entry), flush=True)

    torch.cuda.synchronize()
    report = {
        "algorithm": ALGORITHM,
        "method": METHOD,
        "scientific_result": False,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "rollout_steps": args.rollout_steps,
        "updates": args.updates,
        "total_environment_decisions": total_decisions,
        "elapsed_s": time.perf_counter() - started,
        "history": history,
    }
    _write_json(args.output_dir / "train.json", report)


if __name__ == "__main__":
    main()
