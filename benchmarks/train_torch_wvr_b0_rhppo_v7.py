#!/usr/bin/env python3
"""Train the frozen extrinsic-only recurrent B0 policy for DRVOI V7."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from benchmarks.benchmark_torch6dof_ppo import lowlevel_controls
from benchmarks.drvoi import multi_discrete_to_one_hot
from benchmarks.drvoi.checkpointing import (
    atomic_torch_save,
    capture_rng_state,
    restore_rng_state,
)
from benchmarks.drvoi.rhppo import (
    RecurrentMultiDiscreteActorCritic,
    dual_channel_gae,
)
from benchmarks.train_torch_wvr_ppo import fixed_mixture_actions
from benchmarks.train_torch_wvr_rhppo import recurrent_ppo_update
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.utils.utils import get_root_dir
from envs.torch6dof import TorchWVRConfig, TorchWVRVecEnv


ALGORITHM = "v7_b0_extrinsic_rhppo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=96)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--updates", type=int, default=41)
    parser.add_argument("--epochs", type=int, default=2)
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _checkpoint_payload(model, optimizer, env, args, update, decisions, history):
    return {
        "algorithm": ALGORITHM,
        "protocol_version": "DRVOI-RHPPO-V7-WVR-1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "obs_dim": env.observation_size,
        "hidden_size": args.hidden_size,
        "action_dims": [3, 5, 3, 2],
        "seed": args.seed,
        "completed_updates": update,
        "training_environment_decisions": decisions,
        "environment_config": asdict(env.config),
        "history": history,
        "rng_state": capture_rng_state(),
        "torch_speed_adapter": True,
        "extrinsic_only": True,
        "frozen_after_training": True,
    }


def main() -> None:
    args = parse_args()
    if args.rollout_steps % args.sequence_length != 0:
        raise ValueError("rollout-steps must be divisible by sequence-length")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "latest.pt"
    progress_path = args.output_dir / "progress.json"
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    env = TorchWVRVecEnv(
        TorchWVRConfig(
            num_envs=args.num_envs,
            reward_mode="competitive",
            attack_quality_weight=0.0,
            observation_version="v4",
            use_dynamic_launch_envelope=True,
        ),
        device=device,
        seed=args.seed,
        compile_dynamics=True,
    )
    model = RecurrentMultiDiscreteActorCritic(
        obs_dim=env.observation_size, hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, eps=1.0e-5)
    start_update = 0
    total_decisions = 0
    history: list[dict[str, object]] = []
    if args.resume and checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if payload.get("algorithm") != ALGORITHM or payload.get("seed") != args.seed:
            raise ValueError("resume checkpoint does not match V7 B0 seed")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_update = int(payload["completed_updates"])
        total_decisions = int(payload["training_environment_decisions"])
        history = list(payload.get("history", []))
        restore_rng_state(payload["rng_state"])

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

    # Compile before timing without advancing the actual environment.
    with torch.inference_mode():
        warm = torch.zeros(args.num_envs, 4, dtype=torch.long, device=device)
        warm[:, 0] = 1
        warm[:, 1] = 2
        warm[:, 2] = 1
        warm_all = fixed_mixture_actions(obs, env_index.remainder(3))
        warm_all[env_index, learner_side] = warm
        warm_controls, _ = lowlevel_controls(
            lowlevel, lowlevel_hidden, obs, warm_all, torch_speed_adapter=True)
        env.dynamics_runner(
            env.aircraft_state.reshape(-1, env.dynamics.state_size),
            warm_controls,
            substeps=1,
        )
    torch.cuda.synchronize()

    started = time.perf_counter()
    for update in range(start_update + 1, args.updates + 1):
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
                encoded_action = multi_discrete_to_one_hot(learner_action)
                next_previous_action = torch.where(
                    done[:, None], torch.zeros_like(encoded_action), encoded_action)
                preview = model.act(
                    next_learner_obs,
                    next_previous_action,
                    result["hidden"],
                    reset=done,
                    deterministic=True,
                )
            learner_reward = reward[env_index, learner_side]
            for key, value in (
                ("action", learner_action), ("log_prob", result["log_prob"]),
                ("value_ext", result["value_ext"]),
                ("value_int", result["value_int"]),
                ("next_value_ext", preview["value_ext"]),
                ("next_value_int", preview["value_int"]),
                ("reward_ext", learner_reward),
                ("reward_int", torch.zeros_like(learner_reward)),
                ("done", done),
            ):
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
                torch.zeros_like(next_lowlevel_hidden),
                next_lowlevel_hidden,
            )

        rollout_s = time.perf_counter() - rollout_started
        rollout = {key: torch.stack(value) for key, value in buffers.items()}
        rollout.update(dual_channel_gae(
            rollout["reward_ext"], rollout["reward_int"],
            rollout["value_ext"], rollout["value_int"],
            rollout["next_value_ext"], rollout["next_value_int"],
            rollout["done"], args.gamma, args.gae_lambda))
        update_started = time.perf_counter()
        metrics = recurrent_ppo_update(model, optimizer, rollout, args)
        update_s = time.perf_counter() - update_started
        total_decisions += args.num_envs * args.rollout_steps
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
            "rollout_s": rollout_s,
            "ppo_update_s": update_s,
            **metrics,
        }
        history.append(entry)
        checkpoint = _checkpoint_payload(
            model, optimizer, env, args, update, total_decisions, history)
        atomic_torch_save(checkpoint, checkpoint_path)
        _write_json(progress_path, entry)
        print(json.dumps(entry), flush=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    report = {
        "algorithm": ALGORITHM,
        "protocol_version": "DRVOI-RHPPO-V7-WVR-1",
        "scientific_result": False,
        "environment_frozen": True,
        "extrinsic_only": True,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "rollout_steps": args.rollout_steps,
        "updates": args.updates,
        "total_environment_decisions": total_decisions,
        "elapsed_s_this_invocation": elapsed,
        "history": history,
    }
    _write_json(args.output_dir / "train.json", report)
    (args.output_dir / "FROZEN").write_text(
        "V7 B0 checkpoint; do not update during response validation.\n",
        encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
