"""Behavior-clone the LOS-rate intercept teacher in the Torch WVR environment."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmarks.benchmark_torch6dof_ppo import lowlevel_controls
from benchmarks.evaluate_torch_wvr_fixed_opponents import (
    scripted_actions,
    scripted_learner_actions,
)
from benchmarks.train_ppo_1v1_shoot_smoke import MultiDiscreteActorCritic
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.utils.utils import get_root_dir
from envs.torch6dof import TorchWVRConfig, TorchWVRVecEnv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=96)
    parser.add_argument("--collection-steps", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument(
        "--teacher-opponents", choices=("evasive", "mixed"),
        default="evasive")
    parser.add_argument("--seed", type=int, default=85)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    return parser.parse_args()


def behavior_cloning_loss(model, obs: torch.Tensor, actions: torch.Tensor):
    features = model._features(obs)
    logits = [head(features) for head in model.action_heads]
    losses = [F.cross_entropy(logits[index], actions[:, index])
              for index in range(3)]
    legal = obs[:, 23] >= 0.5
    if bool(legal.any()):
        fire_loss = F.cross_entropy(logits[3][legal], actions[legal, 3])
    else:
        fire_loss = logits[3].sum() * 0.0
    loss = sum(losses) + fire_loss
    with torch.no_grad():
        predicted, _, _ = model.act(obs, deterministic=True)
        head_accuracy = (predicted == actions).float().mean(0)
        exact_accuracy = (predicted == actions).all(-1).float().mean()
    return loss, head_accuracy, exact_accuracy, int(legal.sum())


def teacher_opponent_actions(obs: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "evasive":
        return scripted_actions(obs, "evasive")
    env_index = torch.arange(obs.shape[0], device=obs.device)
    kind = env_index.remainder(3)[:, None, None]
    hold = scripted_actions(obs, "hold")
    pursuit = scripted_actions(obs, "pursuit")
    evasive = scripted_actions(obs, "evasive")
    return torch.where(kind == 0, hold, torch.where(kind == 1, pursuit, evasive))


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if min(args.num_envs, args.collection_steps, args.iterations,
           args.epochs, args.minibatch_size) <= 0:
        raise ValueError("all count arguments must be positive")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    env = TorchWVRVecEnv(
        TorchWVRConfig(num_envs=args.num_envs), device=device,
        seed=args.seed, compile_dynamics=True)
    lowlevel = BaselineActor().to(device).eval()
    lowlevel.load_state_dict(torch.load(
        get_root_dir() + "/model/baseline_model.pt",
        map_location=device, weights_only=True))
    lowlevel.requires_grad_(False)
    lowlevel.tpdv = {"dtype": torch.float32, "device": device}
    model = MultiDiscreteActorCritic(
        obs_dim=env.observation_size, action_dims=[3, 5, 3, 2],
        hidden_size=args.hidden_size, fire_mask_obs_index=23,
        altitude_safety_obs_index=0,
        altitude_descend_mask_threshold=0.8,
        altitude_force_climb_threshold=0.6).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, eps=1.0e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    obs = env.observations()
    rnn = torch.zeros(args.num_envs * 2, 1, 128, device=device)
    env_index = torch.arange(args.num_envs, device=device)
    learner_side = (env_index % 2).long()
    learner_flat = 2 * env_index + learner_side

    # Compile both environment graphs outside the measured collection loop.
    with torch.inference_mode():
        warm_action = scripted_learner_actions(obs, "los-intercept")
        warm_control, _ = lowlevel_controls(
            lowlevel, rnn, obs, warm_action, torch_speed_adapter=True)
        env.dynamics_runner(
            env.aircraft_state.reshape(-1, env.dynamics.state_size),
            warm_control, substeps=1)
        position, velocity = env._aircraft_neu()
        warm_missile = env.missile.launch(position, velocity)
        env.missile_runner(
            warm_missile.reshape(-1, env.missile.state_size),
            position.flip(1).reshape(-1, 3),
            velocity.flip(1).reshape(-1, 3))
    torch.cuda.synchronize()

    history = []
    total_samples = 0
    start = time.perf_counter()
    for iteration in range(1, args.iterations + 1):
        obs_batch = []
        action_batch = []
        completed = wins = legal_opportunities = launches = 0
        for _ in range(args.collection_steps):
            teacher = scripted_learner_actions(obs, "los-intercept")
            opponent = teacher_opponent_actions(obs, args.teacher_opponents)
            selector = torch.zeros(
                args.num_envs, 2, 1, dtype=torch.bool, device=device)
            selector[env_index, learner_side, 0] = True
            executed = torch.where(selector, teacher, opponent)
            obs_batch.append(obs.reshape(-1, env.observation_size)[learner_flat])
            action_batch.append(teacher[env_index, learner_side])
            with torch.inference_mode():
                controls, next_rnn = lowlevel_controls(
                    lowlevel, rnn, obs, executed, torch_speed_adapter=True)
                obs, _, done, info = env.step(
                    controls.reshape(args.num_envs, 2, 4),
                    executed[..., 3] == 1)
            learner_outcome = torch.where(
                learner_side == 0, info["outcome_red"], -info["outcome_red"])
            completed += int(done.sum())
            wins += int(((learner_outcome > 0) & done).sum())
            legal_opportunities += int(info["legal_fire_opportunities"][
                env_index, learner_side].sum())
            launches += int(info["accepted_launches"][
                env_index, learner_side].sum())
            rnn = torch.where(
                done.repeat_interleave(2)[:, None, None],
                torch.zeros_like(next_rnn), next_rnn)
        observations = torch.stack(obs_batch).reshape(-1, env.observation_size)
        actions = torch.stack(action_batch).reshape(-1, 4)
        total_samples += observations.shape[0]
        loss_values = []
        for _ in range(args.epochs):
            permutation = torch.randperm(observations.shape[0], device=device)
            for offset in range(0, observations.shape[0], args.minibatch_size):
                indices = permutation[offset:offset + args.minibatch_size]
                with torch.autocast("cuda", dtype=torch.float16):
                    loss, _, _, _ = behavior_cloning_loss(
                        model, observations[indices], actions[indices])
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(optimizer)
                scaler.update()
                loss_values.append(float(loss.detach()))
        with torch.inference_mode():
            final_loss, head_accuracy, exact_accuracy, legal_count = (
                behavior_cloning_loss(model, observations, actions))
        record = {
            "iteration": iteration,
            "teacher_samples": total_samples,
            "chunk_samples": observations.shape[0],
            "loss": sum(loss_values) / len(loss_values),
            "chunk_loss": float(final_loss),
            "head_accuracy": head_accuracy.cpu().tolist(),
            "exact_action_accuracy": float(exact_accuracy),
            "legal_fire_samples": legal_count,
            "teacher_completed_episodes": completed,
            "teacher_wins": wins,
            "teacher_legal_fire_opportunities": legal_opportunities,
            "teacher_launches": launches,
        }
        history.append(record)
        print(json.dumps(record), flush=True)

    elapsed = time.perf_counter() - start
    report = {
        "backend": "torch_wvr_vec_env_v4",
        "method": "behavior_cloning",
        "teacher": "los-intercept",
        "opponent": args.teacher_opponents,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "teacher_samples": total_samples,
        "elapsed_s": elapsed,
        "samples_per_s": total_samples / elapsed,
        "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "obs_dim": env.observation_size,
        "action_dims": [3, 5, 3, 2],
        "hidden_size": args.hidden_size,
        "seed": args.seed,
        "training_environment_decisions": 0,
        "backend": "torch_wvr_vec_env_v4",
        "torch_speed_adapter": True,
        "curriculum": "formal",
        "training_opponents": f"{args.teacher_opponents}-teacher",
        "reward_variant": "competitive",
        "pretraining": "behavior-cloning-los-intercept",
        "teacher_samples": total_samples,
        "teacher_opponents": args.teacher_opponents,
    }, args.checkpoint)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
