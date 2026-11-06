"""PPO engineering trainer for the complete Torch 1v1 WVR environment."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch

from benchmarks.benchmark_torch6dof_ppo import lowlevel_controls
from benchmarks.evaluate_torch_wvr_fixed_opponents import (
    scripted_learner_actions,
)
from benchmarks.train_ppo_1v1_shoot_smoke import (
    MultiDiscreteActorCritic, Rollout, ppo_update,
)
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.utils.utils import get_root_dir
from envs.torch6dof import TorchWVRConfig, TorchWVRVecEnv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=96)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=85)
    parser.add_argument("--curriculum", choices=("formal", "close"),
                        default="formal")
    parser.add_argument("--training-opponents",
                        choices=("selfplay", "fixed-mix", "staged", "gated",
                                 "evasive"),
                        default="selfplay")
    parser.add_argument(
        "--reward-variant",
        choices=("auto", "competitive", "engagement", "attack-quality"),
        default="auto")
    parser.add_argument("--gate-min-episodes-per-kind", type=int, default=24)
    parser.add_argument("--gate-consecutive-updates", type=int, default=2)
    parser.add_argument("--gate-window-updates", type=int, default=4)
    parser.add_argument("--teacher-aux-coef", type=float, default=0.0)
    parser.add_argument("--teacher-aux-final-coef", type=float, default=0.0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def fixed_mixture_actions(obs: torch.Tensor,
                          opponent_kind: torch.Tensor) -> torch.Tensor:
    """Three deterministic curriculum opponents in one static GPU batch."""
    action = torch.zeros(*obs.shape[:2], 4, dtype=torch.long,
                         device=obs.device)
    action[..., 0] = 1
    action[..., 2] = 1
    side = obs[..., 14]
    hold = torch.full_like(side, 2, dtype=torch.long)
    pursuit = torch.where(side >= 0.0, 4, 0)
    evasive = torch.where(side >= 0.0, 0, 4)
    kind = opponent_kind[:, None]
    action[..., 1] = torch.where(
        kind == 0, hold, torch.where(kind == 1, pursuit, evasive))
    action[..., 2] = torch.where(
        kind == 0, torch.ones_like(action[..., 2]),
        torch.zeros_like(action[..., 2]))
    action[..., 3] = (obs[..., 23] >= 0.5).long()
    return action


def staged_opponent_kind(env_index: torch.Tensor, update: int,
                         total_updates: int):
    """Deterministic easy-to-hard opponent schedule.

    Stage 1 contains pursuit, stage 2 adds straight/hold, and stage 3 adds
    evasive.  The order follows paired V3/V4 empirical difficulty.
    """
    kind_order = torch.tensor([1, 0, 2], device=env_index.device)
    progress = (update - 1) / max(total_updates, 1)
    if progress < 1.0 / 3.0:
        return torch.ones_like(env_index), "pursuit"
    if progress < 2.0 / 3.0:
        return kind_order[env_index.remainder(2)], "pursuit+hold"
    return (kind_order[env_index.remainder(3)],
            "pursuit+hold+evasive")


def gated_opponent_kind(env_index: torch.Tensor, stage: int):
    """Return the empirically ordered mixture for a metric-gated stage.

    Paired V3/V4 evaluations consistently show pursuit to be easier than the
    armed straight/hold opponent, so curriculum order follows measured
    competence rather than the scripted controller's name.
    """
    labels = ("pursuit", "pursuit+hold", "pursuit+hold+evasive")
    kind_order = torch.tensor([1, 0, 2], device=env_index.device)
    active_kinds = min(max(int(stage), 0) + 1, 3)
    return kind_order[env_index.remainder(active_kinds)], labels[
        active_kinds - 1]


def curriculum_gate_metrics(stage: int, completed, wins, timeouts, *,
                            accepted_launches: int, fire_requests: int,
                            rejected_requests: int,
                            min_episodes_per_kind: int):
    """Evaluate whether the current curriculum competence gate is met.

    Stage 0 requires 50% wins and at most 45% timeouts against pursuit. Stage 1
    requires every active opponent to reach 45% wins and at most 55% timeouts.
    Both stages also require a usable fire-control interface: at least one
    request, >=70% accepted requests, and <=10% rejected requests.
    """
    active = min(stage + 1, 3)
    kind_order = (1, 0, 2)[:active]
    completed = [int(completed[index]) for index in kind_order]
    wins = [int(wins[index]) for index in kind_order]
    timeouts = [int(timeouts[index]) for index in kind_order]
    win_rates = [win / max(count, 1)
                 for win, count in zip(wins, completed)]
    timeout_rates = [timeout / max(count, 1)
                     for timeout, count in zip(timeouts, completed)]
    win_threshold = 0.50 if stage == 0 else 0.45
    timeout_threshold = 0.45 if stage == 0 else 0.55
    accepted_per_request = accepted_launches / max(fire_requests, 1)
    rejected_per_request = rejected_requests / max(fire_requests, 1)
    enough_episodes = all(
        count >= min_episodes_per_kind for count in completed)
    passed = (
        stage < 2
        and enough_episodes
        and all(rate >= win_threshold for rate in win_rates)
        and all(rate <= timeout_threshold for rate in timeout_rates)
        and fire_requests > 0
        and accepted_per_request >= 0.70
        and rejected_per_request <= 0.10
    )
    return {
        "passed": passed,
        "active_kinds": active,
        "opponent_kind_order": list(kind_order),
        "completed_by_kind": completed,
        "win_rates_by_kind": win_rates,
        "timeout_rates_by_kind": timeout_rates,
        "accepted_per_request": accepted_per_request,
        "rejected_per_request": rejected_per_request,
        "enough_episodes": enough_episodes,
        "win_threshold": win_threshold,
        "timeout_threshold": timeout_threshold,
    }


def reward_variant_config(training_opponents: str, reward_variant: str):
    """Resolve one explicit reward contract without touching environment physics."""
    variant = reward_variant
    if variant == "auto":
        # The paired three-seed V4 ablation rejected the stronger engagement
        # penalties and the 0.5 attack-quality potential.  Keep auto on the
        # least-regret competitive contract for every opponent curriculum.
        variant = "competitive"
    if variant == "competitive":
        return variant, {
            "reward_mode": "competitive",
            "attack_quality_weight": 0.0,
        }
    config = {
        "reward_mode": "engagement",
        "timeout_penalty": 1.0,
        "exhausted_penalty": 0.5,
        "attack_quality_weight": 0.0,
    }
    if variant == "attack-quality":
        config["attack_quality_weight"] = 0.5
    return variant, config


def los_teacher_auxiliary_loss(model, observations: torch.Tensor):
    """Cross-entropy to the non-privileged LOS-rate intercept teacher."""
    teacher = scripted_learner_actions(
        observations[:, None, :], "los-intercept")[:, 0]
    features = model._features(observations)
    distributions = model._distributions(observations, features)
    return torch.stack([
        -distribution.log_prob(teacher[:, index])
        for index, distribution in enumerate(distributions)
    ], dim=-1).sum(-1).mean()


def main():
    args = parse_args()
    if args.teacher_aux_coef < 0.0 or args.teacher_aux_final_coef < 0.0:
        raise ValueError("teacher auxiliary coefficients must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    resolved_reward_variant, reward_kwargs = reward_variant_config(
        args.training_opponents, args.reward_variant)
    config_kwargs = {"num_envs": args.num_envs, **reward_kwargs}
    if args.curriculum == "close":
        config_kwargs.update({
            "initial_min_range_m": 4000.0,
            "initial_max_range_m": 9000.0,
            "initial_aspect_probabilities": (0.10, 0.20, 0.70),
            "max_episode_steps": 600,
        })
    env = TorchWVRVecEnv(
        TorchWVRConfig(**config_kwargs), device=device,
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
    if args.resume is not None:
        resume_payload = torch.load(
            args.resume, map_location=device, weights_only=False)
        if int(resume_payload["obs_dim"]) != env.observation_size:
            raise ValueError(
                "resume checkpoint observation dimension does not match "
                f"the current environment: {resume_payload['obs_dim']} != "
                f"{env.observation_size}")
        model.load_state_dict(resume_payload["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=3.0e-4, eps=1.0e-5)
    update_args = SimpleNamespace(
        epochs=args.epochs, minibatch_size=args.minibatch_size,
        clip_ratio=0.2, value_coef=0.5, entropy_coef=1.0e-3,
        max_grad_norm=2.0)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    obs = env.observations()
    rnn_states = torch.zeros(args.num_envs * 2, 1, 128, device=device)
    env_index = torch.arange(args.num_envs, device=device)
    learner_side = (env_index % 2).long()
    learner_flat_index = 2 * env_index + learner_side
    episode_returns = torch.zeros(args.num_envs, 2, device=device)
    # Compile both physics graphs outside the measured region.
    with torch.inference_mode():
        warm_action, _, _ = model.act(obs.reshape(-1, env.observation_size))
        warm_control, _ = lowlevel_controls(
            lowlevel, rnn_states, obs,
            warm_action.reshape(args.num_envs, 2, 4),
            torch_speed_adapter=True)
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
    total_decisions = 0
    gate_stage = 0
    gate_consecutive_passes = 0
    gate_window = []
    start_all = time.perf_counter()
    for update in range(1, args.updates + 1):
        if args.training_opponents == "staged":
            opponent_kind, opponent_stage = staged_opponent_kind(
                env_index, update, args.updates)
        elif args.training_opponents == "gated":
            opponent_kind, opponent_stage = gated_opponent_kind(
                env_index, gate_stage)
        elif args.training_opponents == "evasive":
            opponent_kind = torch.full_like(env_index, 2)
            opponent_stage = "evasive"
        else:
            opponent_kind = env_index.remainder(3)
            opponent_stage = args.training_opponents
        obs_buffer, action_buffer, logprob_buffer = [], [], []
        value_buffer, reward_buffer, done_buffer = [], [], []
        completed = torch.zeros((), dtype=torch.long, device=device)
        red_wins = torch.zeros_like(completed)
        blue_wins = torch.zeros_like(completed)
        draws = torch.zeros_like(completed)
        timeouts = torch.zeros_like(completed)
        exhausted_draws = torch.zeros_like(completed)
        launches = torch.zeros_like(completed)
        hits = torch.zeros_like(completed)
        crashes = torch.zeros_like(completed)
        learner_wins = torch.zeros_like(completed)
        learner_losses = torch.zeros_like(completed)
        learner_launches = torch.zeros_like(completed)
        learner_hits = torch.zeros_like(completed)
        legal_opportunities = torch.zeros_like(completed)
        fire_requests = torch.zeros_like(completed)
        rejected_requests = torch.zeros_like(completed)
        learner_legal_opportunities = torch.zeros_like(completed)
        learner_fire_requests = torch.zeros_like(completed)
        learner_rejected_requests = torch.zeros_like(completed)
        learner_attack_quality_sum = torch.zeros((), device=device)
        learner_launch_quality_sum = torch.zeros((), device=device)
        learner_launch_quality_count = torch.zeros_like(completed)
        completed_by_kind = torch.zeros(3, dtype=torch.long, device=device)
        wins_by_kind = torch.zeros_like(completed_by_kind)
        timeouts_by_kind = torch.zeros_like(completed_by_kind)
        completed_abs_return = torch.zeros((), device=device)
        completed_length = torch.zeros((), device=device)
        rollout_start = time.perf_counter()
        for _ in range(args.rollout_steps):
            flat_obs = obs.reshape(-1, env.observation_size)
            with torch.inference_mode():
                action, log_prob, value = model.act(flat_obs)
                shaped_action = action.reshape(args.num_envs, 2, 4)
                if args.training_opponents != "selfplay":
                    opponent_action = fixed_mixture_actions(
                        obs, opponent_kind)
                    selector = torch.zeros(
                        args.num_envs, 2, 1, dtype=torch.bool,
                        device=device)
                    selector[env_index, learner_side, 0] = True
                    executed_action = torch.where(
                        selector, shaped_action, opponent_action)
                else:
                    executed_action = shaped_action
                controls, next_rnn = lowlevel_controls(
                    lowlevel, rnn_states, obs, executed_action,
                    torch_speed_adapter=True)
                next_obs, reward, done, info = env.step(
                    controls.reshape(args.num_envs, 2, 4),
                    executed_action[..., 3] == 1)
            if args.training_opponents != "selfplay":
                obs_buffer.append(flat_obs[learner_flat_index])
                action_buffer.append(action[learner_flat_index])
                logprob_buffer.append(log_prob[learner_flat_index])
                value_buffer.append(value[learner_flat_index])
                reward_buffer.append(reward[env_index, learner_side])
            else:
                obs_buffer.append(flat_obs)
                action_buffer.append(action)
                logprob_buffer.append(log_prob)
                value_buffer.append(value)
                reward_buffer.append(reward.reshape(-1))
            done_buffer.append(done)
            episode_returns += reward
            completed += done.sum()
            red_wins += ((info["outcome_red"] > 0) & done).sum()
            blue_wins += ((info["outcome_red"] < 0) & done).sum()
            draws += ((info["outcome_red"] == 0) & done).sum()
            timeouts += info["timeout"].sum()
            exhausted_draws += info["weapons_exhausted"].sum()
            launches += info["accepted_launches"].sum()
            hits += info["hits"].sum()
            crashes += info["crashes"].sum()
            legal_opportunities += info["legal_fire_opportunities"].sum()
            fire_requests += info["fire_requests"].sum()
            rejected_requests += info["rejected_fire_requests"].sum()
            learner_outcome = torch.where(
                learner_side == 0, info["outcome_red"],
                -info["outcome_red"])
            learner_wins += ((learner_outcome > 0) & done).sum()
            learner_losses += ((learner_outcome < 0) & done).sum()
            if args.training_opponents in (
                    "fixed-mix", "staged", "gated", "evasive"):
                for kind_index in range(3):
                    kind_done = done & (opponent_kind == kind_index)
                    completed_by_kind[kind_index] += kind_done.sum()
                    wins_by_kind[kind_index] += (
                        (learner_outcome > 0) & kind_done).sum()
                    timeouts_by_kind[kind_index] += (
                        info["timeout"] & kind_done).sum()
            learner_launches += info["accepted_launches"][
                env_index, learner_side].sum()
            learner_hits += info["hits"][env_index, learner_side].sum()
            learner_legal_opportunities += info[
                "legal_fire_opportunities"][env_index, learner_side].sum()
            learner_fire_requests += info["fire_requests"][
                env_index, learner_side].sum()
            learner_rejected_requests += info[
                "rejected_fire_requests"][env_index, learner_side].sum()
            learner_quality = info["prelaunch_attack_quality"][
                env_index, learner_side]
            learner_accepted = info["accepted_launches"][
                env_index, learner_side]
            learner_attack_quality_sum += learner_quality.sum()
            learner_launch_quality_sum += (
                learner_quality * learner_accepted.float()).sum()
            learner_launch_quality_count += learner_accepted.sum()
            completed_abs_return += (
                episode_returns.abs().mean(-1) * done.float()).sum()
            completed_length += (
                info["episode_length"].float() * done.float()).sum()
            episode_returns = torch.where(
                done[:, None], torch.zeros_like(episode_returns),
                episode_returns)
            rnn_states = torch.where(
                done.repeat_interleave(2)[:, None, None],
                torch.zeros_like(next_rnn), next_rnn)
            obs = next_obs
        torch.cuda.synchronize()
        rollout_s = time.perf_counter() - rollout_start

        with torch.inference_mode():
            next_value = model.value(obs.reshape(-1, env.observation_size))
            if args.training_opponents != "selfplay":
                next_value = next_value[learner_flat_index]
        rewards = torch.stack(reward_buffer)
        values = torch.stack(value_buffer)
        dones = torch.stack(done_buffer)
        if args.training_opponents == "selfplay":
            dones = dones.repeat_interleave(2, dim=1)
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros_like(next_value)
        for step in reversed(range(args.rollout_steps)):
            following = (next_value if step == args.rollout_steps - 1
                         else values[step + 1])
            nonterminal = (~dones[step]).float()
            delta = rewards[step] + 0.99 * following * nonterminal - values[step]
            gae = delta + 0.99 * 0.95 * nonterminal * gae
            advantages[step] = gae
        returns = advantages + values
        flat_advantage = advantages.reshape(-1)
        flat_advantage = ((flat_advantage - flat_advantage.mean()) /
                          (flat_advantage.std() + 1.0e-8))
        if args.training_opponents == "selfplay":
            paired_rewards = rewards.reshape(
                args.rollout_steps, args.num_envs, 2)
            zero_sum_error = float(
                torch.abs(paired_rewards.sum(-1)).max())
            mean_red_reward = float(paired_rewards[..., 0].mean())
        else:
            # Only the learner-side rewards enter PPO in fixed-opponent mode;
            # a paired zero-sum diagnostic is therefore not applicable.
            zero_sum_error = 0.0
            mean_red_reward = float(rewards.mean())
        rollout = Rollout(
            observations=torch.stack(obs_buffer).reshape(
                -1, env.observation_size),
            actions=torch.stack(action_buffer).reshape(-1, 4),
            old_log_probs=torch.stack(logprob_buffer).reshape(-1),
            old_values=values.reshape(-1), returns=returns.reshape(-1),
            advantages=flat_advantage,
            mean_reward=float(rewards.mean()),
            mean_abs_reward=float(rewards.abs().mean()),
            zero_sum_error=zero_sum_error,
            completed_episodes=int(completed.cpu()),
            mean_abs_completed_return=(
                float((completed_abs_return / completed.clamp_min(1)).cpu())),
            accepted_launches=int(launches.cpu()))
        update_start = time.perf_counter()
        schedule_fraction = ((update - 1) / max(args.updates - 1, 1))
        teacher_aux_coef = (
            args.teacher_aux_coef
            + schedule_fraction * (
                args.teacher_aux_final_coef - args.teacher_aux_coef))
        metrics = ppo_update(
            model, optimizer, rollout, update_args, scaler, True,
            auxiliary_loss_fn=(
                los_teacher_auxiliary_loss
                if teacher_aux_coef > 0.0 else None),
            auxiliary_coef=teacher_aux_coef)
        torch.cuda.synchronize()
        update_s = time.perf_counter() - update_start
        decisions = args.num_envs * args.rollout_steps
        total_decisions += decisions
        count = completed.clamp_min(1)
        gate_metrics = None
        gate_transition = False
        if args.training_opponents == "gated":
            gate_window.append({
                "completed": completed_by_kind.cpu(),
                "wins": wins_by_kind.cpu(),
                "timeouts": timeouts_by_kind.cpu(),
                "accepted": int(learner_launches.cpu()),
                "requests": int(learner_fire_requests.cpu()),
                "rejected": int(learner_rejected_requests.cpu()),
            })
            gate_window = gate_window[-args.gate_window_updates:]
            window_completed = sum(
                (item["completed"] for item in gate_window),
                torch.zeros(3, dtype=torch.long))
            window_wins = sum(
                (item["wins"] for item in gate_window),
                torch.zeros(3, dtype=torch.long))
            window_timeouts = sum(
                (item["timeouts"] for item in gate_window),
                torch.zeros(3, dtype=torch.long))
            window_accepted = sum(item["accepted"] for item in gate_window)
            window_requests = sum(item["requests"] for item in gate_window)
            window_rejected = sum(item["rejected"] for item in gate_window)
            gate_metrics = curriculum_gate_metrics(
                gate_stage, window_completed.tolist(),
                window_wins.tolist(), window_timeouts.tolist(),
                accepted_launches=window_accepted,
                fire_requests=window_requests,
                rejected_requests=window_rejected,
                min_episodes_per_kind=args.gate_min_episodes_per_kind)
            gate_metrics["window_updates"] = len(gate_window)
            gate_consecutive_passes = (
                gate_consecutive_passes + 1
                if gate_metrics["passed"] else 0)
            if (gate_stage < 2 and gate_consecutive_passes
                    >= args.gate_consecutive_updates):
                gate_stage += 1
                gate_transition = True
                gate_consecutive_passes = 0
                gate_window.clear()
        history.append({
            "update": update,
            "opponent_stage": opponent_stage,
            "environment_decisions": total_decisions,
            "mean_abs_reward": rollout.mean_abs_reward,
            "mean_red_reward": mean_red_reward,
            "mean_learning_reward": float(rewards.mean()),
            "zero_sum_error": rollout.zero_sum_error,
            "completed_episodes": int(completed.cpu()),
            "red_wins": int(red_wins.cpu()),
            "blue_wins": int(blue_wins.cpu()),
            "draws": int(draws.cpu()),
            "timeouts": int(timeouts.cpu()),
            "weapons_exhausted_draws": int(exhausted_draws.cpu()),
            "accepted_launches": int(launches.cpu()),
            "hits": int(hits.cpu()),
            "legal_fire_opportunities": int(legal_opportunities.cpu()),
            "fire_requests": int(fire_requests.cpu()),
            "rejected_fire_requests": int(rejected_requests.cpu()),
            "crashes": int(crashes.cpu()),
            "learner_wins": int(learner_wins.cpu()),
            "learner_losses": int(learner_losses.cpu()),
            "learner_accepted_launches": int(learner_launches.cpu()),
            "learner_hits": int(learner_hits.cpu()),
            "learner_legal_fire_opportunities": int(
                learner_legal_opportunities.cpu()),
            "learner_fire_requests": int(learner_fire_requests.cpu()),
            "learner_rejected_fire_requests": int(
                learner_rejected_requests.cpu()),
            "learner_mean_attack_quality": float(
                (learner_attack_quality_sum
                 / (args.num_envs * args.rollout_steps)).cpu()),
            "learner_mean_launch_attack_quality": float(
                (learner_launch_quality_sum
                 / learner_launch_quality_count.clamp_min(1)).cpu()),
            "completed_by_opponent_kind": completed_by_kind.cpu().tolist(),
            "wins_by_opponent_kind": wins_by_kind.cpu().tolist(),
            "timeouts_by_opponent_kind": timeouts_by_kind.cpu().tolist(),
            "curriculum_gate": gate_metrics,
            "curriculum_gate_consecutive_passes": gate_consecutive_passes,
            "curriculum_gate_transition": gate_transition,
            "curriculum_stage_after_update": gate_stage,
            "teacher_aux_coef": teacher_aux_coef,
            "mean_abs_completed_return": rollout.mean_abs_completed_return,
            "mean_episode_length": float((completed_length / count).cpu()),
            "rollout_s": rollout_s,
            "ppo_update_s": update_s,
            **metrics,
        })
        print(json.dumps(history[-1]), flush=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_all
    report = {
        "backend": "torch_wvr_vec_env_v4",
        "scientific_result": False,
        "curriculum": args.curriculum,
        "training_opponents": args.training_opponents,
        "reward_variant": resolved_reward_variant,
        "curriculum_gate_config": {
            "min_episodes_per_kind": args.gate_min_episodes_per_kind,
            "consecutive_updates": args.gate_consecutive_updates,
            "window_updates": args.gate_window_updates,
        },
        "teacher_aux_schedule": {
            "initial_coef": args.teacher_aux_coef,
            "final_coef": args.teacher_aux_final_coef,
        },
        "reward_zero_sum_diagnostic_applicable": (
            args.training_opponents == "selfplay"),
        "resumed_from": str(args.resume) if args.resume else None,
        "seed": args.seed, "num_envs": args.num_envs,
        "sim_freq_hz": 60, "physics_substeps_per_decision": 12,
        "updates": args.updates, "rollout_steps": args.rollout_steps,
        "total_environment_decisions": total_decisions,
        "elapsed_s": elapsed,
        "environment_decisions_per_s": total_decisions / elapsed,
        "lowlevel_frozen": not any(
            parameter.requires_grad for parameter in lowlevel.parameters()),
        "torch_speed_adapter": True,
        "environment_config": asdict(env.config),
        "history": history,
    }
    rendered = json.dumps(report, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": model.state_dict(), "obs_dim": env.observation_size,
            "action_dims": [3, 5, 3, 2], "hidden_size": args.hidden_size,
            "seed": args.seed,
            "training_environment_decisions": total_decisions,
            "backend": "torch_wvr_vec_env_v4",
            "torch_speed_adapter": True,
            "curriculum": args.curriculum,
            "training_opponents": args.training_opponents,
            "reward_variant": resolved_reward_variant,
            "teacher_aux_coef": args.teacher_aux_coef,
            "teacher_aux_final_coef": args.teacher_aux_final_coef,
            "curriculum_stage": gate_stage,
            "resumed_from": str(args.resume) if args.resume else None,
            "environment_config": asdict(env.config),
        }, args.checkpoint)
    print(rendered)


if __name__ == "__main__":
    main()
