"""Train an option-gated hierarchical PPO baseline on Torch WVR.

The environment and both low-level skills are frozen.  PPO trains only the
high-level gate that selects either a standard combat policy or an LOS-rate
intercept policy for a fixed option horizon.  This keeps the hierarchical
baseline separate from the standard PPO implementation and prevents low-level
catastrophic forgetting from being hidden as an environment change.
"""

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
from benchmarks.train_ppo_1v1_shoot_smoke import (
    MultiDiscreteActorCritic,
    Rollout,
    ppo_update,
)
from benchmarks.train_torch_wvr_ppo import (
    fixed_mixture_actions,
    staged_opponent_kind,
)
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.utils.utils import get_root_dir
from envs.torch6dof import TorchWVRConfig, TorchWVRVecEnv


ALGORITHM = "option_gated_hierarchical_ppo"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--intercept-checkpoint", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=96)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--option-horizon", type=int, default=5)
    parser.add_argument(
        "--gate-history", action=argparse.BooleanOptionalAction, default=True,
        help="augment the gate with one option-boundary observation delta")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--gate-entropy-coef", type=float, default=1.0e-2)
    parser.add_argument("--gate-entropy-final-coef", type=float, default=1.0e-3)
    parser.add_argument("--option-balance-coef", type=float, default=1.0e-2)
    parser.add_argument("--option-balance-final-coef", type=float, default=0.0)
    parser.add_argument("--gate-teacher-coef", type=float, default=0.0)
    parser.add_argument("--gate-teacher-final-coef", type=float, default=0.0)
    parser.add_argument(
        "--gate-teacher-cutoff-fraction", type=float, default=1.0,
        help=("fraction of updates used to anneal teacher supervision; the "
              "remaining updates use the final coefficient"))
    parser.add_argument(
        "--gate-anchor-kl-coef", type=float, default=0.0,
        help=("after teacher cutoff, preserve the frozen warm-start policy "
              "without accessing teacher labels"))
    parser.add_argument(
        "--fire-controller",
        choices=("skill", "base", "quality-gated", "option-aware"),
        default="option-aware")
    parser.add_argument("--fire-max-range-norm", type=float, default=1.0)
    parser.add_argument("--fire-max-ao-rad", type=float, default=0.35)
    parser.add_argument("--fire-max-los-rate", type=float, default=0.12)
    parser.add_argument("--fire-min-closure-norm", type=float, default=-0.10)
    parser.add_argument("--seed", type=int, default=85)
    parser.add_argument(
        "--training-opponents", choices=("fixed-mix", "staged", "evasive"),
        default="staged")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def _portable_skill_payload(payload: dict, source: Path | str) -> dict:
    """Keep a frozen skill self-contained in the hierarchical checkpoint."""
    return {
        "model": payload["model"],
        "obs_dim": int(payload["obs_dim"]),
        "action_dims": list(payload["action_dims"]),
        "hidden_size": int(payload["hidden_size"]),
        "torch_speed_adapter": bool(payload.get("torch_speed_adapter", True)),
        "source_checkpoint": str(source),
    }


def build_frozen_skill(payload: dict, device: torch.device,
                       expected_obs_dim: int) -> MultiDiscreteActorCritic:
    """Build one immutable low-level policy from a standard PPO checkpoint."""
    obs_dim = int(payload["obs_dim"])
    if obs_dim != expected_obs_dim:
        raise ValueError(
            f"skill observation dimension {obs_dim} != {expected_obs_dim}")
    action_dims = list(payload["action_dims"])
    if action_dims != [3, 5, 3, 2]:
        raise ValueError(f"unsupported low-level action dimensions: {action_dims}")
    model = MultiDiscreteActorCritic(
        obs_dim=obs_dim, action_dims=action_dims,
        hidden_size=int(payload["hidden_size"]), fire_mask_obs_index=23,
        altitude_safety_obs_index=0,
        altitude_descend_mask_threshold=0.8,
        altitude_force_climb_threshold=0.6).to(device).eval()
    model.load_state_dict(payload["model"])
    model.requires_grad_(False)
    return model


def select_skill_actions(options: torch.Tensor, base_actions: torch.Tensor,
                         intercept_actions: torch.Tensor) -> torch.Tensor:
    """Select one complete MultiDiscrete action for every environment."""
    if base_actions.shape != intercept_actions.shape:
        raise ValueError("low-level skill action shapes must match")
    if options.ndim != 1 or options.shape[0] != base_actions.shape[0]:
        raise ValueError("options must contain one choice per environment")
    return torch.where(
        (options == 1)[:, None], intercept_actions, base_actions)


def semi_markov_td_target(reward_sum: torch.Tensor,
                          continuation_discount: torch.Tensor,
                          next_value: torch.Tensor,
                          done: torch.Tensor) -> torch.Tensor:
    """One-step SMDP bootstrap target for variable-duration options."""
    return reward_sum + continuation_discount * next_value * (~done).float()


def semi_markov_gae(reward_sum: torch.Tensor,
                    continuation_discount: torch.Tensor,
                    duration: torch.Tensor,
                    next_value: torch.Tensor,
                    old_value: torch.Tensor,
                    done: torch.Tensor,
                    env_ids: torch.Tensor,
                    gae_lambda: float):
    """Compute option-level GAE without treating option samples as IID.

    Transitions must be in chronological collection order.  Recursion is
    performed independently for each vector environment and respects both
    variable option duration and episode termination.
    """
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must lie in [0, 1]")
    delta = semi_markov_td_target(
        reward_sum, continuation_discount, next_value, done) - old_value
    # These tensors are rollout targets and carry no gradients.  Recursing on
    # CPU avoids thousands of scalar CUDA synchronizations from Tensor.item().
    continuation = (
        continuation_discount
        * torch.pow(
            torch.full_like(continuation_discount, gae_lambda),
            duration.to(continuation_discount.dtype))
        * (~done).float())
    delta_cpu = delta.detach().cpu()
    continuation_cpu = continuation.detach().cpu()
    env_ids_cpu = env_ids.detach().cpu()
    advantage_cpu = torch.zeros_like(delta_cpu)
    for env_id in torch.unique(env_ids_cpu).tolist():
        positions = torch.nonzero(
            env_ids_cpu == env_id, as_tuple=False).flatten().tolist()
        running = 0.0
        for position in reversed(positions):
            running = (
                float(delta_cpu[position])
                + float(continuation_cpu[position]) * running)
            advantage_cpu[position] = running
    advantage = advantage_cpu.to(delta.device)
    return advantage, advantage + old_value


def option_balance_loss(model: MultiDiscreteActorCritic,
                        observations: torch.Tensor) -> torch.Tensor:
    """Warm-up regularizer on marginal option use, not per-state uniformity."""
    features = model._features(observations)
    distribution = model._distributions(observations, features)[0]
    marginal = distribution.probs.mean(0)
    target = torch.full_like(marginal, 1.0 / marginal.numel())
    return (marginal - target).square().sum()


def gate_teacher_loss(model: MultiDiscreteActorCritic,
                      observations: torch.Tensor,
                      teacher_options: torch.Tensor) -> torch.Tensor:
    """Imitate curriculum skill labels only after motion history exists."""
    history_available = observations[:, -1] >= 0.5
    if not bool(history_available.any()):
        return model.value(observations).sum() * 0.0
    selected_obs = observations[history_available]
    features = model._features(selected_obs)
    distribution = model._distributions(selected_obs, features)[0]
    return -distribution.log_prob(
        teacher_options[history_available]).mean()


def scheduled_teacher_coefficient(initial: float, final: float,
                                  training_fraction: float,
                                  cutoff_fraction: float) -> float:
    """Anneal privileged warm-start labels, then leave a pure PPO phase."""
    if not 0.0 < cutoff_fraction <= 1.0:
        raise ValueError("teacher cutoff fraction must lie in (0, 1]")
    local_fraction = min(max(training_fraction / cutoff_fraction, 0.0), 1.0)
    return initial + local_fraction * (final - initial)


def gate_anchor_kl_loss(model: MultiDiscreteActorCritic,
                        anchor: MultiDiscreteActorCritic,
                        observations: torch.Tensor) -> torch.Tensor:
    """Forward KL from a frozen teacher-cutoff snapshot to the live gate."""
    features = model._features(observations)
    current = model._distributions(observations, features)[0]
    with torch.no_grad():
        anchor_features = anchor._features(observations)
        reference = anchor._distributions(observations, anchor_features)[0]
    return torch.distributions.kl_divergence(reference, current).mean()


def build_gate_observation(current: torch.Tensor,
                           previous_boundary: torch.Tensor | None = None,
                           previous_option: torch.Tensor | None = None,
                           has_history: torch.Tensor | None = None):
    """Add one-step option history without changing environment observations."""
    if previous_boundary is None:
        return current
    if previous_option is None or has_history is None:
        raise ValueError("historical gate observations require option state")
    valid = has_history.bool()
    delta = (current - previous_boundary) * valid[:, None].float()
    option_one_hot = torch.zeros(
        current.shape[0], 2, dtype=current.dtype, device=current.device)
    valid_option = valid & (previous_option >= 0)
    if bool(valid_option.any()):
        option_one_hot[valid_option, previous_option[valid_option]] = 1.0
    return torch.cat(
        (current, delta, option_one_hot, valid[:, None].float()), dim=-1)


def normalize_advantages_by_group(advantages: torch.Tensor,
                                  groups: torch.Tensor) -> torch.Tensor:
    """Give each scripted opponent family comparable PPO gradient scale."""
    result = torch.empty_like(advantages)
    for group in torch.unique(groups):
        selected = groups == group
        values = advantages[selected]
        result[selected] = (
            (values - values.mean())
            / (values.std(unbiased=False) + 1.0e-8))
    return result


def quality_gated_fire(observations: torch.Tensor, *,
                       max_range_norm: float = 1.0,
                       max_ao_rad: float = 0.35,
                       max_los_rate: float = 0.12,
                       min_closure_norm: float = -0.10) -> torch.Tensor:
    """Conservative shared fire control using only policy observations."""
    los_rate = torch.linalg.vector_norm(observations[:, 26:28], dim=-1)
    return (
        (observations[:, 23] >= 0.5)
        & (observations[:, 13] <= max_range_norm)
        & (observations[:, 11] <= max_ao_rad)
        & (los_rate <= max_los_rate)
        & (observations[:, 28] >= min_closure_norm)
    ).long()


def apply_fire_controller(actions: torch.Tensor, observations: torch.Tensor,
                          base_actions: torch.Tensor, mode: str,
                          options: torch.Tensor | None = None, **kwargs):
    """Decouple maneuver option selection from the shared firing decision."""
    result = actions.clone()
    if mode == "skill":
        return result
    if mode == "base":
        result[:, 3] = base_actions[:, 3]
    elif mode == "quality-gated":
        result[:, 3] = quality_gated_fire(observations, **kwargs)
    elif mode == "option-aware":
        if options is None:
            raise ValueError("option-aware fire control requires options")
        quality_fire = quality_gated_fire(observations, **kwargs)
        result[:, 3] = torch.where(
            options == 1, quality_fire, base_actions[:, 3])
    else:
        raise ValueError(f"unsupported fire controller: {mode}")
    return result


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def main():
    args = parse_args()
    if min(args.num_envs, args.rollout_steps, args.updates, args.epochs,
           args.minibatch_size, args.option_horizon) <= 0:
        raise ValueError("training sizes and option horizon must be positive")
    if not 0.0 < args.gamma <= 1.0 or not 0.0 <= args.gae_lambda <= 1.0:
        raise ValueError("gamma and gae-lambda must lie in valid ranges")
    if min(args.gate_entropy_coef, args.gate_entropy_final_coef,
           args.option_balance_coef, args.option_balance_final_coef,
           args.gate_teacher_coef, args.gate_teacher_final_coef,
           args.gate_anchor_kl_coef) < 0.0:
        raise ValueError("entropy and balance coefficients must be non-negative")
    if not 0.0 < args.gate_teacher_cutoff_fraction <= 1.0:
        raise ValueError("gate-teacher-cutoff-fraction must lie in (0, 1]")
    if (args.gate_teacher_coef > 0.0 or args.gate_teacher_final_coef > 0.0) \
            and not args.gate_history:
        raise ValueError("gate teacher supervision requires gate history")
    if args.gate_anchor_kl_coef > 0.0 and args.gate_teacher_coef <= 0.0:
        raise ValueError("gate anchoring requires a teacher warm-start")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    env = TorchWVRVecEnv(
        TorchWVRConfig(
            num_envs=args.num_envs, reward_mode="competitive",
            attack_quality_weight=0.0),
        device=device, seed=args.seed, compile_dynamics=True)
    base_payload = _load_checkpoint(args.base_checkpoint, device)
    intercept_payload = _load_checkpoint(args.intercept_checkpoint, device)
    base_skill = build_frozen_skill(
        base_payload, device, env.observation_size)
    intercept_skill = build_frozen_skill(
        intercept_payload, device, env.observation_size)

    gate_obs_dim = (
        2 * env.observation_size + 3
        if args.gate_history else env.observation_size)
    gate = MultiDiscreteActorCritic(
        obs_dim=gate_obs_dim, action_dims=[2],
        hidden_size=args.hidden_size).to(device)
    resumed_updates = 0
    gate_anchor = None
    if args.resume is not None:
        resume_payload = _load_checkpoint(args.resume, device)
        if resume_payload.get("algorithm") != ALGORITHM:
            raise ValueError("resume checkpoint is not hierarchical PPO")
        if int(resume_payload["obs_dim"]) != env.observation_size:
            raise ValueError("resume observation dimension mismatch")
        if int(resume_payload.get("gate_obs_dim", env.observation_size)) != gate_obs_dim:
            raise ValueError("resume gate observation dimension mismatch")
        gate.load_state_dict(resume_payload["gate_model"])
        resumed_updates = int(resume_payload.get("completed_updates", 0))
        if resume_payload.get("gate_anchor_model") is not None:
            gate_anchor = copy.deepcopy(gate).eval()
            gate_anchor.load_state_dict(resume_payload["gate_anchor_model"])
            gate_anchor.requires_grad_(False)
    optimizer = torch.optim.Adam(gate.parameters(), lr=3.0e-4, eps=1.0e-5)
    update_args = SimpleNamespace(
        epochs=args.epochs, minibatch_size=args.minibatch_size,
        clip_ratio=0.2, value_coef=0.5, entropy_coef=1.0e-3,
        max_grad_norm=2.0)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    lowlevel = BaselineActor().to(device).eval()
    lowlevel.load_state_dict(torch.load(
        get_root_dir() + "/model/baseline_model.pt",
        map_location=device, weights_only=True))
    lowlevel.requires_grad_(False)
    lowlevel.tpdv = {"dtype": torch.float32, "device": device}

    env_index = torch.arange(args.num_envs, device=device)
    learner_side = env_index.remainder(2)
    obs = env.observations()
    rnn_states = torch.zeros(args.num_envs * 2, 1, 128, device=device)
    option_remaining = torch.zeros(
        args.num_envs, dtype=torch.long, device=device)
    current_option = torch.zeros_like(option_remaining)
    active_obs = torch.zeros(args.num_envs, gate_obs_dim, device=device)
    active_log_prob = torch.zeros(args.num_envs, device=device)
    active_value = torch.zeros(args.num_envs, device=device)
    active_reward = torch.zeros(args.num_envs, device=device)
    active_discount = torch.ones(args.num_envs, device=device)
    active_duration = torch.zeros(
        args.num_envs, dtype=torch.long, device=device)
    previous_option = torch.full_like(current_option, -1)
    previous_boundary_obs = torch.zeros(
        args.num_envs, env.observation_size, device=device)
    has_gate_history = torch.zeros(
        args.num_envs, dtype=torch.bool, device=device)

    # Compile physics graphs before timing.
    with torch.inference_mode():
        learner_obs = obs[env_index, learner_side]
        warm_action, _, _ = base_skill.act(learner_obs, deterministic=True)
        warm_all = fixed_mixture_actions(obs, env_index.remainder(3))
        warm_all[env_index, learner_side] = warm_action
        warm_control, _ = lowlevel_controls(
            lowlevel, rnn_states, obs, warm_all,
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

    gamma = args.gamma
    history = []
    total_decisions = 0
    started = time.perf_counter()
    for local_update in range(1, args.updates + 1):
        update = resumed_updates + local_update
        if args.training_opponents == "staged":
            opponent_kind, opponent_stage = staged_opponent_kind(
                env_index, update, resumed_updates + args.updates)
        elif args.training_opponents == "evasive":
            opponent_kind = torch.full_like(env_index, 2)
            opponent_stage = "evasive"
        else:
            opponent_kind = env_index.remainder(3)
            opponent_stage = "fixed-mix"
        transition_obs = []
        transition_actions = []
        transition_log_probs = []
        transition_values = []
        transition_rewards = []
        transition_discounts = []
        transition_durations = []
        transition_next_values = []
        transition_dones = []
        transition_env_ids = []
        transition_opponent_kinds = []
        transition_teacher_options = []
        option_counts = torch.zeros(2, dtype=torch.long, device=device)
        option_switches = torch.zeros((), dtype=torch.long, device=device)
        completed = torch.zeros((), dtype=torch.long, device=device)
        learner_wins = torch.zeros_like(completed)
        learner_losses = torch.zeros_like(completed)
        timeouts = torch.zeros_like(completed)
        learner_launches = torch.zeros_like(completed)
        learner_hits = torch.zeros_like(completed)
        reward_total = torch.zeros((), device=device)
        rollout_started = time.perf_counter()

        for step in range(args.rollout_steps):
            learner_obs = obs[env_index, learner_side]
            new_option = option_remaining == 0
            if bool(new_option.any()):
                indices = torch.nonzero(new_option, as_tuple=False).flatten()
                gate_obs = build_gate_observation(
                    learner_obs, previous_boundary_obs, previous_option,
                    has_gate_history) if args.gate_history else learner_obs
                with torch.inference_mode():
                    sampled, log_prob, value = gate.act(gate_obs[indices])
                selected = sampled[:, 0]
                option_switches += (
                    (previous_option[indices] >= 0)
                    & (previous_option[indices] != selected)).sum()
                previous_option[indices] = selected
                current_option[indices] = selected
                option_remaining[indices] = args.option_horizon
                active_obs[indices] = gate_obs[indices]
                active_log_prob[indices] = log_prob
                active_value[indices] = value
                active_reward[indices] = 0.0
                active_discount[indices] = 1.0
                active_duration[indices] = 0
                previous_boundary_obs[indices] = learner_obs[indices]
                has_gate_history[indices] = True
                option_counts.scatter_add_(
                    0, selected, torch.ones_like(selected))

            with torch.inference_mode():
                base_action, _, _ = base_skill.act(
                    learner_obs, deterministic=True)
                intercept_action, _, _ = intercept_skill.act(
                    learner_obs, deterministic=True)
                learner_action = select_skill_actions(
                    current_option, base_action, intercept_action)
                learner_action = apply_fire_controller(
                    learner_action, learner_obs, base_action,
                    args.fire_controller, options=current_option,
                    max_range_norm=args.fire_max_range_norm,
                    max_ao_rad=args.fire_max_ao_rad,
                    max_los_rate=args.fire_max_los_rate,
                    min_closure_norm=args.fire_min_closure_norm)
                executed_action = fixed_mixture_actions(obs, opponent_kind)
                executed_action[env_index, learner_side] = learner_action
                controls, next_rnn = lowlevel_controls(
                    lowlevel, rnn_states, obs, executed_action,
                    torch_speed_adapter=True)
                next_obs, reward, done, info = env.step(
                    controls.reshape(args.num_envs, 2, 4),
                    executed_action[..., 3] == 1)

            learner_reward = reward[env_index, learner_side]
            active_reward += active_discount * learner_reward
            active_discount *= gamma
            active_duration += 1
            option_remaining -= 1
            force_boundary = step == args.rollout_steps - 1
            boundary = done | (option_remaining == 0)
            if force_boundary:
                boundary = torch.ones_like(boundary)
            if bool(boundary.any()):
                indices = torch.nonzero(boundary, as_tuple=False).flatten()
                next_learner_obs = next_obs[env_index, learner_side]
                next_gate_obs = build_gate_observation(
                    next_learner_obs, previous_boundary_obs,
                    current_option, has_gate_history
                ) if args.gate_history else next_learner_obs
                with torch.inference_mode():
                    next_value = gate.value(next_gate_obs[indices])
                transition_obs.append(active_obs[indices].clone())
                transition_actions.append(
                    current_option[indices, None].clone())
                transition_log_probs.append(active_log_prob[indices].clone())
                transition_values.append(active_value[indices].clone())
                transition_rewards.append(active_reward[indices].clone())
                transition_discounts.append(active_discount[indices].clone())
                transition_durations.append(active_duration[indices].clone())
                transition_next_values.append(next_value.clone())
                transition_dones.append(done[indices].clone())
                transition_env_ids.append(indices.clone())
                transition_opponent_kinds.append(
                    opponent_kind[indices].clone())
                transition_teacher_options.append(
                    (opponent_kind[indices] == 2).long())
                option_remaining[indices] = 0

            learner_outcome = torch.where(
                learner_side == 0, info["outcome_red"],
                -info["outcome_red"])
            completed += done.sum()
            learner_wins += ((learner_outcome > 0) & done).sum()
            learner_losses += ((learner_outcome < 0) & done).sum()
            timeouts += info["timeout"].sum()
            learner_launches += info["accepted_launches"][
                env_index, learner_side].sum()
            learner_hits += info["hits"][env_index, learner_side].sum()
            reward_total += learner_reward.sum()
            previous_option = torch.where(
                done, torch.full_like(previous_option, -1), previous_option)
            previous_boundary_obs = torch.where(
                done[:, None], torch.zeros_like(previous_boundary_obs),
                previous_boundary_obs)
            has_gate_history = torch.where(
                done, torch.zeros_like(has_gate_history), has_gate_history)
            option_remaining = torch.where(
                done, torch.zeros_like(option_remaining), option_remaining)
            rnn_states = torch.where(
                done.repeat_interleave(2)[:, None, None],
                torch.zeros_like(next_rnn), next_rnn)
            obs = next_obs

        torch.cuda.synchronize()
        rollout_s = time.perf_counter() - rollout_started
        observations = torch.cat(transition_obs)
        actions = torch.cat(transition_actions)
        old_log_probs = torch.cat(transition_log_probs)
        old_values = torch.cat(transition_values)
        teacher_options = torch.cat(transition_teacher_options)
        opponent_kinds_for_transitions = torch.cat(
            transition_opponent_kinds)
        advantages, returns = semi_markov_gae(
            torch.cat(transition_rewards),
            torch.cat(transition_discounts),
            torch.cat(transition_durations),
            torch.cat(transition_next_values), old_values,
            torch.cat(transition_dones), torch.cat(transition_env_ids),
            args.gae_lambda)
        advantages = normalize_advantages_by_group(
            advantages, opponent_kinds_for_transitions)
        with torch.inference_mode():
            teacher_prediction, _, _ = gate.act(
                observations, deterministic=True)
            teacher_prediction = teacher_prediction[:, 0]
            history_mask = observations[:, -1] >= 0.5
            teacher_accuracy = float(
                (teacher_prediction[history_mask]
                 == teacher_options[history_mask]).float().mean()
            ) if bool(history_mask.any()) else float("nan")
            teacher_accuracy_by_kind = []
            for kind_index in range(3):
                selected = history_mask & (
                    opponent_kinds_for_transitions == kind_index)
                teacher_accuracy_by_kind.append(
                    float((teacher_prediction[selected]
                           == teacher_options[selected]).float().mean())
                    if bool(selected.any()) else float("nan"))
        rollout = Rollout(
            observations=observations, actions=actions,
            old_log_probs=old_log_probs, old_values=old_values,
            returns=returns, advantages=advantages,
            mean_reward=float(reward_total / (args.num_envs * args.rollout_steps)),
            mean_abs_reward=float("nan"), zero_sum_error=0.0,
            completed_episodes=int(completed),
            mean_abs_completed_return=float("nan"),
            accepted_launches=int(learner_launches))
        update_started = time.perf_counter()
        schedule_fraction = (
            (local_update - 1) / max(args.updates - 1, 1))
        entropy_coef = (
            args.gate_entropy_coef
            + schedule_fraction * (
                args.gate_entropy_final_coef - args.gate_entropy_coef))
        balance_coef = (
            args.option_balance_coef
            + schedule_fraction * (
                args.option_balance_final_coef - args.option_balance_coef))
        teacher_coef = scheduled_teacher_coefficient(
            args.gate_teacher_coef, args.gate_teacher_final_coef,
            schedule_fraction, args.gate_teacher_cutoff_fraction)
        if (args.gate_anchor_kl_coef > 0.0 and teacher_coef == 0.0
                and gate_anchor is None):
            gate_anchor = copy.deepcopy(gate).eval()
            gate_anchor.requires_grad_(False)
        anchor_coef = (
            args.gate_anchor_kl_coef if gate_anchor is not None else 0.0)
        update_args.entropy_coef = entropy_coef
        auxiliary_fn = None
        if balance_coef > 0.0 or teacher_coef > 0.0 or anchor_coef > 0.0:
            def auxiliary_fn(model, observations, teacher_options):
                loss = (
                    balance_coef * option_balance_loss(model, observations)
                    + teacher_coef * gate_teacher_loss(
                        model, observations, teacher_options))
                if anchor_coef > 0.0:
                    loss = loss + anchor_coef * gate_anchor_kl_loss(
                        model, gate_anchor, observations)
                return loss
        metrics = ppo_update(
            gate, optimizer, rollout, update_args, scaler, True,
            auxiliary_loss_fn=auxiliary_fn,
            auxiliary_coef=1.0 if auxiliary_fn is not None else 0.0,
            auxiliary_targets=teacher_options)
        torch.cuda.synchronize()
        update_s = time.perf_counter() - update_started
        decisions = args.num_envs * args.rollout_steps
        total_decisions += decisions
        total_options = option_counts.sum().clamp_min(1)
        entry = {
            "update": update,
            "opponent_stage": opponent_stage,
            "environment_decisions": total_decisions,
            "high_level_transitions": int(observations.shape[0]),
            "completed_episodes": int(completed),
            "learner_wins": int(learner_wins),
            "learner_losses": int(learner_losses),
            "timeouts": int(timeouts),
            "learner_accepted_launches": int(learner_launches),
            "learner_hits": int(learner_hits),
            "mean_learning_reward": rollout.mean_reward,
            "base_option_fraction": float(option_counts[0] / total_options),
            "intercept_option_fraction": float(option_counts[1] / total_options),
            "option_switches": int(option_switches),
            "gate_entropy_coef": entropy_coef,
            "option_balance_coef": balance_coef,
            "gate_teacher_coef": teacher_coef,
            "gate_anchor_kl_coef": anchor_coef,
            "gate_teacher_accuracy_preupdate": teacher_accuracy,
            "gate_teacher_accuracy_by_kind_preupdate": (
                teacher_accuracy_by_kind),
            "rollout_s": rollout_s,
            "ppo_update_s": update_s,
            **metrics,
        }
        history.append(entry)
        print(json.dumps(entry), flush=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    report = {
        "algorithm": ALGORITHM,
        "algorithm_variant": (
            "teacher_warmstart_anchored_ppo"
            if args.gate_anchor_kl_coef > 0.0
            else "teacher_warmstart_then_pure_ppo"
            if args.gate_teacher_coef > 0.0
            and args.gate_teacher_final_coef == 0.0
            and args.gate_teacher_cutoff_fraction < 1.0
            else "teacher_assisted" if args.gate_teacher_coef > 0.0
            or args.gate_teacher_final_coef > 0.0 else "pure_ppo"),
        "scientific_result": False,
        "environment_frozen": True,
        "low_level_skills_frozen": True,
        "high_level_optimizer": "PPO",
        "high_level_return_estimator": "semi_markov_gae",
        "gate_history": args.gate_history,
        "gate_obs_dim": gate_obs_dim,
        "advantage_normalization": "per_opponent_kind",
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "gate_entropy_schedule": {
            "initial": args.gate_entropy_coef,
            "final": args.gate_entropy_final_coef,
        },
        "option_balance_schedule": {
            "initial": args.option_balance_coef,
            "final": args.option_balance_final_coef,
        },
        "gate_teacher_schedule": {
            "initial": args.gate_teacher_coef,
            "final": args.gate_teacher_final_coef,
            "cutoff_fraction": args.gate_teacher_cutoff_fraction,
        },
        "gate_anchor_kl_coef": args.gate_anchor_kl_coef,
        "fire_controller": args.fire_controller,
        "fire_controller_config": {
            "max_range_norm": args.fire_max_range_norm,
            "max_ao_rad": args.fire_max_ao_rad,
            "max_los_rate": args.fire_max_los_rate,
            "min_closure_norm": args.fire_min_closure_norm,
        },
        "option_names": ["base_ppo", "los_intercept"],
        "option_horizon": args.option_horizon,
        "base_checkpoint": str(args.base_checkpoint),
        "intercept_checkpoint": str(args.intercept_checkpoint),
        "training_opponents": args.training_opponents,
        "reward_variant": "competitive",
        "seed": args.seed,
        "num_envs": args.num_envs,
        "sim_freq_hz": 60,
        "physics_substeps_per_decision": 12,
        "updates": args.updates,
        "completed_updates": resumed_updates + args.updates,
        "rollout_steps": args.rollout_steps,
        "total_environment_decisions": total_decisions,
        "elapsed_s": elapsed,
        "environment_decisions_per_s": total_decisions / elapsed,
        "environment_config": asdict(env.config),
        "history": history,
    }
    rendered = json.dumps(report, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "algorithm": ALGORITHM,
            "algorithm_variant": (
                "teacher_warmstart_anchored_ppo"
                if args.gate_anchor_kl_coef > 0.0
                else "teacher_warmstart_then_pure_ppo"
                if args.gate_teacher_coef > 0.0
                and args.gate_teacher_final_coef == 0.0
                and args.gate_teacher_cutoff_fraction < 1.0
                else "teacher_assisted" if args.gate_teacher_coef > 0.0
                or args.gate_teacher_final_coef > 0.0 else "pure_ppo"),
            "gate_model": gate.state_dict(),
            "gate_anchor_model": (
                gate_anchor.state_dict() if gate_anchor is not None else None),
            "obs_dim": env.observation_size,
            "gate_obs_dim": gate_obs_dim,
            "gate_history": args.gate_history,
            "gate_action_dims": [2],
            "hidden_size": args.hidden_size,
            "option_names": ["base_ppo", "los_intercept"],
            "option_horizon": args.option_horizon,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "fire_controller": args.fire_controller,
            "fire_controller_config": {
                "max_range_norm": args.fire_max_range_norm,
                "max_ao_rad": args.fire_max_ao_rad,
                "max_los_rate": args.fire_max_los_rate,
                "min_closure_norm": args.fire_min_closure_norm,
            },
            "base_skill": _portable_skill_payload(
                base_payload, args.base_checkpoint),
            "intercept_skill": _portable_skill_payload(
                intercept_payload, args.intercept_checkpoint),
            "seed": args.seed,
            "completed_updates": resumed_updates + args.updates,
            "training_environment_decisions": total_decisions,
            "training_opponents": args.training_opponents,
            "reward_variant": "competitive",
            "torch_speed_adapter": True,
            "environment_config": asdict(env.config),
        }, args.checkpoint)
    print(rendered)


if __name__ == "__main__":
    main()
