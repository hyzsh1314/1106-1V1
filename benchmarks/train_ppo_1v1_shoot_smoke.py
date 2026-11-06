"""Small hierarchical PPO smoke test for 1v1 missile combat.

This is deliberately a correctness/ integration test, not a converged
scientific training run.  Following LAG, PPO selects altitude, heading,
velocity and fire commands while a frozen recurrent low-level controller
converts the first three commands into physical flight controls.  Both
aircraft share one decentralised high-level policy and contribute on-policy
samples to the same PPO update.
"""

from __future__ import annotations

import argparse
from functools import partial
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from envs.JSBSim.envs.singlecombat_env import SingleCombatEnv
from envs.env_wrappers import DummyVecEnv, SubprocVecEnv


SCENARIO = "1v1/ShootMissile/HierarchySelfplay"
DEFAULT_REWARD_VARIANT = "R1_attack_pbrs"

REWARD_VARIANT_OVERRIDES = {
    "R0_terminal": {
        "RewardVariant": "R0_terminal",
        "FormalRewardContract_version": "WVR-1V1-V5-R0-terminal-timeout-zero",
        "TeamOutcomeReward_scale": 1.0,
        "TimeoutOutcomeReward_scale": 0.0,
        "PostureReward_scale": 0.0,
        "AltitudeReward_scale": 0.0,
        "ShootPenaltyReward_scale": 0.0,
    },
    "R1_attack_pbrs": {
        "RewardVariant": "R1_attack_pbrs",
        "FormalRewardContract_version": "WVR-1V1-V5-R1-attack-PBRS-timeout-zero",
        "TeamOutcomeReward_scale": 1.0,
        "TimeoutOutcomeReward_scale": 0.0,
        "PostureReward_scale": 0.1,
        "AltitudeReward_scale": 0.0,
        "ShootPenaltyReward_scale": 0.0,
    },
    "R1_timeout025": {
        "RewardVariant": "R1_timeout025",
        "FormalRewardContract_version": (
            "WVR-1V1-V6-R1-attack-PBRS-timeout-adjudicated-025"),
        "TeamOutcomeReward_scale": 1.0,
        "TimeoutOutcomeReward_scale": 0.25,
        "PostureReward_scale": 0.1,
        "AltitudeReward_scale": 0.0,
        "ShootPenaltyReward_scale": 0.0,
    },
    "R2_threat_pbrs": {
        "RewardVariant": "R2_threat_pbrs",
        "FormalRewardContract_version": "WVR-1V1-V5-R2-threat-PBRS-timeout-zero",
        "TeamOutcomeReward_scale": 1.0,
        "TimeoutOutcomeReward_scale": 0.0,
        "PostureReward_scale": 0.1,
        "ThreatConditionedPotentialReward_scale": 0.1,
        "AltitudeReward_scale": 0.0,
        "ShootPenaltyReward_scale": 0.0,
    },
    "R3_constraints": {
        "RewardVariant": "R3_constraints",
        "FormalRewardContract_version": "WVR-1V1-V5-R3-constraints-timeout-zero",
        "TeamOutcomeReward_scale": 1.0,
        "TimeoutOutcomeReward_scale": 0.0,
        "PostureReward_scale": 0.1,
        "ThreatConditionedPotentialReward_scale": 0.1,
        "AltitudeReward_scale": 0.001,
        "ShootPenaltyReward_scale": 0.001,
    },
}


def make_environment(seed: int, scenario: str = SCENARIO,
                     reward_variant: str | None = None,
                     threat_observation_v2: bool = False,
                     heading_command_span_deg: float = 30.0,
                     threat_relative_break_skill: bool = False,
                     threat_break_option: bool = False,
                     threat_break_min_steps: int = 10,
                     threat_break_clear_steps: int = 5,
                     threat_break_recovery_steps: int = 10,
                     threat_break_recovery_ta_max_deg: float = 180.0,
                     threat_break_pulse: bool = False,
                     threat_break_pulse_steps: int = 4,
                     threat_break_pulse_urgency: float = 0.5):
    """Top-level factory so spawn-based WSL/Windows workers can pickle it."""
    overrides = ({} if reward_variant is None else
                 dict(REWARD_VARIANT_OVERRIDES[reward_variant]))
    if threat_observation_v2:
        overrides.update({
            "ThreatObservation_v2_enabled": True,
            "ThreatObservation_time_to_go_horizon_s": 30.0,
        })
    if float(heading_command_span_deg) != 30.0:
        overrides["Hierarchical_heading_command_span_deg"] = float(
            heading_command_span_deg)
    if threat_relative_break_skill:
        overrides["ThreatRelativeBreakSkill_enabled"] = True
    if threat_break_option:
        overrides.update({
            "ThreatRelativeBreakOption_enabled": True,
            "ThreatRelativeBreakOption_min_steps": int(
                threat_break_min_steps),
            "ThreatRelativeBreakOption_clear_steps": int(
                threat_break_clear_steps),
            "ThreatRelativeBreakOption_recovery_steps": int(
                threat_break_recovery_steps),
            "ThreatRelativeBreakOption_recovery_ta_max_deg": float(
                threat_break_recovery_ta_max_deg),
        })
    if threat_break_pulse:
        overrides.update({
            "ThreatRelativeBreakPulse_enabled": True,
            "ThreatRelativeBreakPulse_steps": int(
                threat_break_pulse_steps),
            "ThreatRelativeBreakPulse_urgency_threshold": float(
                threat_break_pulse_urgency),
        })
    env = SingleCombatEnv(
        scenario, reuse_fdm=True, config_overrides=overrides)
    env.seed(int(seed))
    return env


class MultiDiscreteActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dims: list[int], hidden_size: int,
                 fire_mask_obs_index: int | None = None,
                 altitude_safety_obs_index: int | None = None,
                 altitude_descend_mask_threshold: float = 0.8,
                 altitude_force_climb_threshold: float = 0.65):
        super().__init__()
        self.fire_mask_obs_index = fire_mask_obs_index
        self.altitude_safety_obs_index = altitude_safety_obs_index
        self.altitude_descend_mask_threshold = float(
            altitude_descend_mask_threshold)
        self.altitude_force_climb_threshold = float(
            altitude_force_climb_threshold)
        if (altitude_safety_obs_index is not None and
                (not action_dims or int(action_dims[0]) < 3)):
            raise ValueError(
                "altitude safety mask requires a three-command altitude head")
        if (self.altitude_force_climb_threshold >
                self.altitude_descend_mask_threshold):
            raise ValueError(
                "force-climb threshold cannot exceed descend-mask threshold")
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.action_heads = nn.ModuleList(
            nn.Linear(hidden_size, action_dim) for action_dim in action_dims
        )
        self.value_head = nn.Linear(hidden_size, 1)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, np.sqrt(2.0))
                nn.init.constant_(module.bias, 0.0)
        for head in self.action_heads:
            nn.init.orthogonal_(head.weight, 0.01)
        nn.init.orthogonal_(self.value_head.weight, 1.0)

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.backbone(obs)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(self._features(obs)).squeeze(-1)

    def _distributions(self, obs: torch.Tensor, features: torch.Tensor):
        logits = [head(features) for head in self.action_heads]
        if self.altitude_safety_obs_index is not None:
            altitude_logits = logits[0].clone()
            altitude = obs[:, self.altitude_safety_obs_index]
            below_safe = altitude <= self.altitude_descend_mask_threshold
            below_critical = altitude <= self.altitude_force_climb_threshold
            impossible = torch.finfo(altitude_logits.dtype).min
            # Hierarchical command 0 climbs, 1 holds altitude and 2 descends.
            altitude_logits[:, 2] = altitude_logits[:, 2].masked_fill(
                below_safe, impossible)
            altitude_logits[:, 1] = altitude_logits[:, 1].masked_fill(
                below_critical, impossible)
            logits[0] = altitude_logits
        if self.fire_mask_obs_index is not None:
            fire_logits = logits[-1].clone()
            legal = obs[:, self.fire_mask_obs_index] >= 0.5
            fire_logits[:, 1] = fire_logits[:, 1].masked_fill(
                ~legal, torch.finfo(fire_logits.dtype).min)
            logits[-1] = fire_logits
        return [Categorical(logits=value) for value in logits]

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        features = self._features(obs)
        distributions = self._distributions(obs, features)
        actions = torch.stack([
            (torch.argmax(distribution.logits, dim=-1)
             if deterministic else distribution.sample())
            for distribution in distributions
        ], dim=-1)
        log_prob = torch.stack([
            distribution.log_prob(actions[:, index])
            for index, distribution in enumerate(distributions)
        ], dim=-1).sum(-1)
        value = self.value_head(features).squeeze(-1)
        return actions, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        features = self._features(obs)
        distributions = self._distributions(obs, features)
        log_prob = torch.stack([
            distribution.log_prob(actions[:, index])
            for index, distribution in enumerate(distributions)
        ], dim=-1).sum(-1)
        entropy = torch.stack([
            distribution.entropy() for distribution in distributions
        ], dim=-1).sum(-1)
        value = self.value_head(features).squeeze(-1)
        return log_prob, entropy, value


@dataclass
class Rollout:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    mean_reward: float
    mean_abs_reward: float
    zero_sum_error: float
    completed_episodes: int
    mean_abs_completed_return: float
    accepted_launches: int


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--start-method", choices=("spawn", "forkserver"),
                        default="spawn")
    parser.add_argument(
        "--envs-per-worker", type=int, default=1,
        help=("number of independent environments stepped serially by each "
              "worker process; this changes only process topology"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=1.0e-3)
    parser.add_argument("--max-grad-norm", type=float, default=2.0)
    parser.add_argument("--scenario", default=SCENARIO)
    parser.add_argument(
        "--reward-variant", choices=tuple(REWARD_VARIANT_OVERRIDES),
        default=DEFAULT_REWARD_VARIANT)
    # Seed 85 is a reproducible standard-config geometry that starts one
    # aircraft inside the legal launch envelope, so this short smoke run also
    # exercises the weapon branch without relaxing any firing constraint.
    parser.add_argument("--seed", type=int, default=85)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"),
                        default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Use CUDA automatic mixed precision for PPO updates")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but this PyTorch build has no CUDA")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def parameter_vector(model: nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().cpu().reshape(-1)
                      for parameter in model.parameters()])


def count_launch_events(infos) -> int:
    count = 0
    for info in np.asarray(infos, dtype=object).reshape(-1):
        if not isinstance(info, dict):
            continue
        count += sum(
            event.get("event_type") == "launch"
            for event in info.get("weapon_events", ())
            if isinstance(event, dict)
        )
    return int(count)


def collect_rollout(env, model, obs, args, device):
    observations = []
    actions = []
    log_probs = []
    values = []
    rewards = []
    dones = []
    episode_returns = np.zeros(
        (env.num_envs, env.num_agents), dtype=np.float64)
    completed_returns = []
    completed_episodes = 0
    accepted_launches = 0
    zero_sum_error = 0.0

    for _ in range(args.rollout_steps):
        obs_tensor = torch.as_tensor(
            obs.reshape(-1, obs.shape[-1]),
            dtype=torch.float32, device=device)
        with torch.inference_mode():
            action_tensor, log_prob_tensor, value_tensor = model.act(obs_tensor)
        action = action_tensor.cpu().numpy().reshape(
            env.num_envs, env.num_agents, -1)

        next_obs, reward, done, infos = env.step(action)
        accepted_launches += count_launch_events(infos)

        reward = np.asarray(reward, dtype=np.float32).reshape(
            env.num_envs, env.num_agents)
        done = np.asarray(done, dtype=np.bool_).reshape(
            env.num_envs, env.num_agents)
        zero_sum_error = max(
            zero_sum_error,
            float(np.max(np.abs(reward.sum(axis=-1)))),
        )
        episode_returns += reward

        observations.append(np.asarray(obs, dtype=np.float32))
        actions.append(action.astype(np.int64))
        log_probs.append(log_prob_tensor.cpu().numpy().reshape(
            env.num_envs, env.num_agents))
        values.append(value_tensor.cpu().numpy().reshape(
            env.num_envs, env.num_agents))
        rewards.append(reward)
        dones.append(done)

        obs = next_obs
        completed = np.all(done, axis=1)
        if np.any(completed):
            completed_returns.extend(episode_returns[completed].reshape(-1).tolist())
            completed_episodes += int(np.count_nonzero(completed))
            episode_returns[completed] = 0.0

    with torch.inference_mode():
        next_value = model.value(torch.as_tensor(
            obs.reshape(-1, obs.shape[-1]), dtype=torch.float32,
            device=device)).cpu().numpy().reshape(
                env.num_envs, env.num_agents)

    rewards_array = np.asarray(rewards, dtype=np.float32)
    dones_array = np.asarray(dones, dtype=np.float32)
    values_array = np.asarray(values, dtype=np.float32)
    advantages = np.zeros_like(rewards_array)
    gae = np.zeros((env.num_envs, env.num_agents), dtype=np.float32)

    for step in reversed(range(args.rollout_steps)):
        if step == args.rollout_steps - 1:
            following_value = next_value
        else:
            following_value = values_array[step + 1]
        nonterminal = 1.0 - dones_array[step]
        delta = (rewards_array[step]
                 + args.gamma * following_value * nonterminal
                 - values_array[step])
        gae = delta + args.gamma * args.gae_lambda * nonterminal * gae
        advantages[step] = gae

    returns = advantages + values_array
    flat_advantages = advantages.reshape(-1)
    flat_advantages = (
        flat_advantages - flat_advantages.mean()
    ) / (flat_advantages.std() + 1.0e-8)

    rollout = Rollout(
        observations=torch.as_tensor(
            np.asarray(observations).reshape(-1, obs.shape[-1]),
            dtype=torch.float32, device=device),
        actions=torch.as_tensor(
            np.asarray(actions).reshape(-1, action.shape[-1]),
            dtype=torch.long, device=device),
        old_log_probs=torch.as_tensor(
            np.asarray(log_probs).reshape(-1), dtype=torch.float32,
            device=device),
        old_values=torch.as_tensor(
            values_array.reshape(-1), dtype=torch.float32, device=device),
        returns=torch.as_tensor(
            returns.reshape(-1), dtype=torch.float32, device=device),
        advantages=torch.as_tensor(
            flat_advantages, dtype=torch.float32, device=device),
        mean_reward=float(rewards_array.mean()),
        mean_abs_reward=float(np.abs(rewards_array).mean()),
        zero_sum_error=float(zero_sum_error),
        completed_episodes=completed_episodes,
        mean_abs_completed_return=(float(np.mean(np.abs(completed_returns)))
                                   if completed_returns else float("nan")),
        accepted_launches=accepted_launches,
    )
    return rollout, obs


def ppo_update(model, optimizer, rollout, args, scaler, amp_enabled,
               auxiliary_loss_fn=None, auxiliary_coef: float = 0.0,
               auxiliary_targets=None):
    sample_count = rollout.observations.shape[0]
    metrics = {"policy_loss": [], "value_loss": [], "entropy": [],
               "approx_kl": [], "clip_fraction": [], "grad_norm": []}
    if auxiliary_loss_fn is not None and auxiliary_coef > 0.0:
        metrics["auxiliary_loss"] = []

    for _ in range(args.epochs):
        permutation = torch.randperm(sample_count,
                                     device=rollout.observations.device)
        for start in range(0, sample_count, args.minibatch_size):
            indices = permutation[start:start + args.minibatch_size]
            with torch.autocast(
                    device_type="cuda", dtype=torch.float16,
                    enabled=amp_enabled):
                new_log_prob, entropy, new_value = model.evaluate_actions(
                    rollout.observations[indices], rollout.actions[indices])
                log_ratio = new_log_prob - rollout.old_log_probs[indices]
                ratio = log_ratio.exp()
                advantage = rollout.advantages[indices]
                unclipped = ratio * advantage
                clipped = torch.clamp(
                    ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio
                ) * advantage
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_delta = new_value - rollout.old_values[indices]
                clipped_value = (rollout.old_values[indices]
                                 + value_delta.clamp(-args.clip_ratio,
                                                     args.clip_ratio))
                value_loss = 0.5 * torch.max(
                    (new_value - rollout.returns[indices]).square(),
                    (clipped_value - rollout.returns[indices]).square(),
                ).mean()
                entropy_mean = entropy.mean()
                loss = (policy_loss + args.value_coef * value_loss
                        - args.entropy_coef * entropy_mean)
                auxiliary_loss = None
                if auxiliary_loss_fn is not None and auxiliary_coef > 0.0:
                    if auxiliary_targets is None:
                        auxiliary_loss = auxiliary_loss_fn(
                            model, rollout.observations[indices])
                    else:
                        auxiliary_loss = auxiliary_loss_fn(
                            model, rollout.observations[indices],
                            auxiliary_targets[indices])
                    loss = loss + auxiliary_coef * auxiliary_loss

            if not torch.isfinite(loss):
                raise RuntimeError("PPO produced a non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(),
                                                 args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs()
                                 > args.clip_ratio).float().mean()
            metrics["policy_loss"].append(float(policy_loss.detach()))
            metrics["value_loss"].append(float(value_loss.detach()))
            metrics["entropy"].append(float(entropy_mean.detach()))
            metrics["approx_kl"].append(float(approx_kl.detach()))
            metrics["clip_fraction"].append(float(clip_fraction.detach()))
            metrics["grad_norm"].append(float(grad_norm.detach()))
            if auxiliary_loss is not None:
                metrics["auxiliary_loss"].append(
                    float(auxiliary_loss.detach()))

    return {name: float(np.mean(values)) for name, values in metrics.items()}


def main():
    args = parse_args()
    if (args.updates <= 0 or args.rollout_steps <= 0 or args.epochs <= 0 or
            args.num_envs <= 0 or args.minibatch_size <= 0):
        raise ValueError(
            "updates, rollout steps, epochs, num-envs and minibatch size "
            "must be positive")
    if args.envs_per_worker <= 0:
        raise ValueError("envs-per-worker must be positive")
    if args.num_envs > 1 and args.num_envs % args.envs_per_worker:
        raise ValueError("num-envs must be divisible by envs-per-worker")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    probe = make_environment(
        args.seed, scenario=args.scenario,
        reward_variant=args.reward_variant)
    probe_obs = probe.reset()
    action_dims = [*probe.action_space[0].nvec.tolist(),
                   int(probe.action_space[1].n)]
    num_agents = int(probe.num_agents)
    lowlevel_parameters = list(probe.task.lowlevel_policy.parameters())
    if probe.task.lowlevel_policy.training or any(
            parameter.requires_grad for parameter in lowlevel_parameters):
        probe.close()
        raise RuntimeError("the hierarchical low-level controller is not frozen")
    initial_lowlevel_parameters = parameter_vector(probe.task.lowlevel_policy)

    factories = [
        partial(
            make_environment, args.seed + index * 1000,
            scenario=args.scenario, reward_variant=args.reward_variant)
        for index in range(args.num_envs)
    ]
    if args.num_envs == 1:
        env = DummyVecEnv(factories)
    else:
        env = SubprocVecEnv(
            factories, context=args.start_method,
            in_series=args.envs_per_worker)
    obs = env.reset()
    model = MultiDiscreteActorCritic(
        obs_dim=int(probe_obs.shape[-1]), action_dims=action_dims,
        hidden_size=args.hidden_size,
        fire_mask_obs_index=(23 if probe_obs.shape[-1] >= 24 else None),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 eps=1.0e-5)
    initial_parameters = parameter_vector(model)

    print(
        "scenario={0} obs={1} high_level_action={2} envs={3} agents={4} "
        "device={5} amp={6} lowlevel=frozen_recurrent".format(
            SCENARIO, tuple(obs.shape), action_dims, args.num_envs,
            num_agents, device, amp_enabled))
    start_time = time.perf_counter()
    total_decisions = 0
    total_accepted_launches = 0
    try:
        for update in range(1, args.updates + 1):
            rollout, obs = collect_rollout(env, model, obs, args, device)
            metrics = ppo_update(
                model, optimizer, rollout, args, scaler, amp_enabled)
            total_decisions += args.rollout_steps * args.num_envs
            total_accepted_launches += rollout.accepted_launches
            print(
                "update={0}/{1} samples={2} reward={3:+.6f} "
                "abs_reward={4:.6f} episodes={5} abs_ep_return={6:.3f} "
                "launches={7} "
                "policy_loss={8:+.5f} value_loss={9:.5f} "
                "entropy={10:.4f} kl={11:.6f} clip={12:.3f} "
                "grad={13:.4f} zero_sum_err={14:.3e}".format(
                    update, args.updates, rollout.observations.shape[0],
                    rollout.mean_reward, rollout.mean_abs_reward,
                    rollout.completed_episodes,
                    rollout.mean_abs_completed_return,
                    rollout.accepted_launches,
                    metrics["policy_loss"], metrics["value_loss"],
                    metrics["entropy"], metrics["approx_kl"],
                    metrics["clip_fraction"], metrics["grad_norm"],
                    rollout.zero_sum_error,
                )
            )
    finally:
        env.close()
        probe.close()

    elapsed = time.perf_counter() - start_time
    parameter_change = torch.linalg.vector_norm(
        parameter_vector(model) - initial_parameters).item()
    lowlevel_parameter_change = torch.linalg.vector_norm(
        parameter_vector(probe.task.lowlevel_policy)
        - initial_lowlevel_parameters).item()
    if not np.isfinite(parameter_change) or parameter_change <= 0.0:
        raise RuntimeError("PPO did not change the policy parameters")
    if lowlevel_parameter_change != 0.0:
        raise RuntimeError("training changed the frozen low-level controller")
    if total_accepted_launches <= 0:
        raise RuntimeError("smoke training did not exercise an accepted launch")
    print(
        "PASS decisions={0} agent_transitions={1} elapsed_s={2:.3f} "
        "decisions_per_s={3:.1f} launches={4} "
        "highlevel_parameter_l2_change={5:.6f} "
        "lowlevel_parameter_l2_change={6:.6f}".format(
            total_decisions, total_decisions * num_agents, elapsed,
            total_decisions / max(elapsed, 1.0e-9),
            total_accepted_launches, parameter_change,
            lowlevel_parameter_change,
        )
    )


if __name__ == "__main__":
    main()
