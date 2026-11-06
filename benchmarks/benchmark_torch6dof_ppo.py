"""End-to-end PPO throughput ceiling for the experimental Torch 6-DoF path.

This harness includes the project's high-level action heads, frozen recurrent
low-level controller, 12x60 Hz dynamics, 24-D air-combat observations,
zero-sum posture PBRS, GAE and PPO updates.  It intentionally omits missiles;
therefore it is a training-pipeline speed ceiling, not a scientific result.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from benchmarks.train_ppo_1v1_shoot_smoke import (
    MultiDiscreteActorCritic,
    Rollout,
    ppo_update,
)
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.utils.utils import get_root_dir
from envs.torch6dof import (
    CompiledTorchF16Runner, TorchF16BatchDynamics, TorchWVRMissileBatch,
)
from envs.torch6dof.f16_dynamics import quaternion_to_matrix


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=96)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=85)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--include-missiles",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def reset_states(dynamics, num_envs, device, generator):
    separation = 8000.0 + 8000.0 * torch.rand(
        num_envs, device=device, generator=generator)
    bearing = 2.0 * torch.pi * torch.rand(
        num_envs, device=device, generator=generator)
    direction = torch.stack((torch.cos(bearing), torch.sin(bearing)), dim=-1)
    positions = torch.zeros(num_envs, 2, 3, device=device)
    positions[:, 0, :2] = -0.5 * separation[:, None] * direction
    positions[:, 1, :2] = 0.5 * separation[:, None] * direction
    altitude = 5000.0 + 2500.0 * torch.rand(
        num_envs, 2, device=device, generator=generator)
    positions[..., 2] = -altitude
    headings = torch.stack((bearing, bearing + torch.pi), dim=-1)
    speeds = 198.0 + 76.0 * torch.rand(
        num_envs, 2, device=device, generator=generator)
    return dynamics.level_state(
        positions.reshape(-1, 3), headings.reshape(-1),
        speeds.reshape(-1)).reshape(num_envs, 2, -1)


def state_geometry(state):
    rotation = quaternion_to_matrix(state[..., 6:10])
    velocity_ned = torch.matmul(
        rotation, state[..., 3:6].unsqueeze(-1)).squeeze(-1)
    enemy_state = state.flip(1)
    enemy_rotation = rotation.flip(1)
    enemy_velocity_ned = velocity_ned.flip(1)
    relative = enemy_state[..., :3] - state[..., :3]
    distance = torch.linalg.vector_norm(relative, dim=-1).clamp_min(1.0)
    speed = torch.linalg.vector_norm(velocity_ned, dim=-1).clamp_min(1.0)
    enemy_speed = torch.linalg.vector_norm(
        enemy_velocity_ned, dim=-1).clamp_min(1.0)
    ao = torch.acos(((relative * velocity_ned).sum(-1) /
                     (distance * speed)).clamp(-1.0, 1.0))
    ta = torch.acos(((-relative * enemy_velocity_ned).sum(-1) /
                     (distance * enemy_speed)).clamp(-1.0, 1.0))
    side = torch.sign(
        velocity_ned[..., 0] * relative[..., 1]
        - velocity_ned[..., 1] * relative[..., 0])
    return rotation, velocity_ned, distance, ao, ta, side


def observations(state):
    rotation, velocity_ned, distance, ao, ta, side = state_geometry(state)
    roll = torch.atan2(rotation[..., 2, 1], rotation[..., 2, 2])
    pitch = torch.asin((-rotation[..., 2, 0]).clamp(-1.0, 1.0))
    enemy = state.flip(1)
    obs = torch.zeros(*state.shape[:2], 24, device=state.device)
    obs[..., 0] = -state[..., 2] / 5000.0
    obs[..., 1] = torch.sin(roll)
    obs[..., 2] = torch.cos(roll)
    obs[..., 3] = torch.sin(pitch)
    obs[..., 4] = torch.cos(pitch)
    obs[..., 5:8] = state[..., 3:6] / 340.0
    obs[..., 8] = torch.linalg.vector_norm(
        state[..., 3:6], dim=-1) / 340.0
    obs[..., 9] = (enemy[..., 3] - state[..., 3]) / 340.0
    obs[..., 10] = (-enemy[..., 2] + state[..., 2]) / 1000.0
    obs[..., 11] = ao
    obs[..., 12] = ta
    obs[..., 13] = distance / 10000.0
    obs[..., 14] = side
    obs[..., 21] = 1.0
    obs[..., 22] = 1.0
    obs[..., 23] = ((distance <= 14000.0) &
                    (ao <= torch.deg2rad(torch.tensor(
                        45.0, device=state.device)))).float()
    return obs.clamp(-10.0, 10.0)


def posture_potential(state):
    _, _, distance, ao, ta, _ = state_geometry(state)
    safe_argument = (1.0 - torch.maximum(
        2.0 * ta / torch.pi,
        torch.full_like(ta, 1.0e-4))).clamp(-0.9999, 0.9999)
    orientation = (
        1.0 / (50.0 * ao / torch.pi + 2.0) + 1.0
        + torch.minimum(
            torch.atanh(safe_argument) / (2.0 * torch.pi),
            torch.zeros_like(ta))
    )
    range_km = distance / 1000.0
    range_reward = (
        (range_km < 5.0).float()
        + (range_km >= 5.0).float() * torch.clamp(
            -0.032 * range_km.square() + 0.284 * range_km + 0.38,
            0.0, 1.0)
        + torch.clamp(torch.exp(-0.16 * range_km), 0.0, 0.2)
    )
    return 0.1 * orientation * range_reward


def missile_geometry(state):
    rotation = quaternion_to_matrix(state[..., 6:10])
    velocity_ned = torch.matmul(
        rotation, state[..., 3:6].unsqueeze(-1)).squeeze(-1)
    position_neu = torch.stack((
        state[..., 0], state[..., 1], -state[..., 2]), dim=-1)
    velocity_neu = torch.stack((
        velocity_ned[..., 0], velocity_ned[..., 1],
        -velocity_ned[..., 2]), dim=-1)
    launch_position = position_neu.repeat_interleave(2, dim=1)
    launch_velocity = velocity_neu.repeat_interleave(2, dim=1)
    target_position = position_neu.flip(1).repeat_interleave(2, dim=1)
    target_velocity = velocity_neu.flip(1).repeat_interleave(2, dim=1)
    return launch_position, launch_velocity, target_position, target_velocity


def lowlevel_controls(policy, rnn_states, obs, action, *,
                      torch_speed_adapter=False,
                      neutral_throttle_bias=-0.265,
                      speed_command_gain=4.0,
                      high_speed_throttle_bias=-0.18):
    altitude_commands = torch.tensor(
        [0.1, 0.0, -0.1], device=obs.device)
    heading_commands = torch.linspace(
        -torch.pi / 6.0, torch.pi / 6.0, 5, device=obs.device)
    velocity_commands = torch.tensor(
        [0.05, 0.0, -0.05], device=obs.device)
    flat_obs = obs.reshape(-1, obs.shape[-1])
    flat_action = action.reshape(-1, action.shape[-1])
    inputs = torch.zeros(flat_obs.shape[0], 12, device=obs.device)
    inputs[:, 0] = altitude_commands[flat_action[:, 0]]
    inputs[:, 1] = heading_commands[flat_action[:, 1]]
    inputs[:, 2] = velocity_commands[flat_action[:, 2]]
    inputs[:, 3:12] = flat_obs[:, :9]
    with torch.inference_mode():
        low_action, next_states = policy(inputs, rnn_states)
    low_action = low_action.float()
    controls = torch.stack((
        low_action[:, 0] / 20.0 - 1.0,
        low_action[:, 1] / 20.0 - 1.0,
        low_action[:, 2] / 20.0 - 1.0,
        low_action[:, 3] / 58.0 + 0.4,
    ), dim=-1)
    if torch_speed_adapter:
        # The frozen LAG controller was fitted to JSBSim's engine/drag
        # response.  The tensorized model uses the same public F100 and aero
        # tables, but its omitted subsystems shift the longitudinal trim.
        # Preserve the learned surface commands and apply only a documented
        # affine calibration to the throttle channel.  The command input is
        # +0.05/0/-0.05 for accelerate/hold/decelerate respectively.
        throttle = (
            controls[:, 3]
            + neutral_throttle_bias
            + speed_command_gain * inputs[:, 2]
            + high_speed_throttle_bias * torch.clamp(
                (flat_obs[:, 8] * 340.0 - 250.0) / 50.0, 0.0, 1.0)
        ).clamp(0.0, 1.0)
        controls = torch.cat((controls[:, :3], throttle[:, None]), dim=-1)
    return controls, next_states


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the Torch 6-DoF PPO benchmark requires CUDA")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    dynamics = TorchF16BatchDynamics().to(device).eval()
    dynamics_runner = dynamics
    if args.compile:
        dynamics_runner = CompiledTorchF16Runner(dynamics)
    missile = TorchWVRMissileBatch().to(device).eval()
    missile_runner = missile
    if args.compile and args.include_missiles:
        missile_runner = torch.compile(
            missile, fullgraph=True, dynamic=False,
            options={"triton.cudagraphs": False})
    lowlevel = BaselineActor().to(device).eval()
    lowlevel.load_state_dict(torch.load(
        get_root_dir() + "/model/baseline_model.pt",
        map_location=device, weights_only=True))
    lowlevel.requires_grad_(False)
    lowlevel.tpdv = {"dtype": torch.float32, "device": device}
    model = MultiDiscreteActorCritic(
        obs_dim=24, action_dims=[3, 5, 3, 2],
        hidden_size=args.hidden_size, fire_mask_obs_index=23).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3.0e-4, eps=1.0e-5)
    update_args = SimpleNamespace(
        epochs=args.epochs, minibatch_size=args.minibatch_size,
        clip_ratio=0.2, value_coef=0.5, entropy_coef=1.0e-3,
        max_grad_norm=2.0)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    state = reset_states(dynamics, args.num_envs, device, generator)
    obs = observations(state)
    rnn_states = torch.zeros(
        args.num_envs * 2, 1, 128, device=device)
    # Compile and initialize CUDA kernels outside the timed region.
    with torch.inference_mode():
        warm_action, _, _ = model.act(obs.reshape(-1, 24))
        warm_control, _ = lowlevel_controls(
            lowlevel, rnn_states, obs,
            warm_action.reshape(args.num_envs, 2, 4))
        dynamics_runner(
            state.reshape(-1, dynamics.state_size), warm_control, substeps=12)
        if args.include_missiles:
            launch_p, launch_v, target_p, target_v = missile_geometry(state)
            warm_missile = missile.launch(launch_p, launch_v)
            missile_runner(
                warm_missile.reshape(-1, missile.state_size),
                target_p.reshape(-1, 3), target_v.reshape(-1, 3))
    torch.cuda.synchronize()

    rollout_times, update_times, update_history = [], [], []
    total_environment_decisions = 0
    start_all = time.perf_counter()
    for update_index in range(args.updates):
        state = reset_states(dynamics, args.num_envs, device, generator)
        obs = observations(state)
        previous_potential = posture_potential(state)
        if args.include_missiles:
            launch_p, launch_v, _, _ = missile_geometry(state)
            missile_state = missile.launch(
                launch_p, launch_v).reshape(-1, missile.state_size)
        rnn_states = torch.zeros(
            args.num_envs * 2, 1, 128, device=device)
        obs_buffer, action_buffer = [], []
        logprob_buffer, value_buffer, reward_buffer = [], [], []
        potential_buffer = []
        rollout_start = time.perf_counter()
        for _ in range(args.rollout_steps):
            flat_obs = obs.reshape(-1, 24)
            with torch.inference_mode():
                action, log_prob, value = model.act(flat_obs)
                controls, rnn_states = lowlevel_controls(
                    lowlevel, rnn_states, obs,
                    action.reshape(args.num_envs, 2, 4))
                next_state = dynamics_runner(
                    state.reshape(-1, dynamics.state_size), controls,
                    substeps=12).reshape(
                        args.num_envs, 2, dynamics.state_size)
                if args.include_missiles:
                    _, _, target_p, target_v = missile_geometry(next_state)
                    for _ in range(12):
                        missile_state = missile_runner(
                            missile_state, target_p.reshape(-1, 3),
                            target_v.reshape(-1, 3))
                next_obs = observations(next_state)
                potential = posture_potential(next_state)
                raw_reward = 0.99 * potential - previous_potential
                competitive = 0.5 * (
                    raw_reward[:, 0] - raw_reward[:, 1])
                reward = torch.stack((competitive, -competitive), dim=-1)
            obs_buffer.append(flat_obs)
            action_buffer.append(action)
            logprob_buffer.append(log_prob)
            value_buffer.append(value)
            reward_buffer.append(reward.reshape(-1))
            potential_buffer.append(potential)
            state, obs = next_state, next_obs
            previous_potential = potential
        torch.cuda.synchronize()
        rollout_times.append(time.perf_counter() - rollout_start)

        with torch.inference_mode():
            next_value = model.value(obs.reshape(-1, 24))
        rewards = torch.stack(reward_buffer)
        values = torch.stack(value_buffer)
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros_like(next_value)
        for step in reversed(range(args.rollout_steps)):
            following = next_value if step == args.rollout_steps - 1 else values[step + 1]
            delta = rewards[step] + 0.99 * following - values[step]
            gae = delta + 0.99 * 0.95 * gae
            advantages[step] = gae
        returns = advantages + values
        flat_advantages = advantages.reshape(-1)
        flat_advantages = (
            flat_advantages - flat_advantages.mean()
        ) / (flat_advantages.std() + 1.0e-8)
        rollout = Rollout(
            observations=torch.stack(obs_buffer).reshape(-1, 24),
            actions=torch.stack(action_buffer).reshape(-1, 4),
            old_log_probs=torch.stack(logprob_buffer).reshape(-1),
            old_values=values.reshape(-1),
            returns=returns.reshape(-1),
            advantages=flat_advantages,
            mean_reward=float(rewards.mean()),
            mean_abs_reward=float(rewards.abs().mean()),
            zero_sum_error=float(
                torch.max(torch.abs(rewards.reshape(
                    args.rollout_steps, args.num_envs, 2).sum(-1)))),
            completed_episodes=0,
            mean_abs_completed_return=float("nan"),
            accepted_launches=0,
        )
        update_start = time.perf_counter()
        ppo_metrics = ppo_update(
            model, optimizer, rollout, update_args, scaler, True)
        torch.cuda.synchronize()
        update_times.append(time.perf_counter() - update_start)
        total_environment_decisions += args.num_envs * args.rollout_steps
        paired_rewards = rewards.reshape(
            args.rollout_steps, args.num_envs, 2)
        update_history.append({
            "update": update_index + 1,
            "environment_decisions": total_environment_decisions,
            "mean_red_reward": float(paired_rewards[..., 0].mean()),
            "mean_blue_reward": float(paired_rewards[..., 1].mean()),
            "mean_abs_reward": float(paired_rewards.abs().mean()),
            "mean_posture_potential": float(
                torch.stack(potential_buffer).mean()),
            "rollout_s": rollout_times[-1],
            "ppo_update_s": update_times[-1],
            **ppo_metrics,
        })
    elapsed = time.perf_counter() - start_all
    report = {
        "backend": "experimental_torch6dof_v1",
        "scope": ("PPO speed ceiling with four active missile slots"
                  if args.include_missiles else
                  "PPO speed ceiling without missile lifecycle"),
        "scientific_result": False,
        "compiled": bool(args.compile),
        "num_envs": args.num_envs,
        "sim_freq_hz": 60,
        "physics_substeps_per_decision": 12,
        "missile_slots_per_env": 4 if args.include_missiles else 0,
        "updates": args.updates,
        "rollout_steps": args.rollout_steps,
        "total_environment_decisions": total_environment_decisions,
        "elapsed_s": elapsed,
        "environment_decisions_per_s": total_environment_decisions / elapsed,
        "rollout_environment_decisions_per_s": (
            total_environment_decisions / sum(rollout_times)),
        "rollout_times_s": rollout_times,
        "ppo_update_times_s": update_times,
        "update_history": update_history,
        "zero_sum_error": rollout.zero_sum_error,
        "lowlevel_frozen": not any(
            parameter.requires_grad for parameter in lowlevel.parameters()),
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": model.state_dict(),
            "obs_dim": 24,
            "action_dims": [3, 5, 3, 2],
            "hidden_size": args.hidden_size,
            "seed": args.seed,
            "training_environment_decisions": total_environment_decisions,
            "backend": "experimental_torch6dof_xml_fcs_v3",
            "missile_slots_per_env": 4 if args.include_missiles else 0,
        }, args.checkpoint)


if __name__ == "__main__":
    main()
