"""Evaluate a Torch WVR checkpoint against deterministic fixed opponents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from benchmarks.benchmark_torch6dof_ppo import lowlevel_controls
from benchmarks.train_ppo_1v1_shoot_smoke import MultiDiscreteActorCritic
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.utils.utils import get_root_dir
from envs.torch6dof import TorchWVRConfig, TorchWVRVecEnv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--scripted-learner", choices=("pure-pursuit", "los-intercept"),
        help="evaluate action-space reachability without a learned policy")
    parser.add_argument("--opponent", choices=("hold", "pursuit", "evasive"),
                        required=True)
    parser.add_argument(
        "--hierarchical-gate-mode",
        choices=("learned", "always-base", "always-intercept", "random",
                 "geometry", "oracle"),
        default="learned",
        help="diagnostic gate override for hierarchical checkpoints")
    parser.add_argument("--episodes", type=int, default=192)
    parser.add_argument("--num-envs", type=int, default=96)
    parser.add_argument("--seed", type=int, default=10085)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def scripted_actions(obs: torch.Tensor, name: str) -> torch.Tensor:
    action = torch.zeros(*obs.shape[:2], 4, dtype=torch.long,
                         device=obs.device)
    action[..., 0] = 1
    action[..., 2] = 1
    side = obs[..., 14]
    if name == "hold":
        action[..., 1] = 2
    elif name == "pursuit":
        action[..., 1] = torch.where(side >= 0.0, 4, 0)
        action[..., 2] = 0
    else:
        action[..., 1] = torch.where(side >= 0.0, 0, 4)
        action[..., 2] = 0
    action[..., 3] = (obs[..., 23] >= 0.5).long()
    return action


def scripted_learner_actions(obs: torch.Tensor, name: str) -> torch.Tensor:
    """Deterministic reachability controllers using the PPO action space."""
    action = torch.zeros(*obs.shape[:2], 4, dtype=torch.long,
                         device=obs.device)
    # Track target altitude, command maximum speed, and fire exactly when the
    # environment exposes a legal shot.  No privileged state is used.
    elevation = obs[..., 25]
    action[..., 0] = torch.where(
        elevation > 0.04, 0,
        torch.where(elevation < -0.04, 2, 1))
    altitude = obs[..., 0]
    action[..., 0] = torch.where(
        altitude <= 0.6, torch.zeros_like(action[..., 0]),
        torch.where(
            (altitude <= 0.8) & (action[..., 0] == 2),
            torch.ones_like(action[..., 0]), action[..., 0]))
    action[..., 2] = 0
    if name == "pure-pursuit":
        action[..., 1] = torch.where(obs[..., 14] >= 0.0, 4, 0)
    else:
        # Body LOS azimuth plus LOS-rate lead.  Quantization matches the five
        # relative-heading commands available to PPO.
        steering = torch.pi * obs[..., 24] + 3.0 * obs[..., 26]
        action[..., 1] = torch.where(
            steering < -0.20, 0,
            torch.where(
                steering < -0.06, 1,
                torch.where(
                    steering <= 0.06, 2,
                    torch.where(steering <= 0.20, 3, 4))))
    action[..., 3] = (obs[..., 23] >= 0.5).long()
    return action


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.checkpoint is None and args.scripted_learner is None:
        raise ValueError("--checkpoint or --scripted-learner is required")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    payload = None
    model = None
    gate = None
    base_skill = None
    intercept_skill = None
    hierarchical = False
    if args.scripted_learner is None:
        payload = torch.load(args.checkpoint, map_location=device,
                             weights_only=False)
        hierarchical = (
            payload.get("algorithm") == "option_gated_hierarchical_ppo")
        if hierarchical:
            from benchmarks.train_torch_wvr_hierarchical_ppo import (
                apply_fire_controller,
                build_gate_observation,
                build_frozen_skill,
                select_skill_actions,
            )
            gate = MultiDiscreteActorCritic(
                obs_dim=int(payload.get("gate_obs_dim", payload["obs_dim"])),
                action_dims=[2],
                hidden_size=int(payload["hidden_size"])).to(device).eval()
            gate.load_state_dict(payload["gate_model"])
            base_skill = build_frozen_skill(
                payload["base_skill"], device, int(payload["obs_dim"]))
            intercept_skill = build_frozen_skill(
                payload["intercept_skill"], device,
                int(payload["obs_dim"]))
        else:
            model = MultiDiscreteActorCritic(
                obs_dim=int(payload["obs_dim"]),
                action_dims=payload["action_dims"],
                hidden_size=int(payload["hidden_size"]),
                fire_mask_obs_index=23,
                altitude_safety_obs_index=0,
                altitude_descend_mask_threshold=0.8,
                altitude_force_climb_threshold=0.6).to(device).eval()
            model.load_state_dict(payload["model"])
    lowlevel = BaselineActor().to(device).eval()
    lowlevel.load_state_dict(torch.load(
        get_root_dir() + "/model/baseline_model.pt",
        map_location=device, weights_only=True))
    lowlevel.requires_grad_(False)
    lowlevel.tpdv = {"dtype": torch.float32, "device": device}
    obs_dim = 39 if payload is None else int(payload["obs_dim"])
    observation_version = {24: "v2", 29: "v3", 39: "v4"}.get(obs_dim)
    if observation_version is None:
        raise ValueError(f"unsupported checkpoint obs_dim: {obs_dim}")
    env = TorchWVRVecEnv(
        TorchWVRConfig(
            num_envs=args.num_envs,
            observation_version=observation_version,
            use_dynamic_launch_envelope=obs_dim >= 29), device=device,
        seed=args.seed, compile_dynamics=True)
    use_speed_adapter = (
        True if payload is None
        else bool(payload.get("torch_speed_adapter", False)))
    obs = env.observations()
    rnn = torch.zeros(args.num_envs * 2, 1, 128, device=device)
    # Learner appears on both colours in the same batch.
    learner_side = (torch.arange(args.num_envs, device=device) % 2).long()
    env_index = torch.arange(args.num_envs, device=device)
    completed = wins = losses = draws = timeouts = crashes = 0
    learner_red = learner_blue = 0
    launches = hits = decisions = 0
    legal_opportunities = fire_requests = rejected_requests = 0
    episode_lengths = []
    episode_launches = torch.zeros(
        args.num_envs, dtype=torch.long, device=device)
    episode_hits = torch.zeros_like(episode_launches)
    episode_legal = torch.zeros_like(episode_launches)
    episode_requests = torch.zeros_like(episode_launches)
    episode_rejected = torch.zeros_like(episode_launches)
    episode_legal_steps = torch.zeros_like(episode_launches)
    episode_first_legal = torch.full_like(episode_launches, -1)
    episode_first_launch = torch.full_like(episode_launches, -1)
    episode_launch_quality_sum = torch.zeros(
        args.num_envs, device=device)
    episode_launch_quality_count = torch.zeros_like(episode_launches)
    current_option = torch.zeros(
        args.num_envs, dtype=torch.long, device=device)
    option_remaining = torch.zeros_like(current_option)
    option_counts = torch.zeros(2, dtype=torch.long, device=device)
    previous_option = torch.full_like(current_option, -1)
    previous_boundary_obs = torch.zeros(
        args.num_envs, obs_dim, device=device)
    has_gate_history = torch.zeros(
        args.num_envs, dtype=torch.bool, device=device)
    first_legal_steps = []
    first_launch_steps = []
    total_legal_steps = 0
    launch_quality_sum = 0.0
    launch_quality_count = 0
    target_red = args.episodes // 2
    target_blue = args.episodes - target_red
    with torch.inference_mode():
        while learner_red < target_red or learner_blue < target_blue:
            if args.scripted_learner is None:
                if hierarchical:
                    learner_obs = obs[env_index, learner_side]
                    boundary = option_remaining == 0
                    if bool(boundary.any()):
                        indices = torch.nonzero(
                            boundary, as_tuple=False).flatten()
                        gate_obs = build_gate_observation(
                            learner_obs, previous_boundary_obs,
                            previous_option, has_gate_history
                        ) if payload.get("gate_history", False) else learner_obs
                        if args.hierarchical_gate_mode == "learned":
                            sampled, _, _ = gate.act(
                                gate_obs[indices], deterministic=True)
                            selected = sampled[:, 0]
                        elif args.hierarchical_gate_mode == "always-base":
                            selected = torch.zeros_like(indices)
                        elif args.hierarchical_gate_mode == "always-intercept":
                            selected = torch.ones_like(indices)
                        elif args.hierarchical_gate_mode == "random":
                            selected = torch.randint(
                                0, 2, indices.shape, device=device)
                        elif args.hierarchical_gate_mode == "oracle":
                            selected = torch.full_like(
                                indices, int(args.opponent == "evasive"))
                        else:
                            gate_obs = learner_obs[indices]
                            los_motion = (
                                gate_obs[:, 26:28].square().sum(-1).sqrt()
                                + 0.15 * gate_obs[:, 24].abs())
                            selected = (los_motion > 0.055).long()
                        current_option[indices] = selected
                        previous_option[indices] = selected
                        previous_boundary_obs[indices] = learner_obs[indices]
                        has_gate_history[indices] = True
                        option_remaining[indices] = int(
                            payload["option_horizon"])
                        option_counts.scatter_add_(
                            0, selected, torch.ones_like(selected))
                    base_action, _, _ = base_skill.act(
                        learner_obs, deterministic=True)
                    intercept_action, _, _ = intercept_skill.act(
                        learner_obs, deterministic=True)
                    learner_action = select_skill_actions(
                        current_option, base_action, intercept_action)
                    learner_action = apply_fire_controller(
                        learner_action, learner_obs, base_action,
                        payload.get("fire_controller", "skill"),
                        options=current_option,
                        **payload.get("fire_controller_config", {}))
                    policy_action = scripted_actions(obs, "hold")
                    policy_action[env_index, learner_side] = learner_action
                    option_remaining -= 1
                else:
                    policy_action, _, _ = model.act(
                        obs.reshape(-1, env.observation_size),
                        deterministic=True)
                    policy_action = policy_action.reshape(
                        args.num_envs, 2, 4)
            else:
                policy_action = scripted_learner_actions(
                    obs, args.scripted_learner)
            opponent_action = scripted_actions(obs, args.opponent)
            selector = torch.zeros(
                args.num_envs, 2, 1, dtype=torch.bool, device=device)
            selector[env_index, learner_side, 0] = True
            action = torch.where(selector, policy_action, opponent_action)
            controls, next_rnn = lowlevel_controls(
                lowlevel, rnn, obs, action,
                torch_speed_adapter=use_speed_adapter)
            obs, _, done, info = env.step(
                controls.reshape(args.num_envs, 2, 4), action[..., 3] == 1)
            rnn = torch.where(
                done.repeat_interleave(2)[:, None, None],
                torch.zeros_like(next_rnn), next_rnn)
            if hierarchical:
                option_remaining = torch.where(
                    done, torch.zeros_like(option_remaining),
                    option_remaining)
                previous_option = torch.where(
                    done, torch.full_like(previous_option, -1),
                    previous_option)
                previous_boundary_obs = torch.where(
                    done[:, None], torch.zeros_like(previous_boundary_obs),
                    previous_boundary_obs)
                has_gate_history = torch.where(
                    done, torch.zeros_like(has_gate_history),
                    has_gate_history)
            decisions += args.num_envs
            episode_launches += info["accepted_launches"][
                env_index, learner_side].long()
            episode_hits += info["hits"][env_index, learner_side].long()
            episode_legal += info["legal_fire_opportunities"][
                env_index, learner_side].long()
            episode_requests += info["fire_requests"][
                env_index, learner_side].long()
            episode_rejected += info["rejected_fire_requests"][
                env_index, learner_side].long()
            learner_legal = info["legal_fire_opportunities"][
                env_index, learner_side]
            learner_accepted = info["accepted_launches"][
                env_index, learner_side]
            step_index = info["episode_length"].long()
            episode_legal_steps += learner_legal.long()
            episode_first_legal = torch.where(
                (episode_first_legal < 0) & learner_legal,
                step_index, episode_first_legal)
            episode_first_launch = torch.where(
                (episode_first_launch < 0) & learner_accepted,
                step_index, episode_first_launch)
            learner_quality = info["prelaunch_attack_quality"][
                env_index, learner_side]
            episode_launch_quality_sum += (
                learner_quality * learner_accepted.float())
            episode_launch_quality_count += learner_accepted.long()
            if not bool(done.any()):
                continue
            finished = torch.nonzero(done, as_tuple=False).flatten()
            for index in finished.cpu().tolist():
                side = int(learner_side[index].item())
                if ((side == 0 and learner_red >= target_red)
                        or (side == 1 and learner_blue >= target_blue)):
                    continue
                if side == 0:
                    learner_red += 1
                else:
                    learner_blue += 1
                red_outcome = float(info["outcome_red"][index].item())
                learner_outcome = red_outcome if side == 0 else -red_outcome
                wins += int(learner_outcome > 0)
                losses += int(learner_outcome < 0)
                draws += int(learner_outcome == 0)
                timeouts += int(info["timeout"][index].item())
                crashes += int(info["crashes"][index, side].item())
                launches += int(episode_launches[index].item())
                hits += int(episode_hits[index].item())
                legal_opportunities += int(episode_legal[index].item())
                fire_requests += int(episode_requests[index].item())
                rejected_requests += int(episode_rejected[index].item())
                total_legal_steps += int(episode_legal_steps[index].item())
                if int(episode_first_legal[index].item()) >= 0:
                    first_legal_steps.append(
                        int(episode_first_legal[index].item()))
                if int(episode_first_launch[index].item()) >= 0:
                    first_launch_steps.append(
                        int(episode_first_launch[index].item()))
                launch_quality_sum += float(
                    episode_launch_quality_sum[index].item())
                launch_quality_count += int(
                    episode_launch_quality_count[index].item())
                episode_lengths.append(int(info["episode_length"][index].item()))
                completed += 1
            episode_launches = torch.where(
                done, torch.zeros_like(episode_launches), episode_launches)
            episode_hits = torch.where(
                done, torch.zeros_like(episode_hits), episode_hits)
            episode_legal = torch.where(
                done, torch.zeros_like(episode_legal), episode_legal)
            episode_requests = torch.where(
                done, torch.zeros_like(episode_requests), episode_requests)
            episode_rejected = torch.where(
                done, torch.zeros_like(episode_rejected), episode_rejected)
            episode_legal_steps = torch.where(
                done, torch.zeros_like(episode_legal_steps),
                episode_legal_steps)
            episode_first_legal = torch.where(
                done, torch.full_like(episode_first_legal, -1),
                episode_first_legal)
            episode_first_launch = torch.where(
                done, torch.full_like(episode_first_launch, -1),
                episode_first_launch)
            episode_launch_quality_sum = torch.where(
                done, torch.zeros_like(episode_launch_quality_sum),
                episode_launch_quality_sum)
            episode_launch_quality_count = torch.where(
                done, torch.zeros_like(episode_launch_quality_count),
                episode_launch_quality_count)
    report = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "learner_policy": args.scripted_learner or "checkpoint",
        "algorithm": (
            payload.get("algorithm", "standard_ppo")
            if payload is not None else "scripted"),
        "hierarchical_gate_mode": (
            args.hierarchical_gate_mode if hierarchical else None),
        "opponent": args.opponent,
        "seed": args.seed, "episodes": completed,
        "learner_red_episodes": learner_red,
        "learner_blue_episodes": learner_blue,
        "wins": wins, "losses": losses, "draws": draws,
        "win_rate": wins / completed, "loss_rate": losses / completed,
        "draw_rate": draws / completed, "timeout_rate": timeouts / completed,
        "learner_crash_rate": crashes / completed,
        "accepted_launches": launches, "hits": hits,
        "torch_speed_adapter": use_speed_adapter,
        "legal_fire_opportunities": legal_opportunities,
        "fire_requests": fire_requests,
        "rejected_fire_requests": rejected_requests,
        "request_per_legal_opportunity": (
            fire_requests / max(legal_opportunities, 1)),
        "accepted_per_request": launches / max(fire_requests, 1),
        "hit_per_launch": hits / max(launches, 1),
        "legal_opportunity_episode_rate": (
            len(first_legal_steps) / completed),
        "launch_episode_rate": len(first_launch_steps) / completed,
        "mean_first_legal_step": (
            sum(first_legal_steps) / len(first_legal_steps)
            if first_legal_steps else None),
        "mean_first_launch_step": (
            sum(first_launch_steps) / len(first_launch_steps)
            if first_launch_steps else None),
        "legal_opportunity_step_fraction": (
            total_legal_steps / max(sum(episode_lengths), 1)),
        "mean_launch_attack_quality": (
            launch_quality_sum / max(launch_quality_count, 1)),
        "mean_episode_length": sum(episode_lengths) / len(episode_lengths),
        "environment_decisions_executed": decisions,
        "scientific_result": False,
    }
    if hierarchical:
        total_options = int(option_counts.sum().item())
        report["option_horizon"] = int(payload["option_horizon"])
        report["base_option_fraction"] = (
            int(option_counts[0].item()) / max(total_options, 1))
        report["intercept_option_fraction"] = (
            int(option_counts[1].item()) / max(total_options, 1))
    rendered = json.dumps(report, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
