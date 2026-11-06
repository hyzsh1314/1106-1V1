#!/usr/bin/env python3
"""Evaluate a frozen V7 recurrent B0 against one fixed opponent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from benchmarks.benchmark_torch6dof_ppo import lowlevel_controls
from benchmarks.drvoi import multi_discrete_to_one_hot
from benchmarks.drvoi.rhppo import RecurrentMultiDiscreteActorCritic
from benchmarks.evaluate_torch_wvr_fixed_opponents import scripted_actions
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.utils.utils import get_root_dir
from envs.torch6dof import TorchWVRConfig, TorchWVRVecEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--opponent", choices=("hold", "pursuit", "evasive"),
                        required=True)
    parser.add_argument("--episodes", type=int, default=192)
    parser.add_argument("--num-envs", type=int, default=96)
    parser.add_argument("--seed", type=int, default=10085)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if payload.get("algorithm") != "v7_b0_extrinsic_rhppo":
        raise ValueError("checkpoint is not a V7 frozen B0")
    model = RecurrentMultiDiscreteActorCritic(
        obs_dim=int(payload["obs_dim"]),
        hidden_size=int(payload["hidden_size"]),
    ).to(device).eval()
    model.load_state_dict(payload["model"])
    model.requires_grad_(False)
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
    lowlevel = BaselineActor().to(device).eval()
    lowlevel.load_state_dict(torch.load(
        get_root_dir() + "/model/baseline_model.pt",
        map_location=device,
        weights_only=True,
    ))
    lowlevel.requires_grad_(False)
    lowlevel.tpdv = {"dtype": torch.float32, "device": device}

    env_index = torch.arange(args.num_envs, device=device)
    learner_side = env_index.remainder(2)
    obs = env.observations()
    hidden = model.initial_hidden(args.num_envs, device=device)
    previous_action = torch.zeros(args.num_envs, 13, device=device)
    reset = torch.ones(args.num_envs, dtype=torch.bool, device=device)
    lowlevel_hidden = torch.zeros(args.num_envs * 2, 1, 128, device=device)

    episode_launches = torch.zeros(args.num_envs, dtype=torch.long, device=device)
    episode_hits = torch.zeros_like(episode_launches)
    episode_requests = torch.zeros_like(episode_launches)
    episode_rejected = torch.zeros_like(episode_launches)
    episode_legal = torch.zeros_like(episode_launches)
    episode_reward = torch.zeros(args.num_envs, device=device)
    target_red = args.episodes // 2
    target_blue = args.episodes - target_red
    learner_red = learner_blue = 0
    wins = losses = draws = timeouts = crashes = 0
    launches = hits = requests = rejected = legal = 0
    total_reward = 0.0
    lengths: list[int] = []
    decisions = 0

    with torch.inference_mode():
        while learner_red < target_red or learner_blue < target_blue:
            learner_obs = obs[env_index, learner_side]
            result = model.act(
                learner_obs,
                previous_action,
                hidden,
                reset=reset,
                deterministic=True,
            )
            learner_action = result["action"]
            opponent_action = scripted_actions(obs, args.opponent)
            action = opponent_action.clone()
            action[env_index, learner_side] = learner_action
            controls, next_lowlevel_hidden = lowlevel_controls(
                lowlevel,
                lowlevel_hidden,
                obs,
                action,
                torch_speed_adapter=True,
            )
            next_obs, reward, done, info = env.step(
                controls.reshape(args.num_envs, 2, 4),
                action[..., 3] == 1,
            )
            encoded_action = multi_discrete_to_one_hot(learner_action)
            previous_action = torch.where(
                done[:, None], torch.zeros_like(encoded_action), encoded_action)
            hidden = result["hidden"]
            reset = done
            obs = next_obs
            lowlevel_hidden = torch.where(
                done.repeat_interleave(2)[:, None, None],
                torch.zeros_like(next_lowlevel_hidden),
                next_lowlevel_hidden,
            )
            decisions += args.num_envs
            episode_reward += reward[env_index, learner_side]
            episode_launches += info["accepted_launches"][env_index, learner_side].long()
            episode_hits += info["hits"][env_index, learner_side].long()
            episode_requests += info["fire_requests"][env_index, learner_side].long()
            episode_rejected += info["rejected_fire_requests"][env_index, learner_side].long()
            episode_legal += info["legal_fire_opportunities"][env_index, learner_side].long()
            if not bool(done.any()):
                continue
            for index in torch.nonzero(done, as_tuple=False).flatten().cpu().tolist():
                side = int(learner_side[index].item())
                if ((side == 0 and learner_red >= target_red)
                        or (side == 1 and learner_blue >= target_blue)):
                    continue
                if side == 0:
                    learner_red += 1
                else:
                    learner_blue += 1
                outcome_red = float(info["outcome_red"][index])
                outcome = outcome_red if side == 0 else -outcome_red
                wins += int(outcome > 0)
                losses += int(outcome < 0)
                draws += int(outcome == 0)
                timeouts += int(info["timeout"][index])
                crashes += int(info["crashes"][index, side])
                launches += int(episode_launches[index])
                hits += int(episode_hits[index])
                requests += int(episode_requests[index])
                rejected += int(episode_rejected[index])
                legal += int(episode_legal[index])
                total_reward += float(episode_reward[index])
                lengths.append(int(info["episode_length"][index]))
            for tensor in (
                episode_launches, episode_hits, episode_requests,
                episode_rejected, episode_legal, episode_reward,
            ):
                tensor.masked_fill_(done, 0)

    completed = learner_red + learner_blue
    report = {
        "suite": "DRVOI-RHPPO-V7-B0-fixed-opponent",
        "protocol_version": "DRVOI-RHPPO-V7-WVR-1",
        "scientific_result": False,
        "checkpoint": str(args.checkpoint),
        "checkpoint_seed": int(payload["seed"]),
        "opponent": args.opponent,
        "evaluation_seed": args.seed,
        "episodes": completed,
        "learner_red_episodes": learner_red,
        "learner_blue_episodes": learner_blue,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / completed,
        "loss_rate": losses / completed,
        "draw_rate": draws / completed,
        "timeout_rate": timeouts / completed,
        "crash_rate": crashes / completed,
        "mean_episode_reward": total_reward / completed,
        "accepted_launches": launches,
        "hits": hits,
        "hit_per_launch": hits / max(launches, 1),
        "legal_fire_opportunities": legal,
        "fire_requests": requests,
        "rejected_fire_requests": rejected,
        "accepted_per_request": launches / max(requests, 1),
        "legal_request_rate": (requests - rejected) / max(requests, 1),
        "mean_episode_length": sum(lengths) / len(lengths),
        "environment_decisions_executed": decisions,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
