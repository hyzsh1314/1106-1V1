#!/usr/bin/env python3
"""Collect complete-episode D-cover/D-onpolicy/D-reactive data for V7."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

from benchmarks.benchmark_torch6dof_ppo import lowlevel_controls
from benchmarks.drvoi import multi_discrete_to_one_hot
from benchmarks.drvoi.rhppo import RecurrentMultiDiscreteActorCritic
from benchmarks.train_torch_wvr_ppo import fixed_mixture_actions
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.utils.utils import get_root_dir
from envs.torch6dof import TorchWVRConfig, TorchWVRVecEnv


OPPONENT_NAMES = ("hold", "pursuit", "evasive")
SPLIT_COUNTS = {"train": 120, "validation": 30, "calibration": 30, "test": 30}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b0-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", choices=("cover", "onpolicy", "reactive"),
                        required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-envs", type=int, default=96)
    parser.add_argument("--train-episodes", type=int, default=120)
    parser.add_argument("--validation-episodes", type=int, default=30)
    parser.add_argument("--calibration-episodes", type=int, default=30)
    parser.add_argument("--test-episodes", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def uniform_legal_actions(observation: torch.Tensor,
                          generator: torch.Generator) -> torch.Tensor:
    """Uniform factorized 90-action coverage with the legal fire mask."""
    batch = observation.shape[0]
    action = torch.stack((
        torch.randint(3, (batch,), device=observation.device, generator=generator),
        torch.randint(5, (batch,), device=observation.device, generator=generator),
        torch.randint(3, (batch,), device=observation.device, generator=generator),
        torch.randint(2, (batch,), device=observation.device, generator=generator),
    ), dim=-1)
    action[:, 3] &= (observation[:, 23] >= 0.5).long()
    return action


def reactive_opponent_action(observation: torch.Tensor,
                             delayed_learner_action: torch.Tensor) -> torch.Tensor:
    """Known two-decision delayed response used only as a positive control."""
    action = torch.zeros(
        observation.shape[0], 4, dtype=torch.long, device=observation.device)
    action[:, 0] = 2 - delayed_learner_action[:, 0]
    action[:, 1] = 4 - delayed_learner_action[:, 1]
    action[:, 2] = 2 - delayed_learner_action[:, 2]
    action[:, 3] = (observation[:, 23] >= 0.5).long()
    return action


def build_schedule(dataset: str, seed: int,
                   split_counts: dict[str, int] | None = None
                   ) -> dict[tuple[int, int], list[dict[str, object]]]:
    """Pre-assign every complete episode to exactly one frozen split."""
    generator = torch.Generator().manual_seed(seed)
    counts = SPLIT_COUNTS if split_counts is None else split_counts
    if any(value < 2 or value % 2 for value in counts.values()):
        raise ValueError(
            "every split needs a positive even episode count for colour balance")
    schedules: dict[tuple[int, int], list[dict[str, object]]] = {}
    for opponent in range(3):
        for side in range(2):
            entries: list[dict[str, object]] = []
            for split, total_count in counts.items():
                count = total_count // 2
                if dataset == "cover":
                    # Complement odd per-colour counts so the two colours
                    # jointly retain exact 50/50 uniform/B0 coverage.
                    uniform_count = count // 2 + (side if count % 2 else 0)
                    modes = (["uniform"] * uniform_count
                             + ["b0"] * (count - uniform_count))
                else:
                    modes = ["b0"] * count
                order = torch.randperm(count, generator=generator).tolist()
                entries.extend({"split": split, "mode": modes[index]}
                               for index in order)
            order = torch.randperm(len(entries), generator=generator).tolist()
            schedules[(opponent, side)] = [entries[index] for index in order]
    return schedules


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.num_envs % 6 != 0:
        raise ValueError("num-envs must be divisible by 6 for balanced roles")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed + 991)
    payload = torch.load(args.b0_checkpoint, map_location=device, weights_only=False)
    if payload.get("algorithm") != "v7_b0_extrinsic_rhppo":
        raise ValueError("response collection requires a frozen V7 B0")
    policy = RecurrentMultiDiscreteActorCritic(
        obs_dim=int(payload["obs_dim"]),
        hidden_size=int(payload["hidden_size"]),
    ).to(device).eval()
    policy.load_state_dict(payload["model"])
    policy.requires_grad_(False)
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
    opponent_side = 1 - learner_side
    opponent_kind = (env_index // 2).remainder(3)
    requested_counts = {
        "train": args.train_episodes,
        "validation": args.validation_episodes,
        "calibration": args.calibration_episodes,
        "test": args.test_episodes,
    }
    schedules = build_schedule(args.dataset, args.seed, requested_counts)
    next_slot = {(opponent, side): 0 for opponent in range(3) for side in range(2)}
    assignment: list[dict[str, object] | None] = [None] * args.num_envs

    def assign_lane(lane: int) -> None:
        kind = int(opponent_kind[lane].item())
        side = int(learner_side[lane].item())
        key = (kind, side)
        slot = next_slot[key]
        if slot >= len(schedules[key]):
            assignment[lane] = None
            return
        value = dict(schedules[key][slot])
        value["opponent"] = kind
        value["learner_side"] = side
        value["slot"] = slot
        value["episode_id"] = (
            args.seed * 10_000 + kind * 1_000 + side * 500 + slot)
        assignment[lane] = value
        next_slot[key] += 1

    for lane in range(args.num_envs):
        assign_lane(lane)

    obs = env.observations()
    hidden = policy.initial_hidden(args.num_envs, device=device)
    previous_action = torch.zeros(args.num_envs, 13, device=device)
    reset = torch.ones(args.num_envs, dtype=torch.bool, device=device)
    lowlevel_hidden = torch.zeros(args.num_envs * 2, 1, 128, device=device)
    neutral = torch.tensor([1, 2, 1, 0], device=device).expand(args.num_envs, -1)
    delayed_queue = neutral[:, None, :].expand(-1, 2, -1).clone()
    lane_start = [0] * args.num_envs
    history = {key: [] for key in ("observation", "belief", "action", "reward", "done")}
    episodes: list[dict[str, object]] = []
    completed_by_opponent = [0, 0, 0]
    step = 0

    with torch.inference_mode():
        while any(
                completed_by_opponent[kind]
                < len(schedules[(kind, 0)]) + len(schedules[(kind, 1)])
                for kind in range(3)):
            learner_obs = obs[env_index, learner_side]
            result = policy.act(
                learner_obs, previous_action, hidden, reset=reset)
            learner_action = result["action"]
            if args.dataset == "cover":
                uniform = uniform_legal_actions(learner_obs, generator)
                use_uniform = torch.tensor([
                    value is not None and value["mode"] == "uniform"
                    for value in assignment
                ], dtype=torch.bool, device=device)
                learner_action = torch.where(
                    use_uniform[:, None], uniform, learner_action)
            executed = fixed_mixture_actions(obs, opponent_kind)
            executed[env_index, learner_side] = learner_action
            if args.dataset == "reactive":
                delayed = delayed_queue[:, 0]
                reaction = reactive_opponent_action(
                    obs[env_index, opponent_side], delayed)
                executed[env_index, opponent_side] = reaction
            controls, next_lowlevel_hidden = lowlevel_controls(
                lowlevel, lowlevel_hidden, obs, executed,
                torch_speed_adapter=True)
            next_obs, reward, done, _ = env.step(
                controls.reshape(args.num_envs, 2, 4),
                executed[..., 3] == 1,
            )
            history["observation"].append(learner_obs.clone())
            history["belief"].append(result["belief"].clone())
            history["action"].append(learner_action.clone())
            history["reward"].append(reward[env_index, learner_side].clone())
            history["done"].append(done.clone())
            encoded = multi_discrete_to_one_hot(learner_action)
            previous_action = torch.where(
                done[:, None], torch.zeros_like(encoded), encoded)
            hidden = result["hidden"]
            reset = done
            obs = next_obs
            lowlevel_hidden = torch.where(
                done.repeat_interleave(2)[:, None, None],
                torch.zeros_like(next_lowlevel_hidden),
                next_lowlevel_hidden,
            )
            delayed_queue = torch.cat((delayed_queue[:, 1:], learner_action[:, None]), 1)
            delayed_queue = torch.where(
                done[:, None, None],
                neutral[:, None, :].expand(-1, 2, -1),
                delayed_queue,
            )

            for lane in torch.nonzero(done, as_tuple=False).flatten().cpu().tolist():
                metadata = assignment[lane]
                if metadata is not None:
                    start = lane_start[lane]
                    episode = {key: torch.stack(values[start:step + 1])[:, lane].cpu()
                               for key, values in history.items()}
                    episode.update(metadata)
                    episode["opponent_name"] = OPPONENT_NAMES[int(metadata["opponent"])]
                    episode["collection_seed"] = args.seed
                    episode["dataset"] = args.dataset
                    episodes.append(episode)
                    completed_by_opponent[int(metadata["opponent"])] += 1
                lane_start[lane] = step + 1
                assign_lane(lane)
            step += 1

    counts = Counter(
        (str(episode["split"]), str(episode["opponent_name"]),
         str(episode["learner_side"]), str(episode["mode"]))
        for episode in episodes)
    report = {
        "suite": "DRVOI-RHPPO-V7-response-collection",
        "protocol_version": "DRVOI-RHPPO-V7-WVR-1",
        "collector_sha256": file_sha256(Path(__file__)),
        "protocol_config_sha256": file_sha256(
            Path("configs/drvoi_rhppo_v7_wvr.json")),
        "scientific_result": False,
        "dataset": args.dataset,
        "requested_episodes_per_opponent_split": requested_counts,
        "seed": args.seed,
        "b0_checkpoint": str(args.b0_checkpoint),
        "b0_seed": int(payload["seed"]),
        "episodes": len(episodes),
        "transitions": sum(int(episode["done"].numel()) for episode in episodes),
        "completed_by_opponent": dict(zip(OPPONENT_NAMES, completed_by_opponent)),
        "split_opponent_side_mode_counts": {
            "|".join(key): value for key, value in sorted(counts.items())},
        "complete_episode_invariants": {
            "all_end_done": all(bool(episode["done"][-1]) for episode in episodes),
            "no_early_done": all(not bool(episode["done"][:-1].any()) for episode in episodes),
            "unique_episode_ids": len({int(episode["episode_id"]) for episode in episodes}),
        },
        "reactive_delay_decisions": 2 if args.dataset == "reactive" else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"report": report, "episodes": episodes}, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
