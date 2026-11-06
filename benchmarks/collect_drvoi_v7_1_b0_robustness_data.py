#!/usr/bin/env python3
"""Collect V7.1 train/validation data for B0 seed robustness only."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

from benchmarks.benchmark_torch6dof_ppo import lowlevel_controls
from benchmarks.collect_drvoi_v7_response_data import (
    OPPONENT_NAMES,
    build_schedule,
    reactive_opponent_action,
    uniform_legal_actions,
)
from benchmarks.drvoi import multi_discrete_to_one_hot
from benchmarks.drvoi.rhppo import RecurrentMultiDiscreteActorCritic
from benchmarks.drvoi_v7_1.history import NEUTRAL_ACTION, assemble_episode_history
from benchmarks.train_torch_wvr_ppo import fixed_mixture_actions
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.utils.utils import get_root_dir
from envs.torch6dof import TorchWVRConfig, TorchWVRVecEnv


PROTOCOL = "DRVOI-RHPPO-V7.1-history-response-1"
ALLOWED_SEEDS = (10185, 11185, 12185, 13185, 14185)
DEVELOPMENT_SPLITS = ("train", "validation")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b0-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", choices=("onpolicy",), default="onpolicy")
    parser.add_argument("--expected-b0-seed", type=int, choices=(185, 285), required=True)
    parser.add_argument("--seed", type=int, choices=ALLOWED_SEEDS, required=True)
    parser.add_argument("--num-envs", type=int, default=96)
    parser.add_argument("--train-episodes", type=int, default=120)
    parser.add_argument("--validation-episodes", type=int, default=30)
    parser.add_argument(
        "--authorization", type=Path,
        default=Path("results/drvoi_rhppo_v7_1/w3/confirmation_audit.json"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def add_explicit_histories(episode: dict[str, object]) -> None:
    history = assemble_episode_history(
        episode["observation"], episode["action"], length=3)
    episode["observation_history"] = history.observations
    episode["action_history"] = history.actions
    episode["history_valid"] = history.valid
    episode["wrong_delay_action_history"] = history.wrong_delay_actions


def reactive_diagnostic_controls(delayed_action: torch.Tensor) -> torch.Tensor:
    """Bounded direct controls for the independent two-step positive control.

    The regular high-level opponent still runs through the frozen low-level
    controller, but its recurrent state makes the physical delay longer than
    the declared two decisions.  This diagnostic-only actuation makes three
    known learner heads causally observable at exactly a[t-2] while retaining
    the same Torch 6-DoF dynamics.  It is never used by D-onpolicy.
    """
    controls = torch.zeros(
        delayed_action.shape[0], 4, device=delayed_action.device)
    controls[:, 0] = 0.45 * (delayed_action[:, 0].float() - 1.0)
    controls[:, 1] = 0.22 * (delayed_action[:, 1].float() - 2.0)
    controls[:, 2] = 0.35 * (delayed_action[:, 2].float() - 1.0)
    throttle = torch.tensor(
        [0.82, 0.55, 0.28], device=delayed_action.device)
    controls[:, 3] = throttle[delayed_action[:, 2].long()]
    return controls


def validate_development_payload(report: dict[str, object], episodes) -> None:
    splits = {str(episode["split"]) for episode in episodes}
    if not splits.issubset(DEVELOPMENT_SPLITS):
        raise RuntimeError(f"development collector produced forbidden splits: {splits}")
    if report["b0_seed"] not in (185, 285):
        raise RuntimeError("robustness data must use B0 seed185 or seed285")
    if report["contains_confirmation_test"]:
        raise RuntimeError("confirmation test leakage in development payload")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.num_envs % 6:
        raise ValueError("num-envs must be divisible by 6 for balanced roles")
    if any(value < 2 or value % 2 for value in (
            args.train_episodes, args.validation_episodes)):
        raise ValueError("episode counts must be positive even numbers")

    authorization = json.loads(args.authorization.read_text(encoding="utf-8"))
    if not authorization.get("all_confirmation_gates_pass"):
        raise PermissionError("confirmation gates did not pass")
    if authorization.get("test_access_count") != 1:
        raise PermissionError("confirmation access count is not frozen at one")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    action_generator = torch.Generator(device=device).manual_seed(args.seed + 991)
    payload = torch.load(args.b0_checkpoint, map_location=device, weights_only=False)
    if (payload.get("algorithm") != "v7_b0_extrinsic_rhppo"
            or int(payload["seed"]) != args.expected_b0_seed):
        raise ValueError("B0 checkpoint seed does not match robustness protocol")
    policy = RecurrentMultiDiscreteActorCritic(
        obs_dim=int(payload["obs_dim"]), hidden_size=int(payload["hidden_size"])
    ).to(device).eval()
    policy.load_state_dict(payload["model"])
    policy.requires_grad_(False)

    env = TorchWVRVecEnv(TorchWVRConfig(
        num_envs=args.num_envs,
        reward_mode="competitive",
        attack_quality_weight=0.0,
        observation_version="v4",
        use_dynamic_launch_envelope=True,
    ), device=device, seed=args.seed, compile_dynamics=True)
    lowlevel = BaselineActor().to(device).eval()
    lowlevel.load_state_dict(torch.load(
        get_root_dir() + "/model/baseline_model.pt",
        map_location=device, weights_only=True))
    lowlevel.requires_grad_(False)
    lowlevel.tpdv = {"dtype": torch.float32, "device": device}

    env_index = torch.arange(args.num_envs, device=device)
    learner_side = env_index.remainder(2)
    opponent_side = 1 - learner_side
    opponent_kind = (env_index // 2).remainder(3)
    requested_counts = {
        "train": args.train_episodes,
        "validation": args.validation_episodes,
    }
    schedules = build_schedule(args.dataset, args.seed, requested_counts)
    next_slot = {(opponent, side): 0 for opponent in range(3) for side in range(2)}
    assignment: list[dict[str, object] | None] = [None] * args.num_envs

    def assign_lane(lane: int) -> None:
        kind = int(opponent_kind[lane])
        side = int(learner_side[lane])
        key = (kind, side)
        slot = next_slot[key]
        if slot >= len(schedules[key]):
            assignment[lane] = None
            return
        value = dict(schedules[key][slot])
        value.update({
            "opponent": kind,
            "learner_side": side,
            "slot": slot,
            "episode_id": args.seed * 10_000 + kind * 1_000 + side * 500 + slot,
        })
        assignment[lane] = value
        next_slot[key] += 1

    for lane in range(args.num_envs):
        assign_lane(lane)

    obs = env.observations()
    hidden = policy.initial_hidden(args.num_envs, device=device)
    previous_action = torch.zeros(args.num_envs, 13, device=device)
    reset = torch.ones(args.num_envs, dtype=torch.bool, device=device)
    lowlevel_hidden = torch.zeros(args.num_envs * 2, 1, 128, device=device)
    neutral = torch.tensor(NEUTRAL_ACTION, device=device).expand(args.num_envs, -1)
    delayed_queue = neutral[:, None, :].expand(-1, 2, -1).clone()
    lane_start = [0] * args.num_envs
    traces = {key: [] for key in ("observation", "belief", "action", "reward", "done")}
    episodes: list[dict[str, object]] = []
    completed = [0, 0, 0]
    step = 0

    with torch.inference_mode():
        while any(completed[kind] < (
                len(schedules[(kind, 0)]) + len(schedules[(kind, 1)]))
                for kind in range(3)):
            learner_obs = obs[env_index, learner_side]
            result = policy.act(learner_obs, previous_action, hidden, reset=reset)
            learner_action = result["action"]
            if args.dataset == "reactive":
                # Randomized legal actions break the deterministic
                # observation->B0-action backdoor.  Otherwise H0 can infer
                # nearly every action from its observation history and the
                # independent causal positive control becomes unidentifiable.
                learner_action = uniform_legal_actions(
                    learner_obs, action_generator)
            executed = fixed_mixture_actions(obs, opponent_kind)
            executed[env_index, learner_side] = learner_action
            reactive_delayed = None
            if args.dataset == "reactive":
                # Before queue update, slot zero is exactly learner a[t-2].
                reactive_delayed = delayed_queue[:, 0]
                executed[env_index, opponent_side] = reactive_opponent_action(
                    obs[env_index, opponent_side], reactive_delayed)
            controls, next_lowlevel_hidden = lowlevel_controls(
                lowlevel, lowlevel_hidden, obs, executed,
                torch_speed_adapter=True)
            controls = controls.reshape(args.num_envs, 2, 4)
            if reactive_delayed is not None:
                controls[env_index, opponent_side] = reactive_diagnostic_controls(
                    reactive_delayed)
            next_obs, reward, done, _ = env.step(
                controls, executed[..., 3] == 1)
            traces["observation"].append(learner_obs.clone())
            traces["belief"].append(result["belief"].clone())
            traces["action"].append(learner_action.clone())
            traces["reward"].append(reward[env_index, learner_side].clone())
            traces["done"].append(done.clone())

            encoded = multi_discrete_to_one_hot(learner_action)
            previous_action = torch.where(done[:, None], torch.zeros_like(encoded), encoded)
            hidden = result["hidden"]
            reset = done
            obs = next_obs
            lowlevel_hidden = torch.where(
                done.repeat_interleave(2)[:, None, None],
                torch.zeros_like(next_lowlevel_hidden), next_lowlevel_hidden)
            delayed_queue = torch.cat((delayed_queue[:, 1:], learner_action[:, None]), 1)
            delayed_queue = torch.where(
                done[:, None, None],
                neutral[:, None, :].expand(-1, 2, -1), delayed_queue)

            for lane in torch.nonzero(done, as_tuple=False).flatten().cpu().tolist():
                metadata = assignment[lane]
                if metadata is not None:
                    start = lane_start[lane]
                    episode = {
                        key: torch.stack(values[start:step + 1])[:, lane].cpu()
                        for key, values in traces.items()
                    }
                    episode.update(metadata)
                    episode["opponent_name"] = OPPONENT_NAMES[int(metadata["opponent"])]
                    episode["collection_seed"] = args.seed
                    episode["dataset"] = args.dataset
                    add_explicit_histories(episode)
                    episodes.append(episode)
                    completed[int(metadata["opponent"])] += 1
                lane_start[lane] = step + 1
                assign_lane(lane)
            step += 1

    counts = Counter((str(e["split"]), str(e["opponent_name"]),
                      str(e["learner_side"])) for e in episodes)
    config_path = Path("configs/drvoi_rhppo_v7_1_wvr.json")
    report = {
        "suite": "DRVOI-RHPPO-V7.1-B0-robustness-collection",
        "protocol_version": PROTOCOL,
        "scientific_result": False,
        "dataset": args.dataset,
        "seed": args.seed,
        "collection_seed": args.seed,
        "b0_checkpoint": str(args.b0_checkpoint),
        "b0_seed": int(payload["seed"]),
        "requested_episodes_per_opponent_split": requested_counts,
        "episodes": len(episodes),
        "transitions": sum(int(e["done"].numel()) for e in episodes),
        "completed_by_opponent": dict(zip(OPPONENT_NAMES, completed)),
        "split_opponent_side_counts": {
            "|".join(key): value for key, value in sorted(counts.items())},
        "collector_sha256": sha256(Path(__file__)),
        "protocol_config_sha256": sha256(config_path),
        "history_length": 3,
        "reactive_delay_decisions": 2 if args.dataset == "reactive" else None,
        "reactive_actuation": (
            "diagnostic_direct_control_v1" if args.dataset == "reactive" else None),
        "learner_action_source": (
            "uniform_legal_randomized_intervention"
            if args.dataset == "reactive"
            else f"frozen_b0_seed{int(payload['seed'])}"),
        "available_splits": sorted({str(e["split"]) for e in episodes}),
        "contains_confirmation_test": False,
        "confirmation_test_generated": False,
        "confirmation_test_accessed": False,
        "confirmation_authorization": str(args.authorization),
        "confirmation_authorization_sha256": sha256(args.authorization),
        "complete_episode_invariants": {
            "all_end_done": all(bool(e["done"][-1]) for e in episodes),
            "no_early_done": all(not bool(e["done"][:-1].any()) for e in episodes),
            "unique_episode_ids": len({int(e["episode_id"]) for e in episodes}),
            "all_history_shapes_valid": all(
                e["observation_history"].shape[1:] == (3, 39)
                and e["action_history"].shape[1:] == (3, 13)
                and e["history_valid"].shape[1:] == (3,)
                for e in episodes),
        },
    }
    validate_development_payload(report, episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"report": report, "episodes": episodes}, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
