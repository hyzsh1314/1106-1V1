#!/usr/bin/env python3
"""Collect paired V7.2 state representations with shared all-action MC truth."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from benchmarks.benchmark_torch6dof_ppo import lowlevel_controls
from benchmarks.collect_drvoi_v7_1_p1q_mc_data import load_models
from benchmarks.drvoi import multi_discrete_to_one_hot
from benchmarks.drvoi.targets import response_state_v7
from benchmarks.train_torch_wvr_ppo import fixed_mixture_actions
from benchmarks.validate_drvoi_v7_1_p1q_mc_probe import (
    action_ids_to_components,
    repeat_probe_states,
)
from envs.torch6dof import TorchWVRConfig, TorchWVRVecEnv


REPRESENTATION_KEYS = (
    "registered_v7_1", "full_observation", "future_belief",
    "mechanistic_oracle",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b0-checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "formal_test"),
                        required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, default=Path(
        "configs/drvoi_rhppo_v7_2_state_sufficiency.json"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_representations(base_belief, base_action, decision_belief,
                          decision_observation, lowlevel_hidden):
    representations = {
        "registered_v7_1": torch.cat((
            base_belief, base_action,
            response_state_v7(decision_observation, "rel")), -1),
        "full_observation": torch.cat((
            base_belief, base_action, decision_observation), -1),
        "future_belief": torch.cat((
            decision_belief, decision_observation), -1),
        "mechanistic_oracle": torch.cat((
            decision_belief, decision_observation, lowlevel_hidden), -1),
    }
    expected = {
        "registered_v7_1": 281,
        "full_observation": 308,
        "future_belief": 295,
        "mechanistic_oracle": 423,
    }
    for key, value in representations.items():
        if value.shape[-1] != expected[key]:
            raise RuntimeError(
                f"{key} dimension {value.shape[-1]} != {expected[key]}")
    return representations


@torch.inference_mode()
def collect_horizon(policy, lowlevel, contexts, actions, horizon, seed, device):
    num_envs = contexts * actions
    env = TorchWVRVecEnv(TorchWVRConfig(
        num_envs=num_envs, reward_mode="competitive", attack_quality_weight=0,
        observation_version="v4", use_dynamic_launch_envelope=True),
        device=device, seed=seed, compile_dynamics=True)
    repeat_probe_states(env, contexts, actions)
    env_index = torch.arange(num_envs, device=device)
    learner_side_context = torch.arange(contexts, device=device).remainder(2)
    learner_side = learner_side_context.repeat_interleave(actions)
    opponent_kind_context = torch.arange(contexts, device=device).remainder(3)
    opponent_kind = opponent_kind_context.repeat_interleave(actions)
    hidden = policy.initial_hidden(num_envs, device=device)
    previous_action = torch.zeros(num_envs, 13, device=device)
    reset = torch.ones(num_envs, dtype=torch.bool, device=device)
    lowlevel_hidden = torch.zeros(num_envs * 2, 1, 128, device=device)
    obs = env.observations()
    base_belief = base_action = None

    for step in range(horizon):
        learner_obs = obs[env_index, learner_side]
        result = policy.act(
            learner_obs, previous_action, hidden, reset=reset,
            deterministic=False)
        learner_action = result["action"][::actions].repeat_interleave(actions, 0)
        if step == horizon - 3:
            base_belief = result["belief"][::actions].clone()
            base_action = multi_discrete_to_one_hot(
                learner_action[::actions]).clone()
        executed = fixed_mixture_actions(obs, opponent_kind)
        executed[env_index, learner_side] = learner_action
        controls, next_lowlevel_hidden = lowlevel_controls(
            lowlevel, lowlevel_hidden, obs, executed,
            torch_speed_adapter=True)
        obs, _, done, _ = env.step(
            controls.reshape(num_envs, 2, 4), executed[..., 3] == 1)
        if bool(done.any()):
            raise RuntimeError(
                f"an episode terminated during {horizon}-decision warm-up")
        previous_action = multi_discrete_to_one_hot(learner_action)
        hidden = result["hidden"]
        reset = done
        lowlevel_hidden = next_lowlevel_hidden

    decision_obs_all = obs[env_index, learner_side]
    decision_belief_all, decision_hidden = policy.belief_step(
        decision_obs_all, previous_action, hidden, reset)
    decision_observation = decision_obs_all[::actions]
    decision_belief = decision_belief_all[::actions]
    learner_lowlevel = lowlevel_hidden.reshape(
        num_envs, 2, 1, 128)[env_index, learner_side, 0]
    future_lowlevel_hidden = learner_lowlevel[::actions]
    representations = build_representations(
        base_belief, base_action, decision_belief,
        decision_observation, future_lowlevel_hidden)

    action_ids = torch.arange(actions, device=device).repeat(contexts)
    candidate = action_ids_to_components(action_ids)
    fire = action_ids_to_components(
        torch.arange(actions, device=device))[:, 3]
    fire_legal = decision_observation[:, 23] >= 0.5
    legal = (fire[None] == 0) | fire_legal[:, None]
    executed = fixed_mixture_actions(obs, opponent_kind)
    executed[env_index, learner_side] = candidate
    controls, lowlevel_hidden = lowlevel_controls(
        lowlevel, lowlevel_hidden, obs, executed,
        torch_speed_adapter=True)
    obs, reward, done, _ = env.step(
        controls.reshape(num_envs, 2, 4), executed[..., 3] == 1)
    returns = reward[env_index, learner_side].clone()
    active = ~done
    previous_action = multi_discrete_to_one_hot(candidate)
    hidden = decision_hidden
    reset = done
    lowlevel_hidden = torch.where(
        done.repeat_interleave(2)[:, None, None],
        torch.zeros_like(lowlevel_hidden), lowlevel_hidden)
    discount = 0.99
    decisions_executed = 1
    for _ in range(1, env.config.max_episode_steps - horizon):
        if not bool(active.any()):
            break
        learner_obs = obs[env_index, learner_side]
        result = policy.act(
            learner_obs, previous_action, hidden, reset=reset,
            deterministic=True)
        learner_action = result["action"]
        executed = fixed_mixture_actions(obs, opponent_kind)
        executed[env_index, learner_side] = learner_action
        controls, next_lowlevel_hidden = lowlevel_controls(
            lowlevel, lowlevel_hidden, obs, executed,
            torch_speed_adapter=True)
        obs, reward, done, _ = env.step(
            controls.reshape(num_envs, 2, 4), executed[..., 3] == 1)
        returns += discount * reward[env_index, learner_side] * active.float()
        active &= ~done
        discount *= 0.99
        encoded = multi_discrete_to_one_hot(learner_action)
        previous_action = torch.where(
            done[:, None], torch.zeros_like(encoded), encoded)
        hidden = result["hidden"]
        reset = done
        lowlevel_hidden = torch.where(
            done.repeat_interleave(2)[:, None, None],
            torch.zeros_like(next_lowlevel_hidden), next_lowlevel_hidden)
        decisions_executed += 1
    unfinished = int(active.sum())
    if unfinished:
        raise RuntimeError(f"{unfinished} MC branches did not terminate")
    result = {key: value.cpu() for key, value in representations.items()}
    result.update({
        "return": returns.reshape(contexts, actions).cpu(),
        "legal": legal.cpu(),
        "warmup_decisions": torch.full((contexts,), horizon),
        "learner_side": learner_side_context.cpu(),
        "opponent_kind": opponent_kind_context.cpu(),
        "decisions_executed": decisions_executed,
    })
    del env
    torch.cuda.empty_cache()
    return result


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    parent = json.loads(Path(
        protocol["parent_freeze"]["source"]).read_text(encoding="utf-8"))
    if parent.get("decision") != protocol["parent_freeze"][
            "required_decision"]:
        raise PermissionError("V7.1 parent no-go freeze is missing")
    cfg = protocol["data"]
    allowed = {
        "train": cfg["train_seeds"],
        "validation": cfg["validation_seeds"],
        "formal_test": cfg["formal_test_seeds"],
    }
    if args.seed not in allowed[args.split]:
        raise ValueError(f"seed {args.seed} is not frozen for split {args.split}")
    if args.split == "formal_test":
        freeze = Path(
            "results/drvoi_rhppo_v7_2/state_sufficiency/development_freeze.json")
        if not freeze.exists() or not json.loads(
                freeze.read_text(encoding="utf-8")).get(
                    "formal_test_generation_authorized", False):
            raise PermissionError("V7.2 formal test is not authorized")
    contexts = cfg[f"{args.split}_contexts_per_horizon"]
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    policy, lowlevel = load_models(args.b0_checkpoint, device)
    batches = []
    for index, horizon in enumerate(cfg["warmup_decisions"]):
        batch = collect_horizon(
            policy, lowlevel, contexts, cfg["actions"], horizon,
            args.seed + 1000 * index, device)
        batches.append(batch)
        print(json.dumps({
            "split": args.split, "seed": args.seed, "horizon": horizon,
            "contexts": contexts,
            "decisions_executed": batch["decisions_executed"],
        }), flush=True)
    tensor_keys = (*REPRESENTATION_KEYS, "return", "legal",
                   "warmup_decisions", "learner_side", "opponent_kind")
    data = {
        key: torch.cat([batch[key] for batch in batches])
        for key in tensor_keys}
    data["collection_seed"] = torch.full(
        (data["return"].shape[0],), args.seed, dtype=torch.long)
    dimensions = {key: int(data[key].shape[-1])
                  for key in REPRESENTATION_KEYS}
    frozen_dimensions = {key: int(value["dimension"])
                         for key, value in protocol["representations"].items()}
    if dimensions != frozen_dimensions:
        raise RuntimeError(
            f"representation dimensions {dimensions} != {frozen_dimensions}")
    report = {
        "suite": "DRVOI-RHPPO-V7.2-state-sufficiency-collection",
        "protocol_version": protocol["protocol_version"],
        "scientific_result": False,
        "split": args.split,
        "collection_seed": args.seed,
        "contexts": int(data["return"].shape[0]),
        "actions_per_context": int(data["return"].shape[1]),
        "legal_action_values": int(data["legal"].sum()),
        "representation_dimensions": dimensions,
        "contains_formal_test": args.split == "formal_test",
        "full_episode_return": True,
        "common_random_numbers": True,
        "config_sha256": sha256(args.config),
        "collector_sha256": sha256(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"report": report, "data": data}, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
