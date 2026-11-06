#!/usr/bin/env python3
"""Collect isolated same-state all-action MC data for V7.1 Q calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from benchmarks.benchmark_torch6dof_ppo import lowlevel_controls
from benchmarks.drvoi import multi_discrete_to_one_hot
from benchmarks.drvoi.rhppo import RecurrentMultiDiscreteActorCritic
from benchmarks.drvoi.targets import response_state_v7
from benchmarks.train_torch_wvr_ppo import fixed_mixture_actions
from benchmarks.validate_drvoi_v7_1_p1q_mc_probe import (
    action_ids_to_components,
    repeat_probe_states,
)
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.utils.utils import get_root_dir
from envs.torch6dof import TorchWVRConfig, TorchWVRVecEnv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b0-checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "formal_test"),
                        required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, default=Path(
        "configs/drvoi_rhppo_v7_1_p1q.json"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_models(checkpoint, device):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("algorithm") != "v7_b0_extrinsic_rhppo" or int(
            payload.get("seed", -1)) != 85:
        raise ValueError("MC calibration requires frozen V7 B0 seed85")
    policy = RecurrentMultiDiscreteActorCritic(
        obs_dim=payload["obs_dim"], hidden_size=payload["hidden_size"]
    ).to(device).eval()
    policy.load_state_dict(payload["model"])
    policy.requires_grad_(False)
    lowlevel = BaselineActor().to(device).eval()
    lowlevel.load_state_dict(torch.load(
        get_root_dir() + "/model/baseline_model.pt", map_location=device,
        weights_only=True))
    lowlevel.requires_grad_(False)
    lowlevel.tpdv = {"dtype": torch.float32, "device": device}
    return policy, lowlevel


@torch.inference_mode()
def collect_horizon(policy, lowlevel, contexts, actions, horizon, seed, device):
    num_envs = contexts * actions
    env = TorchWVRVecEnv(TorchWVRConfig(
        num_envs=num_envs, reward_mode="competitive", attack_quality_weight=0,
        observation_version="v4", use_dynamic_launch_envelope=True),
        device=device, seed=seed, compile_dynamics=True)
    repeat_probe_states(env, contexts, actions)
    env_index = torch.arange(num_envs, device=device)
    learner_side_context = torch.arange(
        contexts, device=device).remainder(2)
    learner_side = learner_side_context.repeat_interleave(actions)
    opponent_kind_context = torch.arange(
        contexts, device=device).remainder(3)
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
        # One stochastic warm-up action per context, repeated across its 90
        # branches, keeps every future action comparison on the same state.
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
    if base_belief is None:
        raise RuntimeError("warm-up horizon did not produce a delta=3 base")

    decision_obs = obs[env_index, learner_side]
    _, decision_hidden = policy.belief_step(
        decision_obs, previous_action, hidden, reset)
    response = response_state_v7(decision_obs[::actions], "rel")
    raw_context = torch.cat((base_belief, base_action, response), -1)
    action_ids = torch.arange(actions, device=device).repeat(contexts)
    candidate = action_ids_to_components(action_ids)
    fire = action_ids_to_components(
        torch.arange(actions, device=device))[:, 3]
    fire_legal = decision_obs[::actions, 23] >= 0.5
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
    rollout_limit = env.config.max_episode_steps - horizon
    for _ in range(1, rollout_limit):
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
    result = {
        "context": raw_context.cpu(),
        "return": returns.reshape(contexts, actions).cpu(),
        "legal": legal.cpu(),
        "warmup_decisions": torch.full((contexts,), horizon),
        "learner_side": learner_side_context.cpu(),
        "opponent_kind": opponent_kind_context.cpu(),
        "decisions_executed": decisions_executed,
    }
    del env
    torch.cuda.empty_cache()
    return result


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    cfg = protocol["mc_calibration"]
    allowed = {
        "train": cfg["train_seeds"],
        "validation": cfg["validation_seeds"],
        "formal_test": cfg["formal_test_seeds"],
    }
    if args.seed not in allowed[args.split]:
        raise ValueError(f"seed {args.seed} is not frozen for split {args.split}")
    if args.split == "formal_test":
        freeze = Path("results/drvoi_rhppo_v7_1/p1q/development_freeze.json")
        if not freeze.exists() or not json.loads(
                freeze.read_text(encoding="utf-8")).get(
                    "formal_test_generation_authorized", False):
            raise PermissionError("formal Q test is not authorized by a freeze")
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
    data = {
        key: torch.cat([batch[key] for batch in batches])
        for key in ("context", "return", "legal", "warmup_decisions",
                    "learner_side", "opponent_kind")}
    data["collection_seed"] = torch.full(
        (data["context"].shape[0],), args.seed, dtype=torch.long)
    report = {
        "suite": "DRVOI-RHPPO-V7.1-P1Q-MC-calibration-collection",
        "protocol_version": protocol["protocol_version"],
        "scientific_result": False,
        "split": args.split,
        "collection_seed": args.seed,
        "contexts": int(data["context"].shape[0]),
        "actions_per_context": int(data["return"].shape[1]),
        "legal_action_values": int(data["legal"].sum()),
        "warmup_decisions": cfg["warmup_decisions"],
        "continuation_policy": cfg["continuation_policy"],
        "common_random_numbers": cfg["common_random_numbers"],
        "contains_formal_test": args.split == "formal_test",
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
