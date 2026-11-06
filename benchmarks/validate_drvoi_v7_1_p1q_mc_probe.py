#!/usr/bin/env python3
"""Same-state 90-action extrinsic MC probe for V7.1 Q development."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.benchmark_torch6dof_ppo import lowlevel_controls
from benchmarks.drvoi import multi_discrete_to_one_hot
from benchmarks.drvoi.core import ACTION_DIMS, action_ids_to_multi_one_hot
from benchmarks.validate_drvoi_v7_1_p1q_development import build_q_model
from benchmarks.drvoi.rhppo import RecurrentMultiDiscreteActorCritic
from benchmarks.drvoi.targets import response_state_v7
from benchmarks.drvoi_v7_1.q_controls import bootstrap_context_mean
from benchmarks.train_torch_wvr_ppo import fixed_mixture_actions
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.utils.utils import get_root_dir
from envs.torch6dof import TorchWVRConfig, TorchWVRVecEnv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-artifact", type=Path, required=True)
    parser.add_argument("--b0-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(
        "configs/drvoi_rhppo_v7_1_p1q.json"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def action_ids_to_components(ids):
    remainder = ids.long()
    components = []
    for index, size in enumerate(ACTION_DIMS):
        stride = int(np.prod(ACTION_DIMS[index + 1:]))
        components.append(torch.div(remainder, stride, rounding_mode="floor"))
        remainder = remainder.remainder(stride)
    return torch.stack(components, -1)


def repeat_probe_states(env, contexts, actions):
    source = torch.arange(contexts, device=env.device) * actions
    target_source = source.repeat_interleave(actions)
    for name in (
            "aircraft_state", "missile_state", "alive", "remaining_missiles",
            "last_launch_step", "episode_step", "episode_id",
            "previous_potential"):
        value = getattr(env, name)
        value.copy_(value[target_source])


def per_probe_metrics(prediction, truth, legal, minimum_gap, seed):
    pair_accuracy, regret, random_regret, details = [], [], [], []
    rng = np.random.default_rng(seed)
    for probe in range(truth.shape[0]):
        selected = torch.nonzero(legal[probe]).flatten()
        true = truth[probe, selected]
        pred = prediction[probe, selected]
        value_range = (true.max() - true.min()).clamp_min(1e-8)
        left, right = torch.triu_indices(selected.numel(), selected.numel(), 1,
                                         device=true.device)
        true_gap = true[left] - true[right]
        pred_gap = pred[left] - pred[right]
        valid = true_gap.abs() >= minimum_gap * value_range
        accuracy = float(((true_gap[valid] * pred_gap[valid]) > 0).float().mean())
        chosen = pred.argmax()
        normalized_regret = float((true.max() - true[chosen]) / value_range)
        random_choice = int(rng.integers(0, selected.numel()))
        random_value = float((true.max() - true[random_choice]) / value_range)
        pair_accuracy.append(accuracy); regret.append(normalized_regret)
        random_regret.append(random_value)
        details.append({
            "probe": probe, "legal_actions": int(selected.numel()),
            "valid_action_pairs": int(valid.sum()),
            "pairwise_accuracy": accuracy,
            "normalized_argmax_regret": normalized_regret,
            "random_normalized_regret": random_value,
            "predicted_action_id": int(selected[chosen]),
            "mc_optimal_action_id": int(selected[true.argmax()]),
        })
    tensor_accuracy = torch.tensor(pair_accuracy)
    mean_accuracy, accuracy_ci = bootstrap_context_mean(
        tensor_accuracy, seed + 1, samples=4000)
    return {
        "pairwise_accuracy": mean_accuracy,
        "pairwise_accuracy_ci95": accuracy_ci,
        "normalized_argmax_regret": float(np.mean(regret)),
        "random_normalized_argmax_regret": float(np.mean(random_regret)),
        "probes": details,
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    development = json.loads(Path(
        "runs/drvoi_rhppo_v7_1/p1q/wvr-development/validation.json"
    ).read_text(encoding="utf-8"))
    if not development.get("development_semantics_pass"):
        raise PermissionError("Q development semantics did not pass")
    cfg = protocol["mc_probe"]
    contexts, actions = cfg["contexts"], cfg["actions"]
    num_envs = contexts * actions
    device = torch.device("cuda")
    torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed_all(cfg["seed"])

    b0_payload = torch.load(args.b0_checkpoint, map_location=device,
                            weights_only=False)
    policy = RecurrentMultiDiscreteActorCritic(
        obs_dim=b0_payload["obs_dim"], hidden_size=b0_payload["hidden_size"]
    ).to(device).eval()
    policy.load_state_dict(b0_payload["model"]); policy.requires_grad_(False)
    lowlevel = BaselineActor().to(device).eval()
    lowlevel.load_state_dict(torch.load(
        get_root_dir() + "/model/baseline_model.pt", map_location=device,
        weights_only=True))
    lowlevel.requires_grad_(False)
    lowlevel.tpdv = {"dtype": torch.float32, "device": device}
    env = TorchWVRVecEnv(TorchWVRConfig(
        num_envs=num_envs, reward_mode="competitive", attack_quality_weight=0,
        observation_version="v4", use_dynamic_launch_envelope=True),
        device=device, seed=cfg["seed"], compile_dynamics=True)
    repeat_probe_states(env, contexts, actions)

    env_index = torch.arange(num_envs, device=device)
    learner_side = torch.zeros(num_envs, dtype=torch.long, device=device)
    opponent_kind = torch.arange(contexts, device=device).remainder(3).repeat_interleave(actions)
    hidden = policy.initial_hidden(num_envs, device=device)
    previous_action = torch.zeros(num_envs, 13, device=device)
    reset = torch.ones(num_envs, dtype=torch.bool, device=device)
    lowlevel_hidden = torch.zeros(num_envs * 2, 1, 128, device=device)
    obs = env.observations()
    base_belief = base_action = None

    with torch.inference_mode():
        for step in range(3):
            learner_obs = obs[:, 0]
            result = policy.act(learner_obs, previous_action, hidden,
                                reset=reset, deterministic=True)
            learner_action = result["action"]
            if step == 0:
                base_belief = result["belief"].clone()
                base_action = multi_discrete_to_one_hot(learner_action).clone()
            executed = fixed_mixture_actions(obs, opponent_kind)
            executed[:, 0] = learner_action
            controls, next_lowlevel_hidden = lowlevel_controls(
                lowlevel, lowlevel_hidden, obs, executed,
                torch_speed_adapter=True)
            obs, _, done, _ = env.step(
                controls.reshape(num_envs, 2, 4), executed[..., 3] == 1)
            if bool(done.any()):
                raise RuntimeError("MC probe terminated before decision state")
            previous_action = multi_discrete_to_one_hot(learner_action)
            hidden = result["hidden"]; reset = done
            lowlevel_hidden = next_lowlevel_hidden

        decision_obs = obs[:, 0]
        _, decision_hidden = policy.belief_step(
            decision_obs, previous_action, hidden, reset)
        response = response_state_v7(decision_obs, "rel")
        raw_context = torch.cat((base_belief, base_action, response), -1)
        probe_context = raw_context[::actions]

        q_payload = torch.load(args.q_artifact, map_location=device,
                               weights_only=False)
        selected_period = q_payload["selected"]
        q_model = build_q_model(
            q_payload["context_dim"], q_payload.get(
                "architecture", "action_embedding_v1")).to(device).eval()
        q_model.load_state_dict(q_payload["models"][selected_period])
        standardized = ((probe_context - q_payload["context_mean"].to(device))
                        / q_payload["context_scale"].to(device))
        prediction = q_model.all_actions(standardized)

        action_ids = torch.arange(actions, device=device).repeat(contexts)
        candidate = action_ids_to_components(action_ids)
        legal = torch.ones(contexts, actions, dtype=torch.bool, device=device)
        fire = action_ids_to_components(torch.arange(actions, device=device))[:, 3]
        fire_legal = decision_obs[::actions, 23] >= 0.5
        legal &= (fire[None] == 0) | fire_legal[:, None]

        executed = fixed_mixture_actions(obs, opponent_kind)
        executed[:, 0] = candidate
        controls, lowlevel_hidden = lowlevel_controls(
            lowlevel, lowlevel_hidden, obs, executed,
            torch_speed_adapter=True)
        obs, reward, done, _ = env.step(
            controls.reshape(num_envs, 2, 4), executed[..., 3] == 1)
        returns = reward[:, 0].clone()
        active = ~done
        previous_action = multi_discrete_to_one_hot(candidate)
        hidden = decision_hidden
        reset = done
        lowlevel_hidden = torch.where(
            done.repeat_interleave(2)[:, None, None],
            torch.zeros_like(lowlevel_hidden), lowlevel_hidden)
        discount = 0.99
        decisions_executed = 1
        for _ in range(1, cfg["rollout_decisions"]):
            if not bool(active.any()):
                break
            learner_obs = obs[:, 0]
            result = policy.act(learner_obs, previous_action, hidden,
                                reset=reset, deterministic=True)
            learner_action = result["action"]
            executed = fixed_mixture_actions(obs, opponent_kind)
            executed[:, 0] = learner_action
            controls, next_lowlevel_hidden = lowlevel_controls(
                lowlevel, lowlevel_hidden, obs, executed,
                torch_speed_adapter=True)
            obs, reward, done, _ = env.step(
                controls.reshape(num_envs, 2, 4), executed[..., 3] == 1)
            returns += discount * reward[:, 0] * active.float()
            active &= ~done
            discount *= 0.99
            previous_action = torch.where(
                done[:, None], torch.zeros_like(previous_action),
                multi_discrete_to_one_hot(learner_action))
            hidden = result["hidden"]; reset = done
            lowlevel_hidden = torch.where(
                done.repeat_interleave(2)[:, None, None],
                torch.zeros_like(next_lowlevel_hidden), next_lowlevel_hidden)
            decisions_executed += 1
        unfinished = int(active.sum())
        if cfg.get("require_all_terminal", False) and unfinished:
            raise RuntimeError(
                f"{unfinished} MC branches did not reach an episode terminal")
        truth = returns.reshape(contexts, actions)

    metrics = per_probe_metrics(
        prediction, truth, legal, cfg["pair_minimum_gap_fraction"], cfg["seed"])
    gate = protocol["wvr_value_gate"]
    gates = {
        "pairwise_action_ranking": {"pass": (
            metrics["pairwise_accuracy"] >= gate["pairwise_accuracy_min"]
            and metrics["pairwise_accuracy_ci95"][0]
            > gate["pairwise_accuracy_ci_lower_min"])},
        "normalized_argmax_regret": {"pass": (
            metrics["normalized_argmax_regret"]
            <= gate["normalized_argmax_regret_max"])},
        "random_action_control": {"pass": (
            metrics["normalized_argmax_regret"]
            < metrics["random_normalized_argmax_regret"])},
    }
    report = {
        "suite": "DRVOI-RHPPO-V7.1-P1Q-same-state-MC-development",
        "protocol_version": protocol["protocol_version"],
        "scientific_result": False,
        "wvr_q_test_accessed": False,
        "selected_target_update_epochs": int(selected_period),
        "architecture": q_payload.get("architecture", "action_embedding_v1"),
        "contexts": contexts, "actions_per_context": actions,
        "rollout_decisions": cfg["rollout_decisions"],
        "decisions_executed": decisions_executed,
        "unfinished_branches": unfinished,
        "common_random_numbers": True,
        "metrics": metrics,
        "gates": gates,
        "all_mc_gates_pass": all(item["pass"] for item in gates.values()),
        "formal_q_test_authorized": all(item["pass"] for item in gates.values()),
        "o0_authorized": False, "long_training_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["all_mc_gates_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
