#!/usr/bin/env python3
"""Collect same-state tactical macro-option values for the V8 audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from benchmarks.benchmark_torch6dof_ppo import lowlevel_controls
from benchmarks.collect_drvoi_v7_1_p1q_mc_data import load_models
from benchmarks.collect_drvoi_v7_2_state_sufficiency_data import (
    build_representations,
)
from benchmarks.drvoi import multi_discrete_to_one_hot
from benchmarks.drvoi_v7_1.macro_options import (
    NUM_MACRO_OPTIONS,
    OPTION_NAMES,
    macro_option_actions,
    macro_option_legal_mask,
)
from benchmarks.train_torch_wvr_ppo import fixed_mixture_actions
from benchmarks.validate_drvoi_v7_1_p1q_mc_probe import repeat_probe_states
from envs.torch6dof import TorchWVRConfig, TorchWVRVecEnv


REPRESENTATION_KEYS = (
    "registered_v7_1", "full_observation", "mechanistic_oracle",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b0-checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"),
                        required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, default=Path(
        "configs/drvoi_rhppo_v8_macro_option.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=int,
                        help="development override; recorded in the report")
    parser.add_argument("--warmup", type=int, nargs="+",
                        help="development override; recorded in the report")
    parser.add_argument("--continuation-samples", type=int,
                        help="development override; recorded in the report")
    return parser.parse_args()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add_discounted_reward(total, reward, active, elapsed, gamma):
    return total + (gamma ** elapsed) * reward * active.to(reward.dtype)


def bootstrapped_return(total, tail_value, active, elapsed, gamma):
    """Append a B0 continuation value without bootstrapping terminal branches."""
    return total + (gamma ** elapsed) * tail_value * active.to(total.dtype)


def aggregate_rollouts(value, contexts, options, samples):
    return value.reshape(contexts, options, samples).mean(-1)


def _policy_action(policy, belief, observation, deterministic):
    distributions = policy.distributions(belief, observation)
    return torch.stack([
        distribution.probs.argmax(-1) if deterministic
        else distribution.sample()
        for distribution in distributions], -1)


@torch.inference_mode()
def collect_macro_option_values(
        policy, lowlevel, contexts, warmup, seed, return_horizons, gamma,
        commitment_steps, continuation_samples, stochastic_continuation,
        include_full_episode, device):
    options = NUM_MACRO_OPTIONS
    branches = options * continuation_samples
    num_envs = contexts * branches
    env = TorchWVRVecEnv(TorchWVRConfig(
        num_envs=num_envs, reward_mode="competitive", attack_quality_weight=0,
        observation_version="v4", use_dynamic_launch_envelope=True),
        device=device, seed=seed, compile_dynamics=True)
    repeat_probe_states(env, contexts, branches)
    env_index = torch.arange(num_envs, device=device)
    learner_side_context = torch.arange(contexts, device=device).remainder(2)
    learner_side = learner_side_context.repeat_interleave(branches)
    opponent_kind_context = torch.arange(contexts, device=device).remainder(3)
    opponent_kind = opponent_kind_context.repeat_interleave(branches)
    hidden = policy.initial_hidden(num_envs, device=device)
    previous_action = torch.zeros(num_envs, 13, device=device)
    reset = torch.ones(num_envs, dtype=torch.bool, device=device)
    lowlevel_hidden = torch.zeros(num_envs * 2, 1, 128, device=device)
    obs = env.observations()
    base_belief = base_action = None

    for step in range(warmup):
        learner_obs = obs[env_index, learner_side]
        result = policy.act(
            learner_obs, previous_action, hidden, reset=reset,
            deterministic=False)
        learner_action = result["action"][::branches].repeat_interleave(
            branches, 0)
        if step == warmup - 3:
            base_belief = result["belief"][::branches].clone()
            base_action = multi_discrete_to_one_hot(
                learner_action[::branches]).clone()
        executed = fixed_mixture_actions(obs, opponent_kind)
        executed[env_index, learner_side] = learner_action
        controls, next_lowlevel_hidden = lowlevel_controls(
            lowlevel, lowlevel_hidden, obs, executed,
            torch_speed_adapter=True)
        obs, _, done, _ = env.step(
            controls.reshape(num_envs, 2, 4), executed[..., 3] == 1)
        if bool(done.any()):
            raise RuntimeError(f"episode terminated during {warmup}-step warm-up")
        previous_action = multi_discrete_to_one_hot(learner_action)
        hidden = result["hidden"]
        reset = done
        lowlevel_hidden = next_lowlevel_hidden
    if base_belief is None or base_action is None:
        raise RuntimeError("warmup must contain at least three decisions")

    decision_obs_all = obs[env_index, learner_side]
    decision_belief_all, decision_hidden = policy.belief_step(
        decision_obs_all, previous_action, hidden, reset)
    decision_observation = decision_obs_all[::branches]
    decision_belief = decision_belief_all[::branches]
    learner_lowlevel = lowlevel_hidden.reshape(
        num_envs, 2, 1, 128)[env_index, learner_side, 0][::branches]
    all_representations = build_representations(
        base_belief, base_action, decision_belief,
        decision_observation, learner_lowlevel)
    representations = {
        key: all_representations[key] for key in REPRESENTATION_KEYS}
    legal = macro_option_legal_mask(decision_observation)

    option_per_context = torch.arange(
        options, device=device).repeat_interleave(continuation_samples)
    option_ids = option_per_context.repeat(contexts)
    total = torch.zeros(num_envs, device=device)
    active = torch.ones(num_envs, dtype=torch.bool, device=device)
    hidden = decision_hidden
    elapsed = 0
    checkpoints = {}
    bootstrap_checkpoints = {}
    rollout_limit = (
        env.config.max_episode_steps - warmup
        if include_full_episode else max(return_horizons))

    while elapsed < rollout_limit and bool(active.any()):
        learner_obs = obs[env_index, learner_side]
        if elapsed == 0:
            belief = decision_belief_all
            hidden_after = decision_hidden
        else:
            belief, hidden_after = policy.belief_step(
                learner_obs, previous_action, hidden, reset)
        if elapsed < commitment_steps:
            learner_action = macro_option_actions(
                learner_obs, option_ids, elapsed=elapsed)
        else:
            learner_action = _policy_action(
                policy, belief, learner_obs,
                deterministic=not stochastic_continuation)
        executed = fixed_mixture_actions(obs, opponent_kind)
        executed[env_index, learner_side] = learner_action
        controls, next_lowlevel_hidden = lowlevel_controls(
            lowlevel, lowlevel_hidden, obs, executed,
            torch_speed_adapter=True)
        next_obs, reward, done, _ = env.step(
            controls.reshape(num_envs, 2, 4), executed[..., 3] == 1)
        learner_reward = reward[env_index, learner_side]
        total = add_discounted_reward(
            total, learner_reward, active, elapsed, gamma)
        active &= ~done
        elapsed += 1
        encoded = multi_discrete_to_one_hot(learner_action)
        next_previous_action = torch.where(
            done[:, None], torch.zeros_like(encoded), encoded)

        if elapsed in return_horizons:
            next_learner_obs = next_obs[env_index, learner_side]
            next_belief, _ = policy.belief_step(
                next_learner_obs, next_previous_action, hidden_after, done)
            tail_value = policy.value_ext(next_belief).squeeze(-1)
            checkpoints[elapsed] = total.clone()
            bootstrap_checkpoints[elapsed] = bootstrapped_return(
                total, tail_value, active, elapsed, gamma)

        previous_action = next_previous_action
        hidden = hidden_after
        reset = done
        obs = next_obs
        lowlevel_hidden = torch.where(
            done.repeat_interleave(2)[:, None, None],
            torch.zeros_like(next_lowlevel_hidden), next_lowlevel_hidden)

    for horizon in return_horizons:
        if horizon not in checkpoints:
            checkpoints[horizon] = total.clone()
            bootstrap_checkpoints[horizon] = total.clone()

    result = {key: value.cpu() for key, value in representations.items()}
    for horizon in return_horizons:
        result[f"return_{horizon}"] = aggregate_rollouts(
            checkpoints[horizon], contexts, options, continuation_samples).cpu()
        result[f"bootstrap_return_{horizon}"] = aggregate_rollouts(
            bootstrap_checkpoints[horizon], contexts, options,
            continuation_samples).cpu()
    if include_full_episode:
        result["return_full"] = aggregate_rollouts(
            total, contexts, options, continuation_samples).cpu()
    result.update({
        "legal": legal.cpu(),
        "warmup_decisions": torch.full((contexts,), warmup),
        "learner_side": learner_side_context.cpu(),
        "opponent_kind": opponent_kind_context.cpu(),
    })
    del env
    torch.cuda.empty_cache()
    return result


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    cfg = protocol["data"]
    allowed_seeds = cfg[f"{args.split}_seeds"]
    if args.seed not in allowed_seeds:
        raise ValueError(f"seed {args.seed} is not configured for {args.split}")
    contexts = args.contexts or cfg[f"{args.split}_contexts_per_warmup"]
    warmups = args.warmup or cfg["warmup_decisions"]
    samples = args.continuation_samples or cfg["continuation_samples"]
    if contexts <= 0 or samples <= 0 or min(warmups) < 3:
        raise ValueError("contexts/samples must be positive and warmup >= 3")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    policy, lowlevel = load_models(args.b0_checkpoint, device)
    batches = []
    for index, warmup in enumerate(warmups):
        batch = collect_macro_option_values(
            policy, lowlevel, contexts, warmup, args.seed + 1000 * index,
            cfg["return_horizons"], cfg["gamma"], cfg["commitment_steps"],
            samples, cfg["stochastic_continuation"],
            cfg["include_full_episode_return"], device)
        batches.append(batch)
        print(json.dumps({
            "split": args.split, "seed": args.seed, "warmup": warmup,
            "contexts": contexts, "options": NUM_MACRO_OPTIONS,
            "continuation_samples": samples,
        }), flush=True)

    return_keys = tuple(
        key for horizon in cfg["return_horizons"]
        for key in (f"return_{horizon}", f"bootstrap_return_{horizon}"))
    if cfg["include_full_episode_return"]:
        return_keys = (*return_keys, "return_full")
    tensor_keys = (
        *REPRESENTATION_KEYS, *return_keys, "legal", "warmup_decisions",
        "learner_side", "opponent_kind")
    data = {key: torch.cat([batch[key] for batch in batches])
            for key in tensor_keys}
    data["collection_seed"] = torch.full(
        (data["legal"].shape[0],), args.seed, dtype=torch.long)
    overrides = {
        "contexts": args.contexts,
        "warmup": args.warmup,
        "continuation_samples": args.continuation_samples,
    }
    report = {
        "suite": "DRVOI-RHPPO-V8-macro-option-collection",
        "protocol_version": protocol["protocol_version"],
        "scientific_result": False,
        "split": args.split,
        "collection_seed": args.seed,
        "contexts": int(data["legal"].shape[0]),
        "options_per_context": NUM_MACRO_OPTIONS,
        "option_names": list(OPTION_NAMES),
        "commitment_steps": cfg["commitment_steps"],
        "continuation_samples": samples,
        "stochastic_continuation": cfg["stochastic_continuation"],
        "return_horizons": cfg["return_horizons"],
        "target_variants": ["truncated", "b0_value_bootstrap"],
        "representations": list(REPRESENTATION_KEYS),
        "development_overrides": overrides,
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
