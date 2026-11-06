"""Side-balanced cross-play for two deterministic scripted 1v1 policies."""

from __future__ import annotations

import argparse
import csv
from functools import partial
import json
from pathlib import Path

import numpy as np

from benchmarks.evaluate_ppo_1v1_acceptance import OPPONENT_POLICIES
from benchmarks.train_ppo_1v1_shoot_smoke import make_environment
from envs.env_wrappers import DummyVecEnv, SubprocVecEnv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-a", choices=tuple(OPPONENT_POLICIES),
                        required=True)
    parser.add_argument("--policy-b", choices=tuple(OPPONENT_POLICIES),
                        required=True)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--num-envs", type=int, default=96)
    parser.add_argument("--seed", type=int, default=990085)
    parser.add_argument("--start-method", choices=("spawn", "forkserver"),
                        default="forkserver")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def evaluate_side(policy_a, policy_b, *, a_side, episodes, num_envs, seed,
                  start_method):
    factories = [partial(
        make_environment, seed + index * 1009,
        scenario="1v1/ShootMissile/HierarchySelfplay",
        reward_variant="R1_attack_pbrs")
        for index in range(num_envs)]
    env = (DummyVecEnv(factories) if num_envs == 1 else
           SubprocVecEnv(factories, context=start_method, in_series=1))
    obs = env.reset()
    a_index = 0 if a_side == "ego" else 1
    b_index = 1 - a_index
    records = []
    try:
        while len(records) < episodes:
            joint = np.empty((num_envs, 2, 4), dtype=np.int64)
            joint[:, a_index] = policy_a(obs[:, a_index])
            joint[:, b_index] = policy_b(obs[:, b_index])
            obs, _reward, done, infos = env.step(joint)
            done = np.asarray(done, dtype=np.bool_).reshape(num_envs, 2)
            flat_infos = np.asarray(infos, dtype=object).reshape(-1)
            for index in np.flatnonzero(np.all(done, axis=1)):
                if len(records) >= episodes:
                    break
                info = flat_infos[int(index)]
                outcome = info.get("team_outcome", {})
                ego_result = int(outcome.get("ego_result", 0))
                result = ego_result if a_index == 0 else -ego_result
                reason = str(outcome.get("reason", "unknown"))
                elimination = reason in {
                    "ego_team_eliminated", "enemy_team_eliminated",
                    "simultaneous_elimination",
                }
                records.append({
                    "policy_a_side": a_side,
                    "result": result,
                    "kill_result": result if elimination else 0,
                    "timeout_result": (
                        result if reason.startswith("timeout") else 0),
                    "reason": reason,
                    "episode_steps": int(outcome.get("episode_steps", 0)),
                })
    finally:
        env.close()
    return records


def main():
    args = parse_args()
    if args.episodes < 2 or args.num_envs <= 0:
        raise ValueError("episodes must be at least two and num-envs positive")
    policy_a = OPPONENT_POLICIES[args.policy_a]
    policy_b = OPPONENT_POLICIES[args.policy_b]
    first = args.episodes // 2
    records = evaluate_side(
        policy_a, policy_b, a_side="ego", episodes=first,
        num_envs=args.num_envs, seed=args.seed,
        start_method=args.start_method)
    records.extend(evaluate_side(
        policy_a, policy_b, a_side="enemy",
        episodes=args.episodes - first, num_envs=args.num_envs,
        seed=args.seed + 500_000, start_method=args.start_method))
    reasons = {}
    for record in records:
        reasons[record["reason"]] = reasons.get(record["reason"], 0) + 1
    count = len(records)
    summary = {
        "schema": "WVR1v1ScriptedCrossplayV1",
        "policy_a": args.policy_a,
        "policy_b": args.policy_b,
        "episodes": count,
        "side_balanced": True,
        "kill_win_rate": sum(r["kill_result"] > 0 for r in records) / count,
        "kill_loss_rate": sum(r["kill_result"] < 0 for r in records) / count,
        "timeout_rate": sum(
            r["reason"].startswith("timeout") for r in records) / count,
        "adjudicated_win_rate": sum(r["result"] > 0 for r in records) / count,
        "mean_episode_steps": float(np.mean([
            r["episode_steps"] for r in records])),
        "termination_reason_counts": reasons,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "episodes.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
