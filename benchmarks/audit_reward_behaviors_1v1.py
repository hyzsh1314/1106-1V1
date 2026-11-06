"""Audit R0-R3 rewards on identical fixed-policy 1v1 trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from benchmarks.evaluate_ppo_1v1_acceptance import scripted_pursuit_fire
from benchmarks.train_ppo_1v1_shoot_smoke import (
    REWARD_VARIANT_OVERRIDES,
    make_environment,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-per-matchup", type=int, default=16)
    parser.add_argument("--seed", type=int, default=470085)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("runs/reward_v4_behavior_audit"))
    return parser.parse_args()


def hold_policy(observation):
    actions = np.tile(
        np.asarray([1, 2, 1, 0], dtype=np.int64),
        (len(observation), 1))
    return actions


POLICIES = {
    "pursuit_fire": scripted_pursuit_fire,
    "hold": hold_policy,
}

MATCHUPS = (
    ("pursuit_fire", "hold"),
    ("hold", "pursuit_fire"),
    ("pursuit_fire", "pursuit_fire"),
    ("hold", "hold"),
)


def run_episode(env, ego_policy, enemy_policy, gamma):
    obs = env.reset()
    discounted_return = 0.0
    undiscounted_return = 0.0
    discount = 1.0
    component_returns = {}
    launches = requests = rejected = opportunities = 0
    while True:
        ego_action = ego_policy(obs[0:1])[0]
        enemy_action = enemy_policy(obs[1:2])[0]
        obs, reward, done, info = env.step(np.stack(
            (ego_action, enemy_action), axis=0))
        ego_reward = float(reward[0, 0])
        discounted_return += discount * ego_reward
        undiscounted_return += ego_reward
        for name, values in info.get("team_reward_components", {}).items():
            component_returns[name] = (
                component_returns.get(name, 0.0) +
                discount * float(values[0]))
        launches += int(info.get("weapon_accepted_launches", 0))
        requests += int(info.get("weapon_fire_requests", 0))
        rejected += int(info.get("weapon_rejected_requests", 0))
        opportunities += int(info.get("weapon_legal_opportunities", 0))
        discount *= gamma
        if np.all(done):
            outcome = info["team_outcome"]
            return {
                "result": int(outcome["ego_result"]),
                "reason": str(outcome["reason"]),
                "episode_steps": int(outcome["episode_steps"]),
                "discounted_return": discounted_return,
                "undiscounted_return": undiscounted_return,
                "launches": launches,
                "requests": requests,
                "rejected": rejected,
                "opportunities": opportunities,
                "components": component_returns,
            }


def main():
    args = parse_args()
    if args.episodes_per_matchup <= 0:
        raise ValueError("episodes-per-matchup must be positive")
    records = []
    variants = tuple(REWARD_VARIANT_OVERRIDES)
    for variant in variants:
        env = make_environment(args.seed, reward_variant=variant)
        try:
            gamma = float(env.config.PPO_gamma)
            for ego_name, enemy_name in MATCHUPS:
                matchup = f"{ego_name}_vs_{enemy_name}"
                for episode in range(args.episodes_per_matchup):
                    result = run_episode(
                        env, POLICIES[ego_name], POLICIES[enemy_name], gamma)
                    components = result.pop("components")
                    records.append({
                        "variant": variant,
                        "matchup": matchup,
                        "episode": episode,
                        **result,
                        "components_json": json.dumps(
                            components, sort_keys=True),
                    })
        finally:
            env.close()

    # Reward variants must never change trajectories when actions and seeds
    # are identical.  Compare physical fingerprints against R0.
    keys = ["matchup", "episode"]
    physical = ["result", "reason", "episode_steps", "launches",
                "requests", "rejected", "opportunities"]
    baseline = {
        tuple(row[key] for key in keys): tuple(row[key] for key in physical)
        for row in records if row["variant"] == "R0_terminal"
    }
    for row in records:
        fingerprint = tuple(row[key] for key in physical)
        if fingerprint != baseline[tuple(row[key] for key in keys)]:
            raise AssertionError(
                "reward variant changed a fixed-policy physical trajectory")
        if row["rejected"] != 0:
            raise AssertionError("action-masked scripted policy emitted a rejection")

    r0 = [row for row in records if row["variant"] == "R0_terminal"]
    for row in r0:
        expected = (float(row["result"]) *
                    (0.99 ** (int(row["episode_steps"]) - 1)))
        if not np.isclose(row["discounted_return"], expected, atol=1e-6):
            raise AssertionError("R0 return is not the normalized terminal payoff")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0])
    with (args.output_dir / "behavior-audit.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    summaries = []
    for variant in variants:
        for matchup in (f"{a}_vs_{b}" for a, b in MATCHUPS):
            group = [row for row in records
                     if row["variant"] == variant and row["matchup"] == matchup]
            summaries.append({
                "variant": variant,
                "matchup": matchup,
                "episodes": len(group),
                "wins": sum(row["result"] > 0 for row in group),
                "draws": sum(row["result"] == 0 for row in group),
                "losses": sum(row["result"] < 0 for row in group),
                "mean_discounted_return": float(np.mean([
                    row["discounted_return"] for row in group])),
                "mean_episode_steps": float(np.mean([
                    row["episode_steps"] for row in group])),
                "total_launches": sum(row["launches"] for row in group),
            })
    (args.output_dir / "summary.json").write_text(
        json.dumps({
            "status": "PASS",
            "episodes_per_matchup": args.episodes_per_matchup,
            "physical_trajectory_invariant_across_variants": True,
            "rejected_fire_requests": 0,
            "summaries": summaries,
        }, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "episodes": len(records),
        "variants": list(variants),
        "matchups": [f"{a}_vs_{b}" for a, b in MATCHUPS],
        "output": str(args.output_dir.resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
