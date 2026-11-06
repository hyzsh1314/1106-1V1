"""Audit randomized initial geometry before a formal 1v1 training suite."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from benchmarks.train_ppo_1v1_shoot_smoke import make_environment


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=730085)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def aspect_class(ego_ao, enemy_ao):
    low, high = sorted((float(ego_ao), float(enemy_ao)))
    if high < np.deg2rad(60.0):
        return "head_on"
    if low < np.deg2rad(45.0) and high > np.deg2rad(135.0):
        return "pursuit"
    return "crossing"


def main():
    args = parse_args()
    if args.samples < 30:
        raise ValueError("at least 30 resets are required")
    env = make_environment(args.seed, reward_variant="R1_attack_pbrs")
    rows = []
    try:
        for sample in range(args.samples):
            obs = env.reset()
            agent_ids = list(env.agents)
            rows.append({
                "sample": sample,
                "range_m": float(obs[0, 13] * 10000.0),
                "ego_altitude_m": float(obs[0, 0] * 5000.0),
                "enemy_altitude_m": float(obs[1, 0] * 5000.0),
                "altitude_delta_m": float((obs[1, 0] - obs[0, 0]) * 5000.0),
                "ego_speed_m_s": float(np.linalg.norm(
                    env.agents[agent_ids[0]].get_velocity())),
                "enemy_speed_m_s": float(np.linalg.norm(
                    env.agents[agent_ids[1]].get_velocity())),
                "ego_ao_deg": float(np.rad2deg(obs[0, 11])),
                "enemy_ao_deg": float(np.rad2deg(obs[1, 11])),
                "aspect_class": aspect_class(obs[0, 11], obs[1, 11]),
                "initial_ego_can_fire": int(obs[0, 23] >= 0.5),
                "initial_enemy_can_fire": int(obs[1, 23] >= 0.5),
            })
    finally:
        env.close()
    ranges = np.asarray([row["range_m"] for row in rows])
    altitudes = np.asarray([[row["ego_altitude_m"], row["enemy_altitude_m"]]
                            for row in rows])
    altitude_delta = np.asarray([row["altitude_delta_m"] for row in rows])
    counts = {name: sum(row["aspect_class"] == name for row in rows)
              for name in ("head_on", "crossing", "pursuit")}
    violations = []
    if ranges.min() < 8000.0 - 20.0 or ranges.max() > 16000.0 + 20.0:
        violations.append("range_out_of_contract")
    if altitudes.min() < 5000.0 - 20.0 or altitudes.max() > 7500.0 + 20.0:
        violations.append("altitude_out_of_contract")
    if np.max(np.abs(altitude_delta)) > 600.0 + 20.0:
        violations.append("altitude_delta_out_of_contract")
    if any(count == 0 for count in counts.values()):
        violations.append("missing_aspect_class")
    summary = {
        "status": "PASS" if not violations else "FAIL",
        "samples": args.samples,
        "seed": args.seed,
        "range_m": {"min": float(ranges.min()), "mean": float(ranges.mean()),
                    "max": float(ranges.max()), "sd": float(ranges.std(ddof=1))},
        "altitude_m": {"min": float(altitudes.min()),
                       "max": float(altitudes.max())},
        "altitude_delta_mean_m": float(altitude_delta.mean()),
        "aspect_counts": counts,
        "aspect_fractions": {name: count / args.samples
                             for name, count in counts.items()},
        "initial_can_fire": {
            "ego": sum(row["initial_ego_can_fire"] for row in rows),
            "enemy": sum(row["initial_enemy_can_fire"] for row in rows),
        },
        "violations": violations,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "initial-states.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if violations:
        raise RuntimeError("initial engagement audit failed")


if __name__ == "__main__":
    main()
