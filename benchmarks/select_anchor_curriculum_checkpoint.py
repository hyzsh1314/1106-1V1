"""Select robust anchor-curriculum checkpoints on an independent validation set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SEEDS = (85, 185, 285)
UPDATES = (20, 40, 41)
OPPONENTS = ("pursuit_fire", "hold", "evasive_fire", "standoff_fire")
SIMPLE = ("pursuit_fire", "hold", "standoff_fire")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, required=True)
    return parser.parse_args()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    args = parse_args()
    root = args.root_dir
    rows = []
    selected = {}
    for seed in SEEDS:
        candidates = []
        for update in UPDATES:
            summaries = {
                opponent: load(
                    root / "validation" / f"seed{seed}" /
                    f"update{update}" / opponent / "summary.json")
                for opponent in OPPONENTS
            }
            kills = np.asarray([
                float(summaries[name]["kill_win_rate"])
                for name in OPPONENTS])
            simple_kills = np.asarray([
                float(summaries[name]["kill_win_rate"])
                for name in SIMPLE])
            timeouts = np.asarray([
                float(summaries[name]["timeout_rate"])
                for name in OPPONENTS])
            crashes = np.asarray([
                int(summaries[name].get(
                    "learner_death_causes", {}).get("crash", 0)) /
                int(summaries[name]["episodes"])
                for name in OPPONENTS])
            # A constrained robust score.  The 70% simple-opponent floor
            # prevents selecting an evasive specialist that forgot basic
            # pursuit, while the minimum term rewards worst-case competence.
            simple_floor_shortfall = max(0.0, 0.70 - float(simple_kills.min()))
            score = (
                float(kills.mean()) + 0.5 * float(kills.min())
                - 0.5 * float(timeouts.mean())
                - 0.5 * float(crashes.mean())
                - 2.0 * simple_floor_shortfall
            )
            row = {
                "seed": seed,
                "update": update,
                "score": score,
                "mean_kill_rate": float(kills.mean()),
                "worst_kill_rate": float(kills.min()),
                "simple_min_kill_rate": float(simple_kills.min()),
                "evasive_kill_rate": float(
                    summaries["evasive_fire"]["kill_win_rate"]),
                "mean_timeout_rate": float(timeouts.mean()),
                "mean_crash_rate": float(crashes.mean()),
            }
            for opponent in OPPONENTS:
                row[f"{opponent}_kill_rate"] = float(
                    summaries[opponent]["kill_win_rate"])
                row[f"{opponent}_timeout_rate"] = float(
                    summaries[opponent]["timeout_rate"])
            rows.append(row)
            candidates.append(row)
        winner = max(candidates, key=lambda item: (item["score"],
                                                   -item["update"]))
        checkpoint_file = (
            "latest.pt" if winner["update"] == 41 else
            f"checkpoints/checkpoint_{winner['update']:06d}.pt")
        selected[str(seed)] = {
            **winner,
            "checkpoint_file": checkpoint_file,
        }

    output = root / "checkpoint-selection"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "validation-scores.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "selected.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(selected, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
