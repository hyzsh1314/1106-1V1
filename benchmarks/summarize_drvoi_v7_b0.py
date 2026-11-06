#!/usr/bin/env python3
"""Aggregate the three frozen V7 B0 runs without post-hoc performance gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = (85, 185, 285)
    opponents = ("hold", "pursuit", "evasive")
    runs = {}
    artifact_reasons = []
    for seed in seeds:
        directory = args.root / f"seed{seed}"
        train = json.loads((directory / "train.json").read_text(encoding="utf-8"))
        if int(train["total_environment_decisions"]) != 1_007_616:
            artifact_reasons.append(f"seed{seed}_training_budget")
        evaluations = {}
        for opponent in opponents:
            report = json.loads(
                (directory / f"eval-{opponent}.json").read_text(encoding="utf-8"))
            if int(report["episodes"]) != 192:
                artifact_reasons.append(f"seed{seed}_{opponent}_episodes")
            numeric = (
                report["win_rate"], report["timeout_rate"],
                report["hit_per_launch"], report["legal_request_rate"])
            if not all(math.isfinite(float(value)) for value in numeric):
                artifact_reasons.append(f"seed{seed}_{opponent}_finite")
            evaluations[opponent] = report
        runs[str(seed)] = {"training": train, "evaluations": evaluations}
    aggregate = {}
    for opponent in opponents:
        aggregate[opponent] = {}
        for metric in (
                "win_rate", "timeout_rate", "hit_per_launch",
                "legal_request_rate", "mean_episode_reward"):
            values = np.asarray([
                runs[str(seed)]["evaluations"][opponent][metric]
                for seed in seeds], dtype=float)
            aggregate[opponent][metric] = {
                "mean": float(values.mean()),
                "sample_std": float(values.std(ddof=1)),
                "by_seed": dict(zip(map(str, seeds), values.tolist())),
            }
    report = {
        "suite": "DRVOI-RHPPO-V7-B0-summary",
        "protocol_version": "DRVOI-RHPPO-V7-WVR-1",
        "scientific_result": False,
        "seeds": list(seeds),
        "artifact_gate": {
            "pass": not artifact_reasons,
            "reasons": artifact_reasons,
        },
        "performance_threshold_preregistered": False,
        "performance_note": (
            "B0 performance is descriptive; P1R directly gates whether its "
            "frozen belief is useful. No threshold is invented after seeing results."),
        "aggregate": aggregate,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if artifact_reasons:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
