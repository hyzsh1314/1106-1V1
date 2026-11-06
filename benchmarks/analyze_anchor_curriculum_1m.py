"""Analyze the 1M fixed-anchor population-curriculum experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEEDS = (85, 185, 285)
OPPONENTS = ("pursuit_fire", "hold", "evasive_fire", "standoff_fire")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    return parser.parse_args()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def metrics_rows(path):
    by_update = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            by_update[int(item["update"])] = item
    return [by_update[key] for key in sorted(by_update)]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    root = args.root_dir
    baseline = args.baseline_root
    output = root / "analysis-output"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    arm_roots = {
        "old_contract": baseline / "control",
        "timeout_zero": baseline / "candidate",
        "anchor_curriculum": root / "candidate",
    }
    rows = []
    for arm, arm_root in arm_roots.items():
        for seed in SEEDS:
            for opponent in OPPONENTS:
                summary = load(
                    arm_root / f"seed{seed}" / opponent / "summary.json")
                deaths = summary.get("learner_death_causes", {})
                episodes = int(summary["episodes"])
                rows.append({
                    "arm": arm,
                    "seed": seed,
                    "opponent": opponent,
                    "episodes": episodes,
                    "kill_win_rate": float(summary["kill_win_rate"]),
                    "kill_loss_rate": float(summary["kill_loss_rate"]),
                    "timeout_rate": float(summary["timeout_rate"]),
                    "crash_rate": int(deaths.get("crash", 0)) / episodes,
                    "shotdown_rate": int(deaths.get("shotdown", 0)) / episodes,
                    "launch_acceptance_rate": summary.get(
                        "legal_launch_acceptance_rate"),
                    "weapon_conservation": bool(summary.get(
                        "weapon_event_conservation_ok", False)),
                    "resolved_weapon_hit_rate": summary.get(
                        "resolved_weapon_hit_rate"),
                    "episodes_with_launch_rate": summary.get(
                        "episodes_with_launch_rate"),
                })
    write_csv(output / "three-arm-fixed-opponent-metrics.csv", rows)

    health = []
    for seed in SEEDS:
        run_dir = root / "candidate-runs" / f"seed{seed}_1m"
        updates = metrics_rows(run_dir / "metrics.jsonl")
        split = max(1, int(np.ceil(0.2 * len(updates))))
        early, late = updates[:split], updates[-split:]

        def fraction(group, key, nested=None):
            episodes = sum(int(item["episodes_this_update"])
                           for item in group)
            if nested is None:
                count = sum(int(item[key]) for item in group)
            else:
                count = sum(int(item.get(key, {}).get(nested, 0))
                            for item in group)
            return count / episodes if episodes else float("nan")

        completion = load(run_dir / "COMPLETED.json")
        early_timeout = fraction(early, "timeouts")
        late_timeout = fraction(late, "timeouts")
        health.append({
            "seed": seed,
            "steps": int(completion["total_steps"]),
            "throughput_steps_s": float(
                completion["session_throughput_steps_s"]),
            "early_timeout_rate": early_timeout,
            "late_timeout_rate": late_timeout,
            "timeout_delta": late_timeout - early_timeout,
            "early_low_altitude_rate": fraction(
                early, "learner_termination_conditions", "LowAltitude"),
            "late_low_altitude_rate": fraction(
                late, "learner_termination_conditions", "LowAltitude"),
            "launch_acceptance_rate": completion[
                "legal_launch_acceptance_rate"],
        })
    write_csv(output / "training-health.csv", health)

    def values(arm, opponent, metric):
        return np.asarray([
            row[metric] for row in rows
            if row["arm"] == arm and row["opponent"] == opponent
        ], dtype=np.float64)

    deltas = {}
    for opponent in OPPONENTS:
        anchor = values("anchor_curriculum", opponent, "kill_win_rate")
        old = values("old_contract", opponent, "kill_win_rate")
        zero = values("timeout_zero", opponent, "kill_win_rate")
        deltas[opponent] = {
            "anchor_minus_old_kill": float(np.mean(anchor - old)),
            "anchor_minus_timeout_zero_kill": float(np.mean(anchor - zero)),
            "anchor_minus_old_timeout": float(np.mean(values(
                "anchor_curriculum", opponent, "timeout_rate") - values(
                "old_contract", opponent, "timeout_rate"))),
            "anchor_minus_timeout_zero_crash": float(np.mean(values(
                "anchor_curriculum", opponent, "crash_rate") - values(
                "timeout_zero", opponent, "crash_rate"))),
        }

    gates = {
        "weapon_event_conservation": all(
            row["weapon_conservation"] for row in rows
            if row["arm"] == "anchor_curriculum"),
        "launch_acceptance": all(
            row["launch_acceptance_rate"] in (None, 1.0) for row in rows
            if row["arm"] == "anchor_curriculum"),
        "simple_opponents_within_5pp_of_old_contract": all(
            deltas[name]["anchor_minus_old_kill"] >= -0.05
            for name in ("pursuit_fire", "hold", "standoff_fire")),
        "evasive_improved_over_old_contract_10pp": (
            deltas["evasive_fire"]["anchor_minus_old_kill"] >= 0.10),
        "evasive_not_worse_than_timeout_zero_2pp": (
            deltas["evasive_fire"]["anchor_minus_timeout_zero_kill"]
            >= -0.02),
        "late_timeout_growth_at_most_10pp": all(
            row["timeout_delta"] <= 0.10 for row in health),
        "late_timeout_at_most_35pct": all(
            row["late_timeout_rate"] <= 0.35 for row in health),
        "evasive_crash_not_worse_than_timeout_zero_5pp": (
            deltas["evasive_fire"]["anchor_minus_timeout_zero_crash"]
            <= 0.05),
    }

    x = np.arange(len(OPPONENTS))
    width = 0.25
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9.0), sharex=True)
    arms = ("old_contract", "timeout_zero", "anchor_curriculum")
    for index, arm in enumerate(arms):
        offset = (index - 1) * width
        for axis, metric, scale in (
                (axes[0], "kill_win_rate", 100.0),
                (axes[1], "timeout_rate", 100.0),
                (axes[2], "crash_rate", 100.0)):
            means = [scale * values(arm, name, metric).mean()
                     for name in OPPONENTS]
            stds = [scale * values(arm, name, metric).std(ddof=1)
                    for name in OPPONENTS]
            axis.bar(x + offset, means, width, yerr=stds, capsize=3,
                     label=arm)
    axes[0].set_ylabel("Kill win rate (%)")
    axes[1].set_ylabel("Timeout rate (%)")
    axes[2].set_ylabel("Learner crash rate (%)")
    axes[2].set_xticks(x, OPPONENTS, rotation=15)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False, ncol=3, fontsize=8)
    fig.suptitle("1M three-arm evaluation: fixed-anchor curriculum")
    fig.tight_layout()
    fig.savefig(figures / "three-arm-kill-timeout-crash.png", dpi=180)
    plt.close(fig)

    delta_lines = "\n".join(
        f"- {name}: vs old kill "
        f"{100 * item['anchor_minus_old_kill']:+.1f} pp; "
        f"vs timeout-zero kill "
        f"{100 * item['anchor_minus_timeout_zero_kill']:+.1f} pp; "
        f"timeout vs old {100 * item['anchor_minus_old_timeout']:+.1f} pp; "
        f"crash vs timeout-zero "
        f"{100 * item['anchor_minus_timeout_zero_crash']:+.1f} pp"
        for name, item in deltas.items())
    gate_lines = "\n".join(
        f"- {'PASS' if passed else 'FAIL'} — {name}"
        for name, passed in gates.items())
    report = f"""# 100万步固定对手锚点种群课程试验

## 设计

三种子、96环境、每种子1,007,616步。每次更新75%环境采样历史自博弈种群，25%环境均分给 pursuit、evasive、standoff 固定锚点。超时学习效用保持为零。固定对手每种子每类192局，和无锚点组使用相同评估随机种子。

## 三臂变化

{delta_lines}

## 验收门

{gate_lines}
"""
    (output / "analysis-report.md").write_text(report, encoding="utf-8")
    completion = {
        "status": "completed",
        "acceptance_pass": all(gates.values()),
        "gates": gates,
        "deltas": deltas,
    }
    (root / "COMPLETED.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(completion, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
