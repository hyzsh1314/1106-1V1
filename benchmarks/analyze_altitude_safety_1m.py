"""Analyze the paired 1M altitude-safety action-mask experiment."""

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


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.root_dir / "analysis-output"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    sources = {
        "timeout_zero": args.baseline_root / "candidate",
        "safety_paired": args.root_dir / "candidate",
        "safety_heldout": args.root_dir / "heldout",
    }
    rows = []
    for arm, root in sources.items():
        for seed in SEEDS:
            for opponent in OPPONENTS:
                item = load(root / f"seed{seed}" / opponent / "summary.json")
                episodes = int(item["episodes"])
                deaths = item.get("learner_death_causes", {})
                rows.append({
                    "arm": arm, "seed": seed, "opponent": opponent,
                    "episodes": episodes,
                    "kill_win_rate": float(item["kill_win_rate"]),
                    "timeout_rate": float(item["timeout_rate"]),
                    "crash_rate": int(deaths.get("crash", 0)) / episodes,
                    "launch_acceptance_rate": item.get(
                        "legal_launch_acceptance_rate"),
                    "weapon_conservation": bool(item.get(
                        "weapon_event_conservation_ok", False)),
                })
    with (output / "fixed-opponent-metrics.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def values(arm, opponent, metric):
        return np.asarray([
            row[metric] for row in rows
            if row["arm"] == arm and row["opponent"] == opponent
        ], dtype=np.float64)

    deltas = {}
    for opponent in OPPONENTS:
        paired = values("safety_paired", opponent, "kill_win_rate")
        baseline = values("timeout_zero", opponent, "kill_win_rate")
        heldout = values("safety_heldout", opponent, "kill_win_rate")
        deltas[opponent] = {
            "paired_minus_baseline_kill": float(np.mean(paired - baseline)),
            "heldout_minus_paired_kill": float(np.mean(heldout - paired)),
            "paired_crash_rate": float(np.mean(values(
                "safety_paired", opponent, "crash_rate"))),
            "heldout_crash_rate": float(np.mean(values(
                "safety_heldout", opponent, "crash_rate"))),
            "paired_timeout_rate": float(np.mean(values(
                "safety_paired", opponent, "timeout_rate"))),
            "heldout_timeout_rate": float(np.mean(values(
                "safety_heldout", opponent, "timeout_rate"))),
        }

    safety_rows = [row for row in rows if row["arm"].startswith("safety_")]
    gates = {
        "weapon_event_conservation": all(
            row["weapon_conservation"] for row in safety_rows),
        "launch_acceptance": all(
            row["launch_acceptance_rate"] in (None, 1.0)
            for row in safety_rows),
        "evasive_crash_at_most_2pct_both_cohorts": (
            deltas["evasive_fire"]["paired_crash_rate"] <= 0.02 and
            deltas["evasive_fire"]["heldout_crash_rate"] <= 0.02),
        "simple_kill_within_5pp_of_baseline": all(
            deltas[name]["paired_minus_baseline_kill"] >= -0.05
            for name in ("pursuit_fire", "hold", "standoff_fire")),
        "heldout_kill_shift_at_most_10pp": all(
            abs(deltas[name]["heldout_minus_paired_kill"]) <= 0.10
            for name in OPPONENTS),
        "timeouts_at_most_30pct_both_cohorts": all(
            deltas[name][cohort] <= 0.30 for name in OPPONENTS
            for cohort in ("paired_timeout_rate", "heldout_timeout_rate")),
    }

    x = np.arange(len(OPPONENTS))
    width = 0.25
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9), sharex=True)
    for index, arm in enumerate(sources):
        offset = (index - 1) * width
        for axis, metric in zip(
                axes, ("kill_win_rate", "timeout_rate", "crash_rate")):
            means = [100 * values(arm, name, metric).mean()
                     for name in OPPONENTS]
            stds = [100 * values(arm, name, metric).std(ddof=1)
                    for name in OPPONENTS]
            axis.bar(x + offset, means, width, yerr=stds, capsize=3,
                     label=arm)
    axes[0].set_ylabel("Kill win rate (%)")
    axes[1].set_ylabel("Timeout rate (%)")
    axes[2].set_ylabel("Learner crash rate (%)")
    axes[2].set_xticks(x, OPPONENTS, rotation=15)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(figures / "altitude-safety-kill-timeout-crash.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.5, 4.5))
    for seed in SEEDS:
        path = (args.root_dir / "candidate-runs" /
                f"seed{seed}_1m" / "metrics.jsonl")
        by_update = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                by_update[int(item["update"])] = item
        ordered = [by_update[key] for key in sorted(by_update)]
        axis.plot([item["total_steps"] for item in ordered],
                  [item["reward"] for item in ordered], label=f"seed {seed}")
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_xlabel("Environment steps")
    axis.set_ylabel("Mean step reward")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "reward-convergence.png", dpi=180)
    plt.close(fig)

    delta_lines = "\n".join(
        f"- {name}: paired kill vs baseline "
        f"{100 * item['paired_minus_baseline_kill']:+.1f} pp; "
        f"heldout shift {100 * item['heldout_minus_paired_kill']:+.1f} pp; "
        f"crash {100 * item['paired_crash_rate']:.1f}%/"
        f"{100 * item['heldout_crash_rate']:.1f}%; timeout "
        f"{100 * item['paired_timeout_rate']:.1f}%/"
        f"{100 * item['heldout_timeout_rate']:.1f}%"
        for name, item in deltas.items())
    gate_lines = "\n".join(
        f"- {'PASS' if passed else 'FAIL'} — {name}"
        for name, passed in gates.items())
    report = f"""# 100万步低空动作约束对照试验

## 设计

三个独立训练种子，96并行环境，每种子1,007,616步；不使用固定对手锚点，奖励和PPO参数不变，仅在策略分布中加入低空离散动作约束。每类对手同时使用配对评估种子和全新留出种子，各192局。

## 结果

{delta_lines}

## 验收门

{gate_lines}
"""
    (output / "analysis-report.md").write_text(report, encoding="utf-8")
    completion = {
        "status": "completed", "acceptance_pass": all(gates.values()),
        "gates": gates, "deltas": deltas,
    }
    (args.root_dir / "COMPLETED.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(completion, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
