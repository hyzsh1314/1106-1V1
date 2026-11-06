"""Analyze the paired 1M timeout-zero reward-contract experiment."""

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
    output = root / "analysis-output"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    rows = []
    for arm in ("control", "candidate"):
        for seed in SEEDS:
            for opponent in OPPONENTS:
                summary = load(
                    root / arm / f"seed{seed}" / opponent / "summary.json")
                rows.append({
                    "arm": arm,
                    "seed": seed,
                    "opponent": opponent,
                    "episodes": int(summary["episodes"]),
                    "kill_win_rate": float(summary["kill_win_rate"]),
                    "kill_loss_rate": float(summary["kill_loss_rate"]),
                    "timeout_rate": float(summary["timeout_rate"]),
                    "adjudicated_win_rate": float(summary["win_rate"]),
                    "launch_acceptance_rate": summary.get(
                        "legal_launch_acceptance_rate"),
                    "weapon_conservation": bool(summary.get(
                        "weapon_event_conservation_ok", False)),
                    "resolved_weapon_hit_rate": summary.get(
                        "resolved_weapon_hit_rate"),
                    "episodes_with_launch_rate": summary.get(
                        "episodes_with_launch_rate"),
                })
    write_csv(output / "paired-fixed-opponent-metrics.csv", rows)

    health = []
    for seed in SEEDS:
        run_dir = root / "candidate-runs" / f"seed{seed}_1m"
        updates = metrics_rows(run_dir / "metrics.jsonl")
        split = max(1, int(np.ceil(0.2 * len(updates))))
        early, late = updates[:split], updates[-split:]

        def timeout_fraction(group):
            episodes = sum(int(item["episodes_this_update"]) for item in group)
            timeouts = sum(int(item["timeouts"]) for item in group)
            return timeouts / episodes if episodes else float("nan")

        completion = load(run_dir / "COMPLETED.json")
        early_timeout = timeout_fraction(early)
        late_timeout = timeout_fraction(late)
        health.append({
            "seed": seed,
            "steps": int(completion["total_steps"]),
            "throughput_steps_s": float(
                completion["session_throughput_steps_s"]),
            "early_timeout_rate": early_timeout,
            "late_timeout_rate": late_timeout,
            "timeout_delta": late_timeout - early_timeout,
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
        deltas[opponent] = {
            "kill_win_delta": float(np.mean(values(
                "candidate", opponent, "kill_win_rate") - values(
                    "control", opponent, "kill_win_rate"))),
            "timeout_delta": float(np.mean(values(
                "candidate", opponent, "timeout_rate") - values(
                    "control", opponent, "timeout_rate"))),
        }

    gates = {
        "weapon_event_conservation": all(
            row["weapon_conservation"] for row in rows
            if row["arm"] == "candidate"),
        "launch_acceptance": all(
            row["launch_acceptance_rate"] in (None, 1.0) for row in rows
            if row["arm"] == "candidate"),
        "simple_opponents_not_degraded_over_5pp": all(
            deltas[name]["kill_win_delta"] >= -0.05
            for name in ("pursuit_fire", "hold", "standoff_fire")),
        "late_timeout_growth_at_most_10pp": all(
            row["timeout_delta"] <= 0.10 for row in health),
        "late_timeout_at_most_35pct": all(
            row["late_timeout_rate"] <= 0.35 for row in health),
        "evasive_improved": (
            deltas["evasive_fire"]["kill_win_delta"] >= 0.10 or
            deltas["evasive_fire"]["timeout_delta"] <= -0.15),
    }

    x = np.arange(len(OPPONENTS))
    width = 0.36
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True)
    for index, arm in enumerate(("control", "candidate")):
        offset = (index - 0.5) * width
        means = [100 * values(arm, opponent, "kill_win_rate").mean()
                 for opponent in OPPONENTS]
        stds = [100 * values(arm, opponent, "kill_win_rate").std(ddof=1)
                for opponent in OPPONENTS]
        axes[0].bar(x + offset, means, width, yerr=stds, capsize=4,
                    label=arm)
        timeout_means = [100 * values(arm, opponent, "timeout_rate").mean()
                         for opponent in OPPONENTS]
        timeout_stds = [100 * values(arm, opponent, "timeout_rate").std(ddof=1)
                        for opponent in OPPONENTS]
        axes[1].bar(x + offset, timeout_means, width, yerr=timeout_stds,
                    capsize=4, label=arm)
    axes[0].set_ylabel("Kill win rate (%)")
    axes[1].set_ylabel("Timeout rate (%)")
    axes[1].set_xticks(x, OPPONENTS, rotation=15)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
    fig.suptitle("1M paired evaluation: old contract vs timeout-zero")
    fig.tight_layout()
    fig.savefig(figures / "paired-kill-timeout.png", dpi=180)
    plt.close(fig)

    gate_lines = "\n".join(
        f"- {'PASS' if passed else 'FAIL'} — {name}"
        for name, passed in gates.items())
    delta_lines = "\n".join(
        f"- {name}: kill {100 * item['kill_win_delta']:+.1f} pp, "
        f"timeout {100 * item['timeout_delta']:+.1f} pp"
        for name, item in deltas.items())
    report = f"""# 100万步超时零奖励配对试验

## 设计

三个独立种子；控制组使用原500万步实验中的约100万步检查点，候选组仅将超时学习效用改为零。固定对手每个种子每类192局并进行物理侧平衡。误差条为三个种子的样本标准差，不把单局当作独立算法重复。

## 配对变化

{delta_lines}

## 验收门

{gate_lines}

## 决策规则

全部门槛通过才进入500万步复验。若超时门通过而规避改进门失败，下一步只增加对手课程，不再修改奖励。若超时门仍失败，先检查策略是否通过坠毁、未发射或战术评分间接利用终局。
"""
    (output / "analysis-report.md").write_text(report, encoding="utf-8")
    completion = {
        "status": "completed",
        "acceptance_pass": all(gates.values()),
        "gates": gates,
        "paired_deltas": deltas,
    }
    (root / "COMPLETED.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(completion, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
