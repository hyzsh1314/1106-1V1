"""Summarize held-out results for validation-selected checkpoints."""

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


def main():
    args = parse_args()
    root = args.root_dir
    baseline = args.baseline_root
    selected = load(root / "checkpoint-selection" / "selected.json")
    rows = []
    for seed in SEEDS:
        for opponent in OPPONENTS:
            summary = load(
                root / "heldout" / f"seed{seed}" / opponent / "summary.json")
            old = load(
                baseline / "control" / f"seed{seed}" /
                opponent / "summary.json")
            no_anchor = load(
                baseline / "candidate" / f"seed{seed}" /
                opponent / "summary.json")
            deaths = summary.get("learner_death_causes", {})
            rows.append({
                "seed": seed,
                "selected_update": selected[str(seed)]["update"],
                "opponent": opponent,
                "episodes": int(summary["episodes"]),
                "kill_win_rate": float(summary["kill_win_rate"]),
                "timeout_rate": float(summary["timeout_rate"]),
                "crash_rate": int(deaths.get("crash", 0)) /
                    int(summary["episodes"]),
                "old_contract_kill_rate": float(old["kill_win_rate"]),
                "timeout_zero_kill_rate": float(no_anchor["kill_win_rate"]),
                "weapon_conservation": bool(summary.get(
                    "weapon_event_conservation_ok", False)),
                "launch_acceptance_rate": summary.get(
                    "legal_launch_acceptance_rate"),
            })
    output = root / "checkpoint-selection"
    with (output / "heldout-results.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def values(field, opponent):
        return np.asarray([
            row[field] for row in rows if row["opponent"] == opponent
        ], dtype=np.float64)

    deltas = {}
    for opponent in OPPONENTS:
        selected_values = values("kill_win_rate", opponent)
        deltas[opponent] = {
            "selected_minus_old_kill": float(np.mean(
                selected_values - values("old_contract_kill_rate", opponent))),
            "selected_minus_timeout_zero_kill": float(np.mean(
                selected_values - values("timeout_zero_kill_rate", opponent))),
            "selected_timeout_rate": float(np.mean(values(
                "timeout_rate", opponent))),
            "selected_crash_rate": float(np.mean(values(
                "crash_rate", opponent))),
        }
    gates = {
        "weapon_event_conservation": all(
            row["weapon_conservation"] for row in rows),
        "launch_acceptance": all(
            row["launch_acceptance_rate"] in (None, 1.0) for row in rows),
        "simple_within_5pp_of_old": all(
            deltas[name]["selected_minus_old_kill"] >= -0.05
            for name in ("pursuit_fire", "hold", "standoff_fire")),
        "evasive_improved_over_old_10pp": (
            deltas["evasive_fire"]["selected_minus_old_kill"] >= 0.10),
        "all_fixed_opponent_timeouts_at_most_20pct": all(
            item["selected_timeout_rate"] <= 0.20
            for item in deltas.values()),
    }

    x = np.arange(len(OPPONENTS))
    width = 0.25
    fig, axis = plt.subplots(figsize=(9.0, 4.6))
    for index, (field, label) in enumerate((
            ("old_contract_kill_rate", "old contract"),
            ("timeout_zero_kill_rate", "timeout zero"),
            ("kill_win_rate", "validation-selected"))):
        means = [100 * values(field, name).mean() for name in OPPONENTS]
        stds = [100 * values(field, name).std(ddof=1) for name in OPPONENTS]
        axis.bar(x + (index - 1) * width, means, width, yerr=stds,
                 capsize=3, label=label)
    axis.set_ylabel("Kill win rate (%)")
    axis.set_xticks(x, OPPONENTS, rotation=15)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "heldout-selected-kill-rate.png", dpi=180)
    plt.close(fig)

    lines = "\n".join(
        f"- {name}: vs old {100 * item['selected_minus_old_kill']:+.1f} pp; "
        f"vs timeout-zero {100 * item['selected_minus_timeout_zero_kill']:+.1f} pp; "
        f"timeout {100 * item['selected_timeout_rate']:.1f}%; "
        f"crash {100 * item['selected_crash_rate']:.1f}%"
        for name, item in deltas.items())
    selected_lines = "\n".join(
        f"- seed {seed}: update {selected[str(seed)]['update']}, "
        f"validation score {selected[str(seed)]['score']:.4f}"
        for seed in SEEDS)
    gate_lines = "\n".join(
        f"- {'PASS' if passed else 'FAIL'} — {name}"
        for name, passed in gates.items())
    report = f"""# 锚点课程检查点选择与独立复验

## 验证集选择

{selected_lines}

## 独立测试集结果

{lines}

## 验收门

{gate_lines}
"""
    (output / "selection-report.md").write_text(report, encoding="utf-8")
    completion = {
        "status": "completed",
        "acceptance_pass": all(gates.values()),
        "selected": selected,
        "deltas": deltas,
        "gates": gates,
    }
    (output / "COMPLETED.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(completion, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
