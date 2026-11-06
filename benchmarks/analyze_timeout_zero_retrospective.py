"""Analyze kill-only checkpoint and scripted-opponent retrospective data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEEDS = (85, 185, 285)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows):
    if not rows:
        return
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
    selected = {}
    for seed in SEEDS:
        for summary_path in sorted(
                (root / f"seed{seed}").glob(
                    "checkpoint_*/evasive_fire/summary.json")):
            summary = load_json(summary_path)
            update = int(summary_path.parents[1].name.split("_")[-1])
            rows.append({
                "seed": seed,
                "update": update,
                "steps": update * 256 * 96,
                "episodes": int(summary["episodes"]),
                "kill_win_rate": float(summary["kill_win_rate"]),
                "kill_loss_rate": float(summary["kill_loss_rate"]),
                "timeout_rate": float(summary["timeout_rate"]),
                "adjudicated_win_rate": float(summary["win_rate"]),
                "resolved_weapon_hit_rate": summary.get(
                    "resolved_weapon_hit_rate"),
                "episodes_with_launch_rate": summary.get(
                    "episodes_with_launch_rate"),
            })
        candidates = [row for row in rows if row["seed"] == seed]
        if candidates:
            # Kill success is primary; timeout is the deterministic tie-break.
            best = max(candidates, key=lambda row: (
                row["kill_win_rate"], -row["timeout_rate"], row["update"]))
            selected[str(seed)] = best

    write_csv(output / "checkpoint-evasive-metrics.csv", rows)
    (output / "selected-checkpoints.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True), encoding="utf-8")

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.6), sharex=True)
    colours = {85: "#0072B2", 185: "#D55E00", 285: "#009E73"}
    for seed in SEEDS:
        group = [row for row in rows if row["seed"] == seed]
        group.sort(key=lambda row: row["steps"])
        if not group:
            continue
        x = [row["steps"] / 1.0e6 for row in group]
        axes[0].plot(x, [100 * row["kill_win_rate"] for row in group],
                     marker="o", label=f"seed {seed}", color=colours[seed])
        axes[1].plot(x, [100 * row["timeout_rate"] for row in group],
                     marker="o", label=f"seed {seed}", color=colours[seed])
    axes[0].set_ylabel("Kill win rate (%)")
    axes[1].set_ylabel("Timeout rate (%)")
    axes[1].set_xlabel("Training steps (million)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.set_ylim(0, 100)
        axis.legend(frameon=False, ncol=3)
    fig.suptitle("Historical checkpoints vs scripted evasive-fire opponent")
    fig.tight_layout()
    fig.savefig(figures / "checkpoint-evasive-kill-timeout.png", dpi=180)
    plt.close(fig)

    scripted_rows = []
    for path in sorted((root / "scripted-crossplay").glob("*/summary.json")):
        summary = load_json(path)
        scripted_rows.append({
            "policy_a": summary["policy_a"],
            "policy_b": summary["policy_b"],
            "episodes": summary["episodes"],
            "kill_win_rate": summary["kill_win_rate"],
            "kill_loss_rate": summary["kill_loss_rate"],
            "timeout_rate": summary["timeout_rate"],
            "mean_episode_steps": summary["mean_episode_steps"],
        })
    write_csv(output / "scripted-crossplay-metrics.csv", scripted_rows)

    selected_lines = "\n".join(
        f"- seed {seed}: update {item['update']} "
        f"({item['steps']:,} steps), kill win "
        f"{100 * item['kill_win_rate']:.1f}%, timeout "
        f"{100 * item['timeout_rate']:.1f}%"
        for seed, item in selected.items()) or "- 尚无完整数据。"
    script_lines = "\n".join(
        f"- {row['policy_a']} vs {row['policy_b']}: kill win "
        f"{100 * row['kill_win_rate']:.1f}%, timeout "
        f"{100 * row['timeout_rate']:.1f}%"
        for row in scripted_rows) or "- 尚无完整数据。"
    report = f"""# 超时零奖励回溯诊断

## 分析口径

主要胜率只统计实际淘汰终局；超时战术判决单独保留，不计为击毁胜利。检查点选择先最大化规避对手击毁率，再最小化超时率。该选择仅用于诊断，不能替代独立测试集验收。

## 各种子候选检查点

{selected_lines}

## 规则策略可行性

{script_lines}

## 图

`figures/checkpoint-evasive-kill-timeout.png` 展示历史检查点对规避发射对手的击毁率与超时率。用途是判断最终策略是否发生退化，以及问题更接近训练稳定性还是环境/武器不可达性。
"""
    (output / "analysis-report.md").write_text(report, encoding="utf-8")
    (root / "COMPLETED.json").write_text(json.dumps({
        "status": "completed",
        "checkpoint_rows": len(rows),
        "scripted_pairs": len(scripted_rows),
        "selected": selected,
    }, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "rows": len(rows), "scripted_pairs": len(scripted_rows),
        "selected": selected,
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
