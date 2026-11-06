"""Analyze the registered three-seed R1 five-million-step validation."""

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
CHECKPOINT_UPDATES = tuple(range(20, 201, 20))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path):
    if not path.exists():
        raise RuntimeError(f"missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, base):
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    output = args.root_dir / "analysis-output"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    audit = load_json(args.root_dir / "initial-state-audit" / "summary.json")
    if audit["status"] != "PASS":
        raise RuntimeError("initial-state audit did not pass")

    final_rows, periodic_rows, train_rows = [], [], []
    curves = {}
    for seed in SEEDS:
        run_dir = args.root_dir / f"seed{seed}_5m"
        completion = load_json(run_dir / "COMPLETED.json")
        raw_curve = [json.loads(line) for line in
                     (run_dir / "metrics.jsonl").read_text(
                         encoding="utf-8").splitlines() if line.strip()]
        # A supervised resume can append an update that was fully computed
        # but not yet checkpointed before interruption.  Keep the last record
        # for each update so monitoring logs remain append-only without
        # double-counting the statistical trajectory.
        by_update = {int(point["update"]): point for point in raw_curve}
        curve = [by_update[update] for update in sorted(by_update)]
        if len(curve) != 204:
            raise RuntimeError(
                f"seed {seed} has {len(curve)} unique updates "
                f"({len(raw_curve)} log rows), expected 204")
        curves[seed] = curve
        first = [point["timeouts"] / point["episodes_this_update"]
                 for point in curve[:41] if point["episodes_this_update"]]
        last = [point["timeouts"] / point["episodes_this_update"]
                for point in curve[-41:] if point["episodes_this_update"]]
        train_rows.append({
            "seed": seed, "steps": completion["total_steps"],
            "throughput_steps_s": completion["session_throughput_steps_s"],
            "episodes": completion["total_episodes"],
            "launches": completion["total_launches"],
            "fire_requests": completion["total_fire_requests"],
            "rejected_requests": completion["total_rejected_fire_requests"],
            "launch_acceptance_rate": completion[
                "legal_launch_acceptance_rate"],
            "early_timeout_fraction": float(np.mean(first)),
            "late_timeout_fraction": float(np.mean(last)),
            "late_minus_early_timeout": float(np.mean(last) - np.mean(first)),
            "termination_reasons_json": json.dumps(
                completion.get("total_termination_reasons", {}),
                sort_keys=True),
        })
        for opponent in OPPONENTS:
            summary = load_json(
                run_dir / "final-evaluation" / opponent / "summary.json")
            final_rows.append({
                "seed": seed, "opponent": opponent,
                "episodes": summary["episodes"],
                "win_rate": summary["win_rate"],
                "draw_rate": summary["draw_rate"],
                "loss_rate": summary["loss_rate"],
                "timeout_rate": summary["timeout_rate"],
                "mean_episode_steps": summary["mean_episode_steps"],
                "launch_acceptance_rate": summary[
                    "legal_launch_acceptance_rate"],
                "opportunity_exploitation_rate": summary[
                    "legal_opportunity_exploitation_rate"],
                "termination_reasons_json": json.dumps(
                    summary["termination_reason_counts"], sort_keys=True),
            })
        for update in CHECKPOINT_UPDATES:
            summary = load_json(
                run_dir / "periodic-evaluation" /
                f"checkpoint_{update:06d}" / "summary.json")
            periodic_rows.append({
                "seed": seed, "update": update,
                "steps": update * 96 * 256,
                "win_rate": summary["win_rate"],
                "draw_rate": summary["draw_rate"],
                "loss_rate": summary["loss_rate"],
                "timeout_rate": summary["timeout_rate"],
                "mean_episode_steps": summary["mean_episode_steps"],
            })
    write_csv(output / "final-fixed-opponent-metrics.csv", final_rows)
    write_csv(output / "periodic-pursuit-metrics.csv", periodic_rows)
    write_csv(output / "training-health.csv", train_rows)

    cross_rows = []
    matrix = np.zeros((len(SEEDS), len(SEEDS)), dtype=np.float64)
    for i, learner_seed in enumerate(SEEDS):
        for j, opponent_seed in enumerate(SEEDS):
            summary = load_json(
                args.root_dir / "crossplay" /
                f"seed{learner_seed}_vs_seed{opponent_seed}" / "summary.json")
            matrix[i, j] = summary["win_rate"]
            cross_rows.append({
                "learner_seed": learner_seed, "opponent_seed": opponent_seed,
                "episodes": summary["episodes"],
                "win_rate": summary["win_rate"],
                "draw_rate": summary["draw_rate"],
                "loss_rate": summary["loss_rate"],
                "timeout_rate": summary["timeout_rate"],
                "mean_episode_steps": summary["mean_episode_steps"],
            })
    write_csv(output / "crossplay-metrics.csv", cross_rows)

    def values(opponent, metric):
        return np.asarray([row[metric] for row in final_rows
                           if row["opponent"] == opponent], dtype=np.float64)

    pursuit = values("pursuit_fire", "win_rate")
    hold = values("hold", "win_rate")
    fixed_timeouts = np.asarray([row["timeout_rate"] for row in final_rows])
    hard_weapon_pass = all(
        row["rejected_requests"] == 0 and
        row["launch_acceptance_rate"] == 1.0 for row in train_rows)
    timeout_trend_pass = all(
        row["late_minus_early_timeout"] <= 0.10 for row in train_rows)
    gates = {
        "initial_state_audit": audit["status"] == "PASS",
        "pursuit_mean_at_least_0_90": float(pursuit.mean()) >= 0.90,
        "pursuit_worst_seed_at_least_0_80": float(pursuit.min()) >= 0.80,
        "hold_mean_at_least_0_95": float(hold.mean()) >= 0.95,
        "fixed_timeout_at_most_0_05": float(fixed_timeouts.max()) <= 0.05,
        "weapon_acceptance": hard_weapon_pass,
        "selfplay_timeout_not_worsened_by_0_10": timeout_trend_pass,
    }
    acceptance_pass = all(gates.values())

    # Longitudinal fixed-opponent validation.
    fig, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for seed in SEEDS:
        group = [row for row in periodic_rows if row["seed"] == seed]
        axis.plot(np.asarray([row["steps"] for row in group]) / 1.0e6,
                  [row["win_rate"] for row in group], marker="o",
                  linewidth=1.3, markersize=3.5, alpha=0.75,
                  label=f"seed {seed}")
    grouped = [[row["win_rate"] for row in periodic_rows
                if row["update"] == update] for update in CHECKPOINT_UPDATES]
    means = np.asarray([np.mean(group) for group in grouped])
    sds = np.asarray([np.std(group, ddof=1) for group in grouped])
    steps = np.asarray(CHECKPOINT_UPDATES) * 96 * 256 / 1.0e6
    axis.plot(steps, means, color="black", linewidth=2.2, label="mean")
    axis.fill_between(steps, means - sds, means + sds, color="black", alpha=0.1)
    axis.set(xlabel="Environment decisions (million)", ylabel="Win rate",
             ylim=(0, 1.03))
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    save_figure(fig, figures / "figure-01-periodic-pursuit-win-rate")

    fig, axis = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    x = np.arange(len(OPPONENTS))
    for seed in SEEDS:
        group = [row for row in final_rows if row["seed"] == seed]
        group.sort(key=lambda row: OPPONENTS.index(row["opponent"]))
        axis.plot(x, [row["win_rate"] for row in group], color="#B8BDC6",
                  linewidth=1.0)
    for index, opponent in enumerate(OPPONENTS):
        metric = values(opponent, "win_rate")
        axis.scatter(np.full(3, index), metric, s=45, color="#0072B2",
                     edgecolor="white", zorder=3)
        axis.plot([index - 0.16, index + 0.16], [metric.mean()] * 2,
                  color="black", linewidth=2.5)
    axis.set_xticks(x, ("Pursuit", "Hold", "Evasive", "Standoff"))
    axis.set(ylabel="Final checkpoint win rate", ylim=(0, 1.03))
    axis.grid(axis="y", alpha=0.25)
    save_figure(fig, figures / "figure-02-final-fixed-opponents")

    fig, axis = plt.subplots(figsize=(5.1, 4.4), constrained_layout=True)
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    for i in range(len(SEEDS)):
        for j in range(len(SEEDS)):
            axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                      color="white" if matrix[i, j] > 0.55 else "black")
    axis.set_xticks(range(3), [str(seed) for seed in SEEDS])
    axis.set_yticks(range(3), [str(seed) for seed in SEEDS])
    axis.set_xlabel("Opponent seed")
    axis.set_ylabel("Learner seed")
    fig.colorbar(image, ax=axis, label="Learner win rate")
    save_figure(fig, figures / "figure-03-crossplay-matrix")

    report = f"""# R1 500万步验证报告

## 状态

3 个独立种子均完成 5,013,504 步；初始状态审计覆盖 {audit['samples']} 次重置并通过。总体验收：**{'通过' if acceptance_pass else '未通过'}**。

## 固定对手

- 追击发射：{pursuit.mean():.3f} ± {pursuit.std(ddof=1):.3f}，最差种子 {pursuit.min():.3f}。
- 直飞：{hold.mean():.3f} ± {hold.std(ddof=1):.3f}。
- 规避发射：{values('evasive_fire', 'win_rate').mean():.3f} ± {values('evasive_fire', 'win_rate').std(ddof=1):.3f}。
- 保持距离发射：{values('standoff_fire', 'win_rate').mean():.3f} ± {values('standoff_fire', 'win_rate').std(ddof=1):.3f}。
- 最大固定对手超时率：{fixed_timeouts.max():.3f}。

## 工程健康度

训练合法发射硬验收：{'通过' if hard_weapon_pass else '失败'}。自博弈后 20% 相对前 20% 的超时变化分别为：{', '.join(f"seed {row['seed']}: {row['late_minus_early_timeout']:+.3f}" for row in train_rows)}。

## 验收门

""" + "\n".join(
        f"- {'PASS' if passed else 'FAIL'} — {name}"
        for name, passed in gates.items()) + """

## 决策

若全部门槛通过，可进入 2000 万步单种子稳定性试验；否则应根据失败门槛定位问题，不直接启动 1 亿步。周期固定评估、最终四类固定对手以及跨种子交叉对战均保留原始 CSV 和检查点路径，可复核而不依赖汇总文字。
"""
    (output / "analysis-report.md").write_text(report, encoding="utf-8")
    (output / "figure-catalog.md").write_text(
        "# 图目录\n\n"
        "- `figure-01-periodic-pursuit-win-rate`: 每约49万步的追击对手胜率；线为种子，黑线/带为均值±SD。\n"
        "- `figure-02-final-fixed-opponents`: 最终检查点对四类固定对手的胜率。\n"
        "- `figure-03-crossplay-matrix`: 三个最终策略的配对交叉对战。\n",
        encoding="utf-8")
    completion = {
        "status": "completed", "acceptance_pass": acceptance_pass,
        "gates": gates, "seeds": list(SEEDS),
        "steps_per_seed": 204 * 96 * 256,
        "total_training_steps": 3 * 204 * 96 * 256,
        "pursuit_mean_win_rate": float(pursuit.mean()),
        "pursuit_worst_seed_win_rate": float(pursuit.min()),
        "hold_mean_win_rate": float(hold.mean()),
    }
    (args.root_dir / "COMPLETED.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(completion, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
