"""Aggregate the preregistered R0-R3 reward ablation after all runs finish."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VARIANTS = (
    "R0_terminal", "R1_attack_pbrs", "R2_threat_pbrs", "R3_constraints")
SEEDS = (85, 185, 285)
OPPONENTS = ("pursuit_fire", "hold")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean_sd(values):
    values = np.asarray([value for value in values if value is not None],
                        dtype=np.float64)
    if not len(values):
        return None, None
    if len(values) == 1:
        return float(values[0]), None
    return float(np.mean(values)), float(np.std(values, ddof=1))


def main():
    args = parse_args()
    rows = []
    curves = {}
    missing = []
    for variant in VARIANTS:
        for seed in SEEDS:
            run_dir = args.root_dir / f"{variant}_seed{seed}_1m"
            completion = run_dir / "COMPLETED.json"
            metrics_path = run_dir / "metrics.jsonl"
            if not completion.exists() or not metrics_path.exists():
                missing.extend(path for path in (completion, metrics_path)
                               if not path.exists())
                continue
            curves[(variant, seed)] = [
                json.loads(line) for line in metrics_path.read_text(
                    encoding="utf-8").splitlines() if line.strip()]
            train = load_json(completion)
            for opponent in OPPONENTS:
                summary_path = run_dir / f"evaluation-{opponent}" / "summary.json"
                if not summary_path.exists():
                    missing.append(summary_path)
                    continue
                summary = load_json(summary_path)
                rows.append({
                    "variant": variant,
                    "seed": seed,
                    "opponent": opponent,
                    "episodes": summary["episodes"],
                    "win_rate": summary["win_rate"],
                    "draw_rate": summary["draw_rate"],
                    "loss_rate": summary["loss_rate"],
                    "timeout_rate": summary["timeout_rate"],
                    "launch_acceptance_rate": summary[
                        "legal_launch_acceptance_rate"],
                    "rejected_requests": summary["learner_rejected_requests"],
                    "opportunity_exploitation_rate": summary[
                        "legal_opportunity_exploitation_rate"],
                    "mean_episode_steps": summary["mean_episode_steps"],
                    "train_steps": train["total_steps"],
                    "train_timeout_rate": train["timeout_rate"],
                    "train_launch_acceptance_rate": train[
                        "legal_launch_acceptance_rate"],
                    "train_rejected_requests": train[
                        "total_rejected_fire_requests"],
                    "train_throughput_steps_s": train[
                        "session_throughput_steps_s"],
                })
    if missing:
        raise RuntimeError("missing ablation artifacts:\n" +
                           "\n".join(str(path) for path in missing))

    output = args.root_dir / "analysis-output"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    with (output / "per-run-metrics.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aggregates = []
    for variant in VARIANTS:
        for opponent in OPPONENTS:
            group = [row for row in rows if row["variant"] == variant and
                     row["opponent"] == opponent]
            item = {"variant": variant, "opponent": opponent, "n_seeds": 3}
            for metric in ("win_rate", "timeout_rate",
                           "launch_acceptance_rate",
                           "opportunity_exploitation_rate"):
                values = [row[metric] for row in group]
                item[f"{metric}_mean"], item[f"{metric}_sd"] = mean_sd(values)
            aggregates.append(item)
    (output / "aggregate-metrics.json").write_text(
        json.dumps(aggregates, indent=2, sort_keys=True), encoding="utf-8")

    colors = ("#6B7280", "#2563EB", "#059669", "#D97706")
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.7), constrained_layout=True)
    specs = (("pursuit_fire", "win_rate", "Win rate vs pursuit-fire"),
             ("pursuit_fire", "timeout_rate", "Timeout rate vs pursuit-fire"),
             ("hold", "win_rate", "Win rate vs hold"))
    x = np.arange(len(VARIANTS))
    for axis, (opponent, metric, title) in zip(axes, specs):
        for index, (variant, color) in enumerate(zip(VARIANTS, colors)):
            vals = [row[metric] for row in rows if row["variant"] == variant
                    and row["opponent"] == opponent]
            axis.scatter(np.full(len(vals), index), vals, color=color,
                         edgecolor="white", zorder=3)
            axis.plot(index, np.mean(vals), marker="_", markersize=18,
                      markeredgewidth=3, color="black", zorder=4)
        axis.set_title(title)
        axis.set_xticks(x, ("R0", "R1", "R2", "R3"))
        axis.set_ylim(-0.03, 1.03)
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(figures / "acceptance_metrics.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True,
                             constrained_layout=True)
    for axis, variant, color in zip(axes.ravel(), VARIANTS, colors):
        for seed in SEEDS:
            curve = curves[(variant, seed)]
            axis.plot([point["total_steps"] for point in curve],
                      [point["reward"] for point in curve],
                      alpha=0.7, linewidth=1.2, label=f"seed {seed}")
        axis.axhline(0, color="black", linewidth=0.7, alpha=0.5)
        axis.set_title(variant)
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.supxlabel("Environment decisions")
    fig.supylabel("Mean rollout reward")
    fig.savefig(figures / "reward_learning_curves.png", dpi=220)
    plt.close(fig)

    component_summary = {}
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4), sharex=True,
                             constrained_layout=True)
    for axis, variant in zip(axes.ravel(), VARIANTS):
        names = sorted({name for seed in SEEDS
                        for point in curves[(variant, seed)]
                        for name in point["mean_reward_components"]})
        component_summary[variant] = {}
        steps = [point["total_steps"] for point in curves[(variant, SEEDS[0])]]
        for name in names:
            seed_curves = np.asarray([
                [point["mean_reward_components"].get(name, 0.0)
                 for point in curves[(variant, seed)]]
                for seed in SEEDS], dtype=np.float64)
            axis.plot(steps, np.mean(seed_curves, axis=0), label=name)
            component_summary[variant][name] = float(
                np.mean(np.abs(seed_curves)))
        axis.axhline(0, color="black", linewidth=0.7, alpha=0.5)
        axis.set_title(variant)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=7)
    fig.supxlabel("Environment decisions")
    fig.supylabel("Mean reward component")
    fig.savefig(figures / "reward_component_curves.png", dpi=220)
    plt.close(fig)
    (output / "reward-component-summary.json").write_text(
        json.dumps(component_summary, indent=2, sort_keys=True),
        encoding="utf-8")

    pursuit = {item["variant"]: item for item in aggregates
               if item["opponent"] == "pursuit_fire"}
    hold = {item["variant"]: item for item in aggregates
            if item["opponent"] == "hold"}
    scores = {
        variant: (pursuit[variant]["win_rate_mean"] -
                  pursuit[variant]["timeout_rate_mean"] +
                  0.25 * hold[variant]["win_rate_mean"])
        for variant in VARIANTS
    }
    best = max(scores, key=scores.get)
    hard_weapon_pass = all(
        row["train_rejected_requests"] == 0 and
        row["train_launch_acceptance_rate"] == 1.0 for row in rows)
    report = [
        "# V4 奖励消融分析报告", "",
        "## 结论", "",
        f"12 个独立训练（4 个奖励变体 × 3 个随机种子）均完成。按预注册综合分数，当前候选为 **{best}**。",
        f"武器硬验收：**{'通过' if hard_weapon_pass else '失败'}**；所有固定对手评估中的非法请求数必须为 0，合法发射接受率必须为 100%。",
        "由于每组仅 3 个种子，本阶段只用于筛选奖励，不据此作算法显著性结论。", "",
        "## 变体", "",
        "- R0：仅终局 ±1/0。",
        "- R1：R0 + 进攻态势 PBRS（0.1）。",
        "- R2：R0 + 威胁条件 PBRS（0.1）。",
        "- R3：R2 + 小幅高度越界和发射成本。", "",
        "## 固定对手结果（均值 ± 跨种子标准差）", "",
        "|变体|追击对手胜率|追击对手超时率|直飞对手胜率|综合分数|",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        p, h = pursuit[variant], hold[variant]
        report.append(
            f"|{variant}|{p['win_rate_mean']:.3f} ± {p['win_rate_sd']:.3f}|"
            f"{p['timeout_rate_mean']:.3f} ± {p['timeout_rate_sd']:.3f}|"
            f"{h['win_rate_mean']:.3f} ± {h['win_rate_sd']:.3f}|"
            f"{scores[variant]:.3f}|")
    report.extend(["", "## 判定规则", "",
                   "候选奖励还需进入 500 万步复验；若追击对手超时率高于 20%、任一武器硬验收失败，或跨种子结果明显退化，则不进入正式基准。",
                   "", "## 奖励分量审计", "",
                   "逐更新记录 TeamOutcome、PBRS、高度约束和发射成本的平均贡献；绝对均值见 `reward-component-summary.json`，曲线见 `reward_component_curves.png`。该审计用于识别塑形项压过终局目标、符号反转或尖峰。", "",
                   "## 局限", "",
                   "固定脚本对手不代表完整策略分布；同一套低层控制器可能限制所有变体。本报告的统计单位是随机种子，而不是回合，避免把并行环境回合误当独立重复。"])
    (output / "analysis-report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8")
    (output / "stats-appendix.md").write_text(
        "# 统计附录\n\n每个变体 n=3 个独立随机种子。报告均值与样本标准差；不对 300 个评估回合做伪重复显著性检验。所有比较保持场景、训练步数、PPO 参数和固定对手一致。\n",
        encoding="utf-8")
    (output / "figure-catalog.md").write_text(
        "# 图目录\n\n- `acceptance_metrics.png`：固定对手胜率与超时率，圆点为随机种子，横线为均值。\n- `reward_learning_curves.png`：各奖励变体三随机种子的训练奖励轨迹。\n- `reward_component_curves.png`：终局、PBRS 与约束项的逐更新平均贡献。\n",
        encoding="utf-8")
    completion = {
        "status": "completed", "runs": 12,
        "evaluation_episodes": sum(row["episodes"] for row in rows),
        "selected_candidate": best, "weapon_acceptance_pass": hard_weapon_pass,
        "scores": scores,
    }
    (args.root_dir / "COMPLETED.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(completion, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
