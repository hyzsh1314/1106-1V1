#!/usr/bin/env python3
"""Analyze paired three-seed standard PPO and hierarchical PPO baselines."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np


STANDARD = Path(
    "runs/torch6dof_validation/v4-reward-ablation-3seed-1m/competitive")
HIERARCHICAL = Path(
    "runs/torch6dof_validation/v4-hierarchical-ppo-1m")
OUTPUT = HIERARCHICAL / "analysis-output"
SEEDS = (85, 185, 285)
OPPONENTS = ("hold", "pursuit", "evasive")
METHODS = ("standard_ppo", "hierarchical_ppo")
METHOD_LABEL = {"standard_ppo": "PPO", "hierarchical_ppo": "Hier. PPO"}
OPPONENT_LABEL = {"hold": "Hold", "pursuit": "Pursuit", "evasive": "Evasive"}
COLORS = {"standard_ppo": "#0072B2", "hierarchical_ppo": "#D55E00"}
TCRIT_N3 = 4.302652729911275


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def describe(values):
    avg = mean(values)
    sd = stdev(values)
    half = TCRIT_N3 * sd / math.sqrt(len(values))
    return avg, sd, avg - half, avg + half


def pct(values):
    avg, sd, _, _ = describe(values)
    return f"{100 * avg:.1f}% ± {100 * sd:.1f}%"


def number(values):
    avg, sd, _, _ = describe(values)
    return f"{avg:.1f} ± {sd:.1f}"


def save(fig, directory: Path, stem: str):
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    figures = OUTPUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    eval_data = {}
    train_data = {}
    seed_rows = []

    for seed in SEEDS:
        standard_train = load(STANDARD / f"seed{seed}" / "train.json")
        hierarchical_train = load(HIERARCHICAL / f"seed{seed}" / "train.json")
        for method, item in (
                ("standard_ppo", standard_train),
                ("hierarchical_ppo", hierarchical_train)):
            if int(item["total_environment_decisions"]) != 1_007_616:
                raise RuntimeError(f"unexpected training budget: {method}, {seed}")
            if item["training_opponents"] != "staged":
                raise RuntimeError(f"opponent curriculum mismatch: {method}, {seed}")
            if item["reward_variant"] != "competitive":
                raise RuntimeError(f"reward mismatch: {method}, {seed}")
            train_data[(method, seed)] = item
        for opponent in OPPONENTS:
            for method, root in (
                    ("standard_ppo", STANDARD / f"seed{seed}"),
                    ("hierarchical_ppo", HIERARCHICAL / f"seed{seed}")):
                item = load(root / f"eval-{opponent}.json")
                if int(item["episodes"]) != 192 or int(item["seed"]) != 10085:
                    raise RuntimeError(
                        f"incomparable evaluation: {method}, {seed}, {opponent}")
                eval_data[(method, seed, opponent)] = item
                seed_rows.append({
                    "method": method,
                    "seed": seed,
                    "opponent": opponent,
                    "win_rate": item["win_rate"],
                    "timeout_rate": item["timeout_rate"],
                    "legal_opportunity_episode_rate": item[
                        "legal_opportunity_episode_rate"],
                    "launch_episode_rate": item["launch_episode_rate"],
                    "accepted_per_request": item["accepted_per_request"],
                    "hit_per_launch": item["hit_per_launch"],
                    "learner_crash_rate": item["learner_crash_rate"],
                    "base_option_fraction": item.get("base_option_fraction"),
                    "intercept_option_fraction": item.get(
                        "intercept_option_fraction"),
                    "training_throughput": item and train_data[
                        (method, seed)]["environment_decisions_per_s"],
                })

    with (OUTPUT / "seed-level-results.csv").open(
            "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)

    def values(method, opponent, metric):
        return [eval_data[(method, seed, opponent)][metric] for seed in SEEDS]

    metrics = (
        "win_rate", "timeout_rate", "legal_opportunity_episode_rate",
        "launch_episode_rate", "accepted_per_request", "hit_per_launch")
    summary_rows = []
    delta_rows = []
    for opponent in OPPONENTS:
        for metric in metrics:
            for method in METHODS:
                avg, sd, low, high = describe(values(method, opponent, metric))
                summary_rows.append({
                    "method": method, "opponent": opponent, "metric": metric,
                    "mean": avg, "sample_sd": sd,
                    "exploratory_ci95_low": low,
                    "exploratory_ci95_high": high, "n_seeds": 3,
                })
            delta = [
                eval_data[("hierarchical_ppo", seed, opponent)][metric]
                - eval_data[("standard_ppo", seed, opponent)][metric]
                for seed in SEEDS]
            avg, sd, low, high = describe(delta)
            delta_rows.append({
                "opponent": opponent, "metric": metric,
                "mean_paired_delta": avg, "sd_paired_delta": sd,
                "exploratory_ci95_low": low,
                "exploratory_ci95_high": high,
                "paired_effect_dz": avg / sd if sd > 0 else None,
                "seed85_delta": delta[0], "seed185_delta": delta[1],
                "seed285_delta": delta[2],
            })
    for method in METHODS:
        throughput = [train_data[(method, seed)][
            "environment_decisions_per_s"] for seed in SEEDS]
        avg, sd, low, high = describe(throughput)
        summary_rows.append({
            "method": method, "opponent": "all", "metric": "throughput",
            "mean": avg, "sample_sd": sd,
            "exploratory_ci95_low": low,
            "exploratory_ci95_high": high, "n_seeds": 3,
        })
    throughput_delta = [
        train_data[("hierarchical_ppo", seed)]["environment_decisions_per_s"]
        - train_data[("standard_ppo", seed)]["environment_decisions_per_s"]
        for seed in SEEDS]
    avg, sd, low, high = describe(throughput_delta)
    delta_rows.append({
        "opponent": "all", "metric": "throughput",
        "mean_paired_delta": avg, "sd_paired_delta": sd,
        "exploratory_ci95_low": low, "exploratory_ci95_high": high,
        "paired_effect_dz": avg / sd if sd > 0 else None,
        "seed85_delta": throughput_delta[0],
        "seed185_delta": throughput_delta[1],
        "seed285_delta": throughput_delta[2],
    })
    for filename, rows in (
            ("summary.csv", summary_rows),
            ("paired-deltas.csv", delta_rows)):
        with (OUTPUT / filename).open(
                "w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.22,
    })
    figure_metrics = (
        ("win_rate", "Win rate"),
        ("timeout_rate", "Timeout rate"),
        ("legal_opportunity_episode_rate", "Episodes with legal shot"),
        ("hit_per_launch", "Hits per launch"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    x = np.arange(len(OPPONENTS))
    offsets = {"standard_ppo": -0.12, "hierarchical_ppo": 0.12}
    for ax, (metric, ylabel) in zip(axes.flat, figure_metrics):
        for method in METHODS:
            means = [mean(values(method, opponent, metric))
                     for opponent in OPPONENTS]
            sds = [stdev(values(method, opponent, metric))
                   for opponent in OPPONENTS]
            ax.errorbar(
                x + offsets[method], means, yerr=sds, fmt="o",
                color=COLORS[method], label=METHOD_LABEL[method],
                capsize=3, linewidth=1.5, markersize=5)
            for opponent_index, opponent in enumerate(OPPONENTS):
                for seed in SEEDS:
                    value = eval_data[(method, seed, opponent)][metric]
                    ax.scatter(
                        opponent_index + offsets[method], value,
                        color=COLORS[method], alpha=0.4, s=13, zorder=2)
        ax.set_xticks(x, [OPPONENT_LABEL[o] for o in OPPONENTS])
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.04, 1.04)
    axes[0, 0].legend(frameon=False, loc="upper right")
    fig.subplots_adjust(wspace=0.25, hspace=0.28)
    save(fig, figures, "figure-01-ppo-vs-hierarchical")

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for seed, marker in zip(SEEDS, ("o", "s", "^")):
        history = train_data[("hierarchical_ppo", seed)]["history"]
        x_update = [item["update"] for item in history]
        y_fraction = [item["intercept_option_fraction"] for item in history]
        ax.plot(x_update, y_fraction, marker=marker, markevery=5,
                linewidth=1.4, markersize=4, label=f"Seed {seed}")
    ax.axvline(14.5, color="#777777", linestyle="--", linewidth=1)
    ax.axvline(28.5, color="#777777", linestyle="--", linewidth=1)
    ax.text(7.5, 1.02, "Pursuit", ha="center", va="bottom")
    ax.text(21.5, 1.02, "+ Hold", ha="center", va="bottom")
    ax.text(35.0, 1.02, "+ Evasive", ha="center", va="bottom")
    ax.set_xlim(1, 41)
    ax.set_ylim(0, 1)
    ax.set_xlabel("PPO update")
    ax.set_ylabel("Intercept option fraction")
    ax.legend(frameon=False, ncol=3, loc="lower center")
    save(fig, figures, "figure-02-option-selection-dynamics")

    fig, ax = plt.subplots(figsize=(4.3, 3.0))
    for seed, marker in zip(SEEDS, ("o", "s", "^")):
        y = [train_data[(method, seed)]["environment_decisions_per_s"]
             for method in METHODS]
        ax.plot([0, 1], y, color="#777777", alpha=0.6,
                marker=marker, linewidth=1)
    for index, method in enumerate(METHODS):
        y = [train_data[(method, seed)]["environment_decisions_per_s"]
             for seed in SEEDS]
        ax.errorbar(index, mean(y), yerr=stdev(y), marker="D",
                    color=COLORS[method], capsize=3, linewidth=1.5,
                    markersize=5, zorder=5)
    ax.set_xticks([0, 1], [METHOD_LABEL[m] for m in METHODS])
    ax.set_ylabel("Environment decisions / s")
    ax.set_ylim(bottom=0)
    save(fig, figures, "figure-03-training-throughput")

    standard_tp = [train_data[("standard_ppo", seed)][
        "environment_decisions_per_s"] for seed in SEEDS]
    hierarchical_tp = [train_data[("hierarchical_ppo", seed)][
        "environment_decisions_per_s"] for seed in SEEDS]
    report = [
        "# 标准 PPO 与分层 PPO：三随机种子基准分析", "",
        "## 比较问题与实验单位", "",
        "比较相同 Torch V4 环境、竞争奖励、分阶段对手课程、96 并行环境、"
        "1,007,616 环境决策和固定评估种子 10085 下的标准 PPO 与选项门控分层 PPO。"
        "独立分析单位为训练种子（85、185、285，n=3）；每个种子对每类对手的192局"
        "评估不是方法级独立重复。", "",
        "## 三种子结果（均值 ± 样本标准差）", "",
        "| 对手 | 方法 | 胜率 | 超时率 | 合法发射机会回合率 | 命中/发射 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for opponent in OPPONENTS:
        for method in METHODS:
            report.append(
                f"| {OPPONENT_LABEL[opponent]} | {METHOD_LABEL[method]} | "
                f"{pct(values(method, opponent, 'win_rate'))} | "
                f"{pct(values(method, opponent, 'timeout_rate'))} | "
                f"{pct(values(method, opponent, 'legal_opportunity_episode_rate'))} | "
                f"{pct(values(method, opponent, 'hit_per_launch'))} |")
    report += [
        "", "## 训练吞吐", "",
        f"标准 PPO：{number(standard_tp)} 环境决策/秒；分层 PPO："
        f"{number(hierarchical_tp)} 环境决策/秒。该吞吐只统计当前在线训练阶段；"
        "分层方法依赖的两个下层技能预训练成本没有计入，因此不能据此声称其总训练成本更低。",
        "", "## 主要观察", "",
        "分层 PPO 的平均规避胜率高于标准 PPO，但提升完全由 seed185 和 seed285 驱动；"
        "seed85 的门控塌缩到标准技能后规避胜率仍为0%。因此当前证据支持“分层门控有潜力"
        "突破规避瓶颈”，不支持“分层 PPO 已稳定优于 PPO”。", "",
        "保持与追击场景中，分层 PPO 的均值不一定全面优于标准 PPO，且种子方差较大。"
        "这说明当前门控学习同时继承了两个下层技能的种子差异，并存在课程早期选项塌缩问题。",
        "", "## 合法发射率口径", "",
        "主表采用“至少出现一次合法发射机会的回合比例”，因为它反映策略能否建立有效发射"
        "几何。`seed-level-results.csv` 同时保留了发射回合率和发射请求接受率；后者受动作掩码"
        "保护，通常接近100%，区分算法的价值较低。", "",
        "## Claim Candidates", "",
        "- Claim:",
        "  - Source evidence: 三个配对训练种子、每种子每类固定对手192局。",
        "  - Allowed wording: 选项门控分层 PPO 在部分种子上利用拦截技能突破了标准 PPO 的规避对手0%胜率瓶颈，但稳定性仍不足。",
        "  - Forbidden stronger wording: 分层 PPO 已显著或稳定优于标准 PPO。",
        "  - Uncertainty: n=3，种子间门控行为差异大；分层方法还使用了额外行为克隆先验。",
        "  - Next check: 在冻结环境下只调整上层算法，加入选项使用约束或平衡课程，并重新进行三种子验收。",
        "  - Decision: weaken",
    ]
    (OUTPUT / "analysis-report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8")

    stats = [
        "# 统计附录", "",
        "## 设计", "",
        "标准 PPO 与分层 PPO 按训练种子配对，n=3。指标方向：胜率、合法发射机会回合率、"
        "命中/发射和吞吐越高越好；超时率越低越好。全部评估使用相同初始状态随机种子10085。",
        "", "## 推断边界", "",
        "n=3 时 Shapiro–Wilk 等正态性检查与配对显著性检验没有有用分辨率；Wilcoxon"
        "精确检验的最小双侧p值也无法达到0.05。因此不报告显著性p值。`summary.csv` 报告"
        "均值、样本标准差和探索性t区间；`paired-deltas.csv` 报告逐种子差值、均值差、"
        "样本标准差和配对效应量dz。由于样本极少，区间与效应量均可能非常不稳定。", "",
        "## 重要限制", "",
        "分层 PPO 嵌入了对应种子的标准 PPO 和行为克隆策略，下层并非重新独立训练。"
        "因此比较回答的是“增加一个PPO门控和拦截技能是否改善固定策略组合”，而不是"
        "相同总数据与总计算预算下两个端到端算法的纯粹比较。",
    ]
    (OUTPUT / "stats-appendix.md").write_text(
        "\n".join(stats) + "\n", encoding="utf-8")

    catalog = [
        "# 图目录", "",
        "## Figure 1 — 固定对手性能比较", "",
        "`figures/figure-01-ppo-vs-hierarchical.pdf`。目的：比较两种算法在三类固定对手"
        "上的胜率、超时、合法发射几何和命中效率。点为单个训练种子，圆点和误差条为"
        "n=3均值±样本标准差。应注意规避改善伴随较大种子方差，不能只看均值。", "",
        "## Figure 2 — 分层选项选择动态", "",
        "`figures/figure-02-option-selection-dynamics.pdf`。目的：解释种子方差来自何处。"
        "曲线为每轮拦截选项占比，无平滑；虚线为课程阶段边界。seed85发生标准技能塌缩，"
        "seed185偏向拦截技能，seed285保持相对均衡。", "",
        "## Figure 3 — 在线训练吞吐", "",
        "`figures/figure-03-training-throughput.pdf`。目的：比较相同环境预算下在线训练速度。"
        "灰线连接相同种子，菱形和误差条为均值±样本标准差。该图不包含下层策略预训练成本。",
    ]
    (OUTPUT / "figure-catalog.md").write_text(
        "\n".join(catalog) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
