#!/usr/bin/env python3
"""Analyze paired three-seed LOS-teacher behavior cloning results."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("runs/torch6dof_validation/v4-bc-los-intercept-3seed")
BASE = Path("runs/torch6dof_validation/v4-reward-ablation-3seed-1m/competitive")
SEEDS = (85, 185, 285)
OPPONENTS = ("pursuit", "hold", "evasive")
LABELS = {"pursuit": "Pursuit", "hold": "Hold", "evasive": "Evasive"}
COLORS = {"baseline": "#0072B2", "bc": "#D55E00"}
TCRIT = 4.302652729911275


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def describe(values):
    avg, sd = mean(values), stdev(values)
    half = TCRIT * sd / math.sqrt(len(values))
    return avg, sd, avg - half, avg + half


def pct(values):
    avg, sd, _, _ = describe(values)
    return f"{100 * avg:.1f}% ± {100 * sd:.1f}%"


def save(fig, figures, stem):
    fig.savefig(figures / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(figures / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    output = ROOT / "analysis-output"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    data = {}
    rows = []
    training = {}
    for seed in SEEDS:
        training[seed] = load(ROOT / f"seed{seed}" / "bc" / "train.json")
        for opponent in OPPONENTS:
            paths = {
                "baseline": BASE / f"seed{seed}" / f"eval-{opponent}.json",
                "bc": ROOT / f"seed{seed}" / "bc" / f"eval-{opponent}.json",
            }
            for cohort, path in paths.items():
                item = load(path)
                if item["episodes"] != 192 or item["seed"] != 10085:
                    raise RuntimeError(f"incomparable evaluation: {path}")
                data[(cohort, seed, opponent)] = item
                rows.append({
                    "cohort": cohort, "seed": seed, "opponent": opponent,
                    "win_rate": item["win_rate"],
                    "loss_rate": item["loss_rate"],
                    "timeout_rate": item["timeout_rate"],
                    "legal_episode_rate": item["legal_opportunity_episode_rate"],
                    "hit_per_launch": item["hit_per_launch"],
                    "crash_rate": item["learner_crash_rate"],
                    "rejected_fire_requests": item["rejected_fire_requests"],
                })
    with (output / "seed-level-results.csv").open(
            "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def vals(cohort, opponent, metric):
        return [data[(cohort, seed, opponent)][metric] for seed in SEEDS]

    delta_rows = []
    for opponent in OPPONENTS:
        for metric in ("win_rate", "timeout_rate", "legal_opportunity_episode_rate"):
            delta = [data[("bc", seed, opponent)][metric]
                     - data[("baseline", seed, opponent)][metric]
                     for seed in SEEDS]
            avg, sd, low, high = describe(delta)
            delta_rows.append({
                "opponent": opponent, "metric": metric,
                "mean_paired_delta": avg, "sd_paired_delta": sd,
                "ci95_low": low, "ci95_high": high,
                "paired_effect_dz": avg / sd if sd > 0 else None,
                "seed85_delta": delta[0], "seed185_delta": delta[1],
                "seed285_delta": delta[2],
            })
    with (output / "paired-deltas.csv").open(
            "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(delta_rows[0]))
        writer.writeheader()
        writer.writerows(delta_rows)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.22,
    })
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex=True, sharey="row")
    for col, opponent in enumerate(OPPONENTS):
        for row, (metric, ylabel) in enumerate(
                (("win_rate", "Win rate"), ("timeout_rate", "Timeout rate"))):
            ax = axes[row, col]
            for seed, marker in zip(SEEDS, ("o", "s", "^")):
                y = [data[(cohort, seed, opponent)][metric]
                     for cohort in ("baseline", "bc")]
                ax.plot([0, 1], y, color="#777777", alpha=0.55,
                        marker=marker, markersize=4, linewidth=1)
            for x, cohort in enumerate(("baseline", "bc")):
                y = vals(cohort, opponent, metric)
                ax.errorbar(x, mean(y), yerr=stdev(y), marker="D",
                            color=COLORS[cohort], capsize=3, linewidth=1.5,
                            markersize=5, zorder=5)
            ax.set_ylim(-0.04, 1.04)
            ax.set_xticks([0, 1], ["Base", "BC"])
            if row == 0:
                ax.set_title(LABELS[opponent])
            if col == 0:
                ax.set_ylabel(ylabel)
    fig.subplots_adjust(wspace=0.15, hspace=0.20)
    save(fig, figures, "figure-01-bc-vs-baseline")

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharex=True)
    accuracy_names = ((0, "Altitude"), (1, "Heading"), (None, "Exact action"))
    for ax, (head, title) in zip(axes, accuracy_names):
        matrix = []
        for seed in SEEDS:
            history = training[seed]["history"]
            y = np.asarray([
                item["exact_action_accuracy"] if head is None
                else item["head_accuracy"][head]
                for item in history])
            matrix.append(y)
            ax.plot(range(1, len(y) + 1), y, color="#777777",
                    alpha=0.45, linewidth=0.9)
        matrix = np.asarray(matrix)
        avg = matrix.mean(0)
        sd = matrix.std(0, ddof=1)
        x = np.arange(1, matrix.shape[1] + 1)
        ax.fill_between(x, avg - sd, avg + sd, color="#E69F00", alpha=0.2)
        ax.plot(x, avg, color="#D55E00", linewidth=1.8)
        ax.set_ylim(0.0, 1.02)
        ax.set_title(title)
        ax.set_xlabel("BC iteration")
    axes[0].set_ylabel("Teacher action accuracy")
    fig.subplots_adjust(wspace=0.18)
    save(fig, figures, "figure-02-bc-training-dynamics")

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
    for ax, opponent in zip(axes, OPPONENTS):
        for seed, marker in zip(SEEDS, ("o", "s", "^")):
            y = [data[(cohort, seed, opponent)]["legal_opportunity_episode_rate"]
                 for cohort in ("baseline", "bc")]
            ax.plot([0, 1], y, color="#777777", alpha=0.55,
                    marker=marker, markersize=4, linewidth=1)
        for x, cohort in enumerate(("baseline", "bc")):
            y = vals(cohort, opponent, "legal_opportunity_episode_rate")
            ax.errorbar(x, mean(y), yerr=stdev(y), marker="D",
                        color=COLORS[cohort], capsize=3, linewidth=1.5,
                        markersize=5, zorder=5)
        ax.set_ylim(-0.04, 1.04)
        ax.set_xticks([0, 1], ["Base", "BC"])
        ax.set_title(LABELS[opponent])
    axes[0].set_ylabel("Episodes with legal shot")
    fig.subplots_adjust(wspace=0.15)
    save(fig, figures, "figure-03-legal-shot-coverage")

    diagnostic = []
    diagnostic_paths = {
        "BC mixed (seed85)": ROOT / "seed85" / "bc-mixed",
        "Plain PPO after BC (seed85)": ROOT / "seed85" / "ppo-0p5m",
        "Aux PPO after BC (seed85)": ROOT / "seed85" / "ppo-aux-0p5m",
    }
    for label, directory in diagnostic_paths.items():
        if all((directory / f"eval-{opponent}.json").exists()
               for opponent in OPPONENTS):
            diagnostic.append((label, {
                opponent: load(directory / f"eval-{opponent}.json")
                for opponent in OPPONENTS}))

    report = [
        "# Torch V4 LOS 教师行为克隆三种子分析",
        "",
        "## 主要结论",
        "",
        "LOS 教师行为克隆可复现地打破规避对手 0% 胜率瓶颈，但不能直接替换竞争奖励基线。"
        "三个训练种子的规避胜率为 46%、43%、17%，平均提升约 35 个百分点；与此同时，"
        "追击和保持胜率分别下降约 30 和 34 个百分点。该结果支持将教师策略作为分层截获"
        "选项或辅助先验，而不支持把单一行为克隆网络作为最终通用策略。",
        "",
        "## 配对三种子结果",
        "",
        "| 配置 | 追击胜率 | 保持胜率 | 规避胜率 | 追击超时 | 保持超时 | 规避超时 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| 竞争奖励基线 | " + " | ".join([
            pct(vals("baseline", "pursuit", "win_rate")),
            pct(vals("baseline", "hold", "win_rate")),
            pct(vals("baseline", "evasive", "win_rate")),
            pct(vals("baseline", "pursuit", "timeout_rate")),
            pct(vals("baseline", "hold", "timeout_rate")),
            pct(vals("baseline", "evasive", "timeout_rate")),
        ]) + " |",
        "| 规避教师 BC | " + " | ".join([
            pct(vals("bc", "pursuit", "win_rate")),
            pct(vals("bc", "hold", "win_rate")),
            pct(vals("bc", "evasive", "win_rate")),
            pct(vals("bc", "pursuit", "timeout_rate")),
            pct(vals("bc", "hold", "timeout_rate")),
            pct(vals("bc", "evasive", "timeout_rate")),
        ]) + " |",
        "",
        "## 单种子诊断分支（不可作统计结论）",
        "",
        "| 分支 | 追击胜率 | 保持胜率 | 规避胜率 | 规避合法发射回合 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, items in diagnostic:
        report.append(
            f"| {label} | {100 * items['pursuit']['win_rate']:.1f}% | "
            f"{100 * items['hold']['win_rate']:.1f}% | "
            f"{100 * items['evasive']['win_rate']:.1f}% | "
            f"{100 * items['evasive']['legal_opportunity_episode_rate']:.1f}% |")
    report += [
        "",
        "普通 PPO 将 seed85 的规避胜率从 46% 降为 0%；教师辅助 PPO 仅保留到 6%；"
        "三类对手混合教师 BC 的规避胜率为 12%。这些均为单种子工程诊断，只能用于否决"
        "当前实现，不能估计方法总体性能。",
        "",
        "## 图形解释",
        "",
        "### Figure 1",
        "**目的：**检验 BC 是否在配对种子上改变固定对手结果。**观察：**规避胜率和超时率"
        "均明显改善，但追击、保持胜率同步下降。**解释：**教师先验学习的是专门截获模式，"
        "不是通用空战策略。**决策：**不替换基线，转为显式分层选项。",
        "",
        "### Figure 2",
        "**目的：**验证教师动作是否被网络稳定学习。**观察：**速度和火控头已饱和，航向头"
        "准确率仍有限且跨种子存在差异。**解释：**关键截获动作是多模态映射，单一小网络"
        "存在策略平均。**决策：**下一实现应分离通用策略与截获选项。",
        "",
        "### Figure 3",
        "**目的：**判断规避改善是否来自更多合法发射几何。**观察：**BC 将规避合法发射回合"
        "从约 7% 提高到约 56%。**解释：**收益机制是截获几何建立，不是放宽武器规则。"
        "**决策：**继续冻结导弹和发射包线。",
        "",
        "## Claim Candidates",
        "",
        "- Claim:",
        "  - Source evidence: 三个配对训练种子的固定对手评估。",
        "  - Allowed wording: LOS 教师行为克隆在本筛选中提高了规避截获能力，但牺牲了简单对手能力。",
        "  - Forbidden stronger wording: 行为克隆已经得到通用最优策略，或已经解决规避空战。",
        "  - Uncertainty: n=3，且仅 196,608 个教师样本；95% 区间很宽。",
        "  - Next check: 实现具有进入/退出门控的截获选项，并进行三种子配对验收。",
        "  - Decision: keep",
    ]
    (output / "analysis-report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    stats = [
        "# 统计附录", "",
        "训练种子是独立单位，n=3，基线与 BC 按种子配对。192 个评估回合只用于估计每个"
        "种子策略的性能，不作为方法级独立样本。n=3 时正态性检验和显著性检验没有可靠"
        "分辨率，因此不报告 p 值；报告均值、样本标准差、探索性 t 区间、逐种子差值以及"
        "配对标准化效应 dz。区间与效应量见 `paired-deltas.csv`，只能作工程筛选解释。", "",
        "单种子普通 PPO、辅助 PPO和混合教师 BC 被隔离为诊断结果，没有与三种子结果混合。",
    ]
    (output / "stats-appendix.md").write_text("\n".join(stats) + "\n", encoding="utf-8")

    catalog = [
        "# 图目录", "",
        "## Figure 1 — BC 与竞争奖励基线", "",
        "`figures/figure-01-bc-vs-baseline.pdf`：胜率和超时率的配对种子图。灰线连接相同"
        "训练种子，菱形和误差条为 n=3 的均值 ± 样本标准差。", "",
        "## Figure 2 — 行为克隆训练动态", "",
        "`figures/figure-02-bc-training-dynamics.pdf`：高度、航向与完整动作的教师一致率。"
        "粗线为三种子均值，阴影为 ±1 样本标准差，无平滑。", "",
        "## Figure 3 — 合法发射几何覆盖率", "",
        "`figures/figure-03-legal-shot-coverage.pdf`：每类对手中出现至少一次合法发射机会的"
        "回合比例；配对线和误差条定义同 Figure 1。",
    ]
    (output / "figure-catalog.md").write_text("\n".join(catalog) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
