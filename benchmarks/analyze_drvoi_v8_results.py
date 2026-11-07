#!/usr/bin/env python3
"""Create an evidence-first V8 analysis bundle from existing artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OPPONENTS = ("hold", "pursuit", "evasive")
METRICS = (
    ("win_rate", "Win rate"),
    ("timeout_rate", "Timeout rate"),
    ("hit_per_launch", "Hit / accepted launch"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--v8-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def wilson(successes: int, total: int, z: float = 1.959963984540054):
    if total <= 0:
        return (math.nan, math.nan)
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    spread = z * math.sqrt(
        p * (1.0 - p) / total + z * z / (4.0 * total * total)
    ) / denominator
    return center - spread, center + spread


def metric_interval(report, metric):
    if metric == "win_rate":
        return wilson(int(report["wins"]), int(report["episodes"]))
    if metric == "timeout_rate":
        return wilson(
            round(float(report["timeout_rate"]) * int(report["episodes"])),
            int(report["episodes"]),
        )
    return wilson(int(report["hits"]), int(report["accepted_launches"]))


def save_figure(fig, stem: Path):
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def comparison_figure(baseline, v8, figures_dir):
    colors = ("#0072B2", "#E69F00")
    x = np.arange(len(OPPONENTS))
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    for axis, (metric, label) in zip(axes, METRICS):
        for offset, (name, reports, color) in enumerate((
            ("B0 RHPPO", baseline, colors[0]),
            ("V8 MODS-RHPPO", v8, colors[1]),
        )):
            values = np.array([reports[o][metric] for o in OPPONENTS])
            intervals = [metric_interval(reports[o], metric) for o in OPPONENTS]
            lower = np.maximum(
                0.0, values - np.array([item[0] for item in intervals]))
            upper = np.maximum(
                0.0, np.array([item[1] for item in intervals]) - values)
            positions = x + (offset - 0.5) * width
            axis.bar(positions, values, width, color=color, label=name,
                     edgecolor="black", linewidth=0.4)
            axis.errorbar(positions, values, yerr=np.vstack((lower, upper)),
                          fmt="none", ecolor="black", capsize=2, linewidth=0.8)
        axis.set_title(label)
        axis.set_xticks(x, [value.capitalize() for value in OPPONENTS], rotation=20)
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel("Proportion")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    save_figure(fig, figures_dir / "figure-01-fixed-opponent-comparison")


def dynamics_figure(history, figures_dir):
    steps = np.array([row["environment_decisions"] for row in history]) / 1e6
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.3), sharex=True)
    specs = (
        ("positive_voi_nonzero_fraction", "Positive VOI coverage", "#0072B2"),
        ("candidate_option_diversity_fraction", "Option diversity", "#E69F00"),
        ("q_fold_mean_absolute_gap", "Cross-fold Q gap", "#D55E00"),
        ("intrinsic_beta", "Intrinsic coefficient", "#009E73"),
    )
    for axis, (key, label, color) in zip(axes.flat, specs):
        axis.plot(steps, [row[key] for row in history], color=color,
                  linewidth=1.7)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25, linewidth=0.6)
    for axis in axes[-1]:
        axis.set_xlabel("Environment decisions (million)")
    fig.tight_layout()
    save_figure(fig, figures_dir / "figure-02-online-voi-dynamics")


def make_table(baseline, v8):
    lines = [
        "| Opponent | Metric | B0 | V8 | Delta (percentage points) |",
        "|---|---:|---:|---:|---:|",
    ]
    for opponent in OPPONENTS:
        for metric, label in METRICS:
            old = float(baseline[opponent][metric])
            new = float(v8[opponent][metric])
            lines.append(
                f"| {opponent} | {label} | {100*old:.2f}% | "
                f"{100*new:.2f}% | {100*(new-old):+.2f} |"
            )
    return "\n".join(lines)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    baseline = {
        opponent: read_json(args.baseline_dir / f"eval-{opponent}.json")
        for opponent in OPPONENTS
    }
    v8 = {
        opponent: read_json(args.v8_dir / f"eval-{opponent}.json")
        for opponent in OPPONENTS
    }
    train = read_json(args.v8_dir / "train.json")
    comparison_figure(baseline, v8, figures_dir)
    dynamics_figure(train["history"], figures_dir)

    table = make_table(baseline, v8)
    first = train["history"][0]
    last = train["history"][-1]
    report = f"""# DRVOI-RHPPO V8 严格结果分析

## 分析问题

比较单种子 V8 masked macro-option soft-VOI RHPPO 与其初始化检查点 B0 RHPPO，在相同评估种子和三类固定对手上的结果，并判断 V8 是否已经具备扩大正式实验的依据。

## 证据清单与可比性

- 训练单位：单个训练随机种子（seed 85），V8 共 {train['total_environment_decisions']:,} 个环境决策。
- 评估单位：每个检查点、每类对手 192 局；双方各承担红方和蓝方 96 局。
- 两个检查点使用相同评估种子 10085、相同环境和评估脚本。
- 缺少逐局配对记录和多训练种子，因此这里只能报告检查点级描述性差异与局级 Wilson 95% 区间，不能宣称算法层面的统计显著优越性。

## 精确数值摘要

{table}

## 关键发现

1. V8 对 pursuit 的胜率从 75.00% 提高到 83.85%（+8.85 个百分点），同时命中/发射从 41.85% 提高到 45.76%。这支持 V8 在容易形成交战机会的局面中提高了局部决策效率。
2. V8 对 hold 的胜率只提高 2.60 个百分点，命中/发射提高 6.52 个百分点，但超时率从 8.33% 增至 20.83%（+12.50 个百分点）。效率提升没有转化为可靠的终局能力。
3. V8 对 evasive 仍为 0 胜、11 负、181 平，94.27% 超时，与 B0 的胜负平计数完全一致。V8 没有解决规避对手覆盖问题。
4. 在线训练中，正 VOI 覆盖率由 {100*first['positive_voi_nonzero_fraction']:.1f}% 降至 {100*last['positive_voi_nonzero_fraction']:.1f}%，候选宏选项分歧由 {100*first['candidate_option_diversity_fraction']:.1f}% 降至 {100*last['candidate_option_diversity_fraction']:.1f}%；末轮跨折 Q 差距为 {last['q_fold_mean_absolute_gap']:.3f}。冻结离线 Q 对在线策略分布的适配性随训练下降。

## 机制解释与决策

V8 将“当前状态的信息价值”作为内在奖励，但奖励与本步真正执行的宏选项没有直接因果绑定。因此策略可能学会进入高 VOI 状态，却不一定执行能利用该信息的动作。冻结的两个 Q 折又随着在线策略漂移产生较大分歧，使这一奖励逐渐稀疏且不可靠。该机制与 hold 超时恶化、evasive 无改善的结果一致，但当前单种子证据只能支持这一诊断，不能证明唯一因果。

决策：停止把 V8 作为待扩种子的主算法；保留其宏选项集合、响应历史模型和跨拟合诊断，转入 V9“置信度门控的在线宏选项策略学习”。V9 不再把 VOI 直接加到奖励，而是让高层策略真正选择并执行宏选项，并用在线实际回报更新选项价值；离线 Q 只作为可退火的先验。

## Claim Candidates

- Claim:
  - Source evidence: pursuit 胜率 +8.85 个百分点、命中/发射 +3.91 个百分点。
  - Allowed wording: 单种子开发结果显示 V8 在 pursuit 对手上具有积极趋势。
  - Forbidden stronger wording: V8 显著优于 B0 或已证明泛化提升。
  - Uncertainty: 无多训练种子，且只有一种评估种子。
  - Next check: V9 与 B0/V8 三种子、固定对手和种群评估。
  - Decision: weaken

- Claim:
  - Source evidence: evasive 结果为 0/11/181，和 B0 完全相同；末轮选项分歧仅 {100*last['candidate_option_diversity_fraction']:.1f}%。
  - Allowed wording: V8 未解决当前规避对手上的超时问题。
  - Forbidden stronger wording: VOI 思路在所有空战任务中无效。
  - Uncertainty: 仅一个训练种子和一种规避脚本。
  - Next check: 在线选项策略和 no-progress 约束消融。
  - Decision: keep
"""
    (args.output_dir / "analysis-report.md").write_text(report, encoding="utf-8")

    appendix = f"""# DRVOI-RHPPO V8 统计附录

## 统计单位

主要算法比较单位应是独立训练随机种子。目前每种方法只有一个训练种子，因此不执行 t 检验、Wilcoxon 检验或算法层面显著性检验。192 个对局只用于刻画已训练检查点，不能代替训练种子重复。

## 区间估计

胜率、超时率和命中/发射率使用二项比例 Wilson 95% 区间。图中的误差条表示评估对局或发射事件的不确定性，不表示训练种子方差。没有逐局配对标识，因此不进行 McNemar 检验或配对 bootstrap。

## 效应量

当前报告的效应量为绝对比例差（百分点）：

{table}

## 限制

- 单训练种子，不能估计种子方差。
- B0 与 V8 不是独立随机初始化：V8 从 B0 初始化；这适合开发诊断，不构成完整的算法对照设计。
- 固定对手只覆盖三种脚本，不能代表种群博弈或未知对手泛化。
- 训练曲线为时间相关序列，各更新轮次不是独立样本，不能把 41 个更新当作 n=41 做显著性检验。
"""
    (args.output_dir / "stats-appendix.md").write_text(appendix, encoding="utf-8")

    catalog = """# DRVOI-RHPPO V8 图目录

## figure-01-fixed-opponent-comparison.pdf

- Purpose: 比较 B0 与 V8 在三类固定对手上的终局质量与武器效率。
- Data source: 六个 `eval-*.json`，每个柱 192 局；命中率分母为 accepted launches。
- Caption requirements: 误差条为局级或发射事件级 Wilson 95% 区间，不是种子方差；训练种子 n=1。
- Key observation: pursuit 改善，hold 超时恶化，evasive 未改善。
- Interpretation checklist: 不得把局级区间解释为算法跨种子显著性。

## figure-02-online-voi-dynamics.pdf

- Purpose: 检查冻结 VOI 模型在在线策略分布移动时是否保持有效覆盖。
- Data source: V8 `train.json` 的 41 次更新，不做平滑。
- Caption requirements: 横轴为累计环境决策；四条曲线均为每次 rollout 的统计量。
- Key observation: VOI 覆盖和选项分歧总体下降，Q 折差距保持较高。
- Interpretation checklist: 时间序列相关，不能把更新轮次当作独立重复；趋势用于机制诊断而非显著性声明。
"""
    (args.output_dir / "figure-catalog.md").write_text(catalog, encoding="utf-8")

    summary = {
        "scientific_result": False,
        "training_seeds_per_method": 1,
        "evaluation_episodes_per_opponent": 192,
        "baseline": baseline,
        "v8": v8,
        "decision": "do_not_expand_v8; proceed_to_v9_online_option_policy",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
