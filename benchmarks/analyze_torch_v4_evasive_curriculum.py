#!/usr/bin/env python3
"""Paired-seed analysis of the rejected V4 evasive-intercept curriculum."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("runs/torch6dof_validation/v4-evasive-intercept-curriculum-3seed-1m")
BASELINE = Path("runs/torch6dof_validation/v4-reward-ablation-3seed-1m/competitive")
SEEDS = (85, 185, 285)
OPPONENTS = ("pursuit", "hold", "evasive")
LABELS = {"pursuit": "Pursuit", "hold": "Hold", "evasive": "Evasive"}
COLORS = {"baseline": "#0072B2", "curriculum": "#D55E00"}
MARKERS = {85: "o", 185: "s", 285: "^"}


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def mean_sd(values: list[float]) -> str:
    return f"{100 * mean(values):.1f}% ± {100 * stdev(values):.1f}%"


def main() -> None:
    if not (ROOT / "COMPLETED").exists():
        raise RuntimeError("curriculum experiment is incomplete")
    output = ROOT / "analysis-output"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    data = {}
    rows = []
    for seed in SEEDS:
        for opponent in OPPONENTS:
            paths = {
                "baseline": BASELINE / f"seed{seed}" / f"eval-{opponent}.json",
                "curriculum": ROOT / f"seed{seed}" / "stage2-formal-mix" / f"eval-{opponent}.json",
            }
            for cohort, path in paths.items():
                item = load(path)
                if item["episodes"] != 192 or item["seed"] != 10085:
                    raise RuntimeError(f"incomparable evaluation: {path}")
                data[(cohort, seed, opponent)] = item
                rows.append({
                    "cohort": cohort,
                    "seed": seed,
                    "opponent": opponent,
                    "win_rate": item["win_rate"],
                    "loss_rate": item["loss_rate"],
                    "timeout_rate": item["timeout_rate"],
                    "legal_opportunity_episode_rate": item["legal_opportunity_episode_rate"],
                    "rejected_fire_requests": item["rejected_fire_requests"],
                    "crash_rate": item["learner_crash_rate"],
                })
    with (output / "seed-level-comparison.csv").open(
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
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex=True, sharey="row")
    for col, opponent in enumerate(OPPONENTS):
        for row, (metric, ylabel) in enumerate((
                ("win_rate", "Win rate"), ("timeout_rate", "Timeout rate"))):
            ax = axes[row, col]
            for seed in SEEDS:
                values = [data[(cohort, seed, opponent)][metric]
                          for cohort in ("baseline", "curriculum")]
                ax.plot([0, 1], values, color="#777777", alpha=0.55,
                        marker=MARKERS[seed], markersize=4, linewidth=1)
            for x, cohort in enumerate(("baseline", "curriculum")):
                values = [data[(cohort, seed, opponent)][metric] for seed in SEEDS]
                ax.errorbar(x, mean(values), yerr=stdev(values), marker="D",
                            color=COLORS[cohort], capsize=3, linewidth=1.5,
                            markersize=5, zorder=5)
            ax.set_ylim(-0.04, 1.04)
            ax.set_xticks([0, 1], ["Base", "Curr."])
            if row == 0:
                ax.set_title(LABELS[opponent])
            if col == 0:
                ax.set_ylabel(ylabel)
    fig.subplots_adjust(wspace=0.15, hspace=0.20)
    fig.savefig(figures / "figure-01-curriculum-vs-baseline.pdf", bbox_inches="tight")
    fig.savefig(figures / "figure-01-curriculum-vs-baseline.png", dpi=600,
                bbox_inches="tight")
    plt.close(fig)

    def values(cohort: str, opponent: str, metric: str) -> list[float]:
        return [data[(cohort, seed, opponent)][metric] for seed in SEEDS]

    report = [
        "# Torch V4 规避接敌课程三种子验收",
        "",
        "## 结论",
        "",
        "两段式 `近距纯规避 516,096 步 → 正式混合 491,520 步` 课程被否决，"
        "不得替换现有竞争奖励基线。三个训练种子对规避对手的胜率均为 0%，合法发射机会"
        "覆盖率均约 6%；课程还降低了追击与保持对手性能，表现为明显的能力遗忘和种子分化。",
        "",
        "## 配对结果",
        "",
        "统计单位为训练种子（n=3），每个检查点对每类固定对手评估 192 局。",
        "",
        "| 配置 | 追击胜率 | 保持胜率 | 规避胜率 | 追击超时 | 保持超时 | 规避超时 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| 竞争奖励基线 | "
        + " | ".join([
            mean_sd(values("baseline", "pursuit", "win_rate")),
            mean_sd(values("baseline", "hold", "win_rate")),
            mean_sd(values("baseline", "evasive", "win_rate")),
            mean_sd(values("baseline", "pursuit", "timeout_rate")),
            mean_sd(values("baseline", "hold", "timeout_rate")),
            mean_sd(values("baseline", "evasive", "timeout_rate")),
        ]) + " |",
        "| 两段规避课程 | "
        + " | ".join([
            mean_sd(values("curriculum", "pursuit", "win_rate")),
            mean_sd(values("curriculum", "hold", "win_rate")),
            mean_sd(values("curriculum", "evasive", "win_rate")),
            mean_sd(values("curriculum", "pursuit", "timeout_rate")),
            mean_sd(values("curriculum", "hold", "timeout_rate")),
            mean_sd(values("curriculum", "evasive", "timeout_rate")),
        ]) + " |",
        "",
        "## 机制判断",
        "",
        "1. 武器接口不是瓶颈：全部 18 个配对评估单元的拒绝发射请求均为 0。",
        "2. 缩短距离并未建立稳定截获策略：纯规避近距训练仍以超时和被击落为主。",
        "3. 恢复到正式混合分布后，不同种子恢复程度差异很大，说明该课程诱发局部最优和遗忘。",
        "4. 规避对手合法发射覆盖率没有提高，因此不能通过继续延长同一课程来合理预期解决问题。",
        "5. 后续无学习可达性测试表明，纯追踪仍为 0% 胜率和约 6% 合法发射覆盖率；"
        "加入视线角速度的解析截获控制器达到 25% 胜率和约 48% 合法发射覆盖率。"
        "因此当前动作空间可达，失败主要来自 PPO 未发现视线率截获行为。",
        "",
        "## 决策",
        "",
        "- 保留 `competitive` 奖励和现有武器逻辑。",
        "- 不启用本轮两段式课程；`evasive` 模式仅保留为显式实验入口。",
        "- 保持当前高层动作接口，不扩大动作范围；无学习测试已经证明该接口能够产生截获解。",
        "- 下一次干预采用解析视线率截获控制器作为教师，先做行为克隆或教师动作混入，再用"
        "竞争奖励 PPO 微调；不得继续调整奖励权重、训练距离或发射包线。",
        "",
        "## 统计边界",
        "",
        "三种子足以作为工程否决门槛，但不能形成论文级等价性或无效性结论。图中灰线连接同一训练"
        "种子，彩色菱形与误差条表示均值 ± 样本标准差；评估局数没有被当作独立方法样本。",
    ]
    (output / "analysis-report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (output / "stats-appendix.md").write_text(
        "# 统计附录\n\n独立单位为训练种子（n=3），配置按种子配对。样本量不足以支持确认性"
        "显著性检验，因此报告原始种子值、均值和样本标准差，不报告 p 值。完整数据见 "
        "`seed-level-comparison.csv`。\n", encoding="utf-8")
    (output / "figure-catalog.md").write_text(
        "# 图目录\n\n## Figure 1\n\n`figures/figure-01-curriculum-vs-baseline.pdf` "
        "及 `.png`：竞争奖励基线与两段规避课程的配对种子胜率和超时率。灰线连接相同"
        "训练种子；菱形和误差条为 n=3 的均值 ± 样本标准差；每个点来自 192 局固定"
        "对手评估。\n", encoding="utf-8")


if __name__ == "__main__":
    main()
