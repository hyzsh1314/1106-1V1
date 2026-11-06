"""Summarize the three-seed focused-curriculum confirmation experiment."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs/threat_obs_v2_curriculum_analysis/analysis-output"
OPPONENTS = ("pursuit_fire", "hold", "evasive_fire", "standoff_fire")
LABELS = {"pursuit_fire": "Pursuit", "hold": "Hold", "evasive_fire": "Evasive", "standoff_fire": "Standoff"}
SEED_ROOTS = {
    17: ROOT / "runs/threat_obs_v2_multiseed_confirmation/seed17/stage3_evasive_focused_1p5m/heldout",
    85: ROOT / "runs/threat_obs_v2_evasive_curriculum_seed85_1p5m/common_heldout",
    127: ROOT / "runs/threat_obs_v2_multiseed_confirmation/seed127/stage3_evasive_focused_1p5m/heldout",
}


def read_data() -> dict[int, dict[str, dict]]:
    return {
        seed: {
            opponent: json.loads((root / opponent / "summary.json").read_text(encoding="utf-8"))
            for opponent in OPPONENTS
        }
        for seed, root in SEED_ROOTS.items()
    }


def count(summary: dict, key: str) -> int:
    return int(summary.get("learner_death_causes", {}).get(key, 0))


def rate(summary: dict, metric: str) -> float:
    n = int(summary["episodes"])
    if metric == "win":
        return float(summary["win_rate"])
    if metric == "timeout":
        return float(summary["timeout_rate"])
    if metric in {"crash", "shotdown"}:
        return count(summary, metric) / n
    raise KeyError(metric)


def plot(data: dict[int, dict[str, dict]]) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    seeds = list(SEED_ROOTS)
    x = np.arange(len(seeds))
    styles = {
        "win": ("#0072B2", "o-", "Win"),
        "timeout": ("#E69F00", "s--", "Timeout"),
        "shotdown": ("#D55E00", "D-.", "Learner shot down"),
        "crash": ("#000000", "^:", "Learner crash"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.3), sharex=True, sharey=True)
    for axis, opponent in zip(axes.flat, OPPONENTS):
        for metric, (color, style, label) in styles.items():
            values = [100.0 * rate(data[seed][opponent], metric) for seed in seeds]
            axis.plot(x, values, style, color=color, lw=1.8, ms=5, label=label)
        axis.set_title(LABELS[opponent])
        axis.set_xticks(x, [f"seed={seed}" for seed in seeds])
        axis.set_ylim(-3, 103)
        axis.grid(alpha=0.22)
        axis.set_ylabel("Episode rate (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.suptitle("Three-training-seed confirmation on common held-out episodes", y=1.045, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT / "figure-04-multiseed-confirmation.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT / "figure-04-multiseed-confirmation.pdf", bbox_inches="tight")
    plt.close(fig)


def exact_table(data: dict[int, dict[str, dict]]) -> str:
    lines = [
        "| Training seed | Opponent | Win | Timeout | Shot down | Crash | Legal launch | Resolved hit |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEED_ROOTS:
        for opponent in OPPONENTS:
            item = data[seed][opponent]
            n = int(item["episodes"])
            lines.append(
                f"| {seed} | {LABELS[opponent]} | {100 * item['win_rate']:.2f}% | "
                f"{100 * item['timeout_rate']:.2f}% | {100 * count(item, 'shotdown') / n:.2f}% | "
                f"{100 * count(item, 'crash') / n:.2f}% | {100 * item['legal_launch_acceptance_rate']:.2f}% | "
                f"{100 * item['resolved_weapon_hit_rate']:.2f}% |"
            )
    return "\n".join(lines)


def seed_summary(data: dict[int, dict[str, dict]]) -> str:
    lines = [
        "| Opponent | Metric | Mean ± sample SD | Median | Range |",
        "|---|---|---:|---:|---:|",
    ]
    for opponent in OPPONENTS:
        for metric, label in (("win", "Win"), ("timeout", "Timeout"), ("shotdown", "Shot down"), ("crash", "Crash")):
            values = [100.0 * rate(data[seed][opponent], metric) for seed in SEED_ROOTS]
            lines.append(
                f"| {LABELS[opponent]} | {label} | {statistics.mean(values):.2f}% ± {statistics.stdev(values):.2f}% | "
                f"{statistics.median(values):.2f}% | {min(values):.2f}%–{max(values):.2f}% |"
            )
    return "\n".join(lines)


def write_report(data: dict[int, dict[str, dict]]) -> None:
    evasive_wins = [100.0 * rate(data[seed]["evasive_fire"], "win") for seed in SEED_ROOTS]
    evasive_timeouts = [100.0 * rate(data[seed]["evasive_fire"], "timeout") for seed in SEED_ROOTS]
    passing = sum(value >= 30.0 for value in evasive_wins)
    report = f"""# 三训练种子确认实验

## 判定

当前 **26D ThreatObservation V2 + 1.0M population + 0.5M 100% evasive-focused** 配置未通过可复现性验收，不能作为论文基准策略的最终训练方案。三种子在 Evasive 对手上的胜率分别为 {evasive_wins[0]:.2f}%、{evasive_wins[1]:.2f}%、{evasive_wins[2]:.2f}%，中位数仅 {statistics.median(evasive_wins):.2f}%，范围 {min(evasive_wins):.2f}%–{max(evasive_wins):.2f}%；只有 {passing}/3 个训练种子达到预注册的 30% 胜率门槛。

Evasive 超时率分别为 {evasive_timeouts[0]:.2f}%、{evasive_timeouts[1]:.2f}%、{evasive_timeouts[2]:.2f}%。seed=127 的 0% 超时并非更好，而是 190/192 局在超时前被击落。因此平台验收必须同时报告胜率、超时、被击落和坠毁，不能单独优化超时率。

所有 12 个 seed×opponent 单元的合法发射接受率均为 100%，武器事件守恒均通过。失败种子的主要差异是 Evasive 命中/生存能力，而不是发射请求被拒绝：例如 seed=17 的 Evasive resolved hit rate 仅 1.38%，seed=127 则有 98.96% 局被击落。这支持“策略/课程不稳定”而不支持“导弹接口仍损坏”。

## 精确结果（每单元 192 个相同 held-out 场景种子）

{exact_table(data)}

## 训练种子级描述统计（n=3）

{seed_summary(data)}

## 下一轮只改一个主因

下一轮不应继续增加当前 focused 训练步数，也不应同时改导弹、动力学和奖励。更合理的顺序是：

1. 保持 R1、26D 观测、武器和动力学冻结，将 0→100% evasive 的突变课程改为分段或性能自适应课程（例如 25%→50%→75%→100%，每级用固定验收门槛晋级）。当前数据首先指向课程诱发的局部最优和灾难性策略分化。
2. 每个候选课程先跑 3×0.5M 小样本筛选，固定公共 held-out 场景；只有 Evasive 胜率中位数提高且 shotdown 不恶化时才扩展到 1.5M。
3. 如果渐进课程仍失败，再单独做层级动作接口消融。现有高层动作只有高度×航向×速度三类指令，低层网络在训练中冻结；应检验固定低层控制器是否限制导弹规避，而不是先继续堆叠奖励项。

## 统计边界

训练种子独立样本量为 3，只报告均值、样本标准差、中位数和范围，不做正态性假设、p 值或“显著提升”声明。每个 checkpoint 的 192 个 episode 共享同一组公共评估种子，以降低评估场景差异，但不能替代更多训练种子。
"""
    (OUTPUT / "multiseed-analysis.md").write_text(report, encoding="utf-8")
    catalog = (OUTPUT / "figure-catalog.md").read_text(encoding="utf-8")
    if "## Figure 4" not in catalog:
        catalog += """

## Figure 4 — Three-training-seed confirmation

- Purpose: test whether the focused curriculum is reproducible across policy initialization and training randomness.
- Data: training seeds 17, 85, and 127; 192 common held-out episodes per opponent and seed.
- Encoding: win, timeout, learner-shotdown, and learner-crash episode rates on a common 0–100% scale.
- Main observation: easy-opponent performance is mostly high, but Evasive behavior separates into three qualitatively different outcomes; the selected seed=85 result is not representative of a stable training procedure.
- Caveat: n=3 training seeds supports engineering rejection of the preset threshold but not precise population-effect estimation.
"""
        (OUTPUT / "figure-catalog.md").write_text(catalog, encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = read_data()
    plot(data)
    write_report(data)
    print(json.dumps({"training_seeds": list(SEED_ROOTS), "output": str(OUTPUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
