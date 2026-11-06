"""Analyze the paired timeout-terminal-utility continuation experiment."""

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
SEEDS = (17, 85, 127)
OPPONENTS = ("pursuit_fire", "hold", "evasive_fire", "standoff_fire")
LABELS = {"pursuit_fire": "Pursuit", "hold": "Hold", "evasive_fire": "Evasive", "standoff_fire": "Standoff"}
WIDE = {seed: ROOT / f"runs/threat_obs_v2_wide_heading_ablation/seed{seed}_span90_1p5m/heldout" for seed in SEEDS}
TIMEOUT = {seed: ROOT / f"runs/threat_obs_v2_timeout025_ablation/seed{seed}_span90_timeout025_2m/heldout" for seed in SEEDS}


def load(roots: dict[int, Path]) -> dict[int, dict[str, dict]]:
    return {seed: {opponent: json.loads((root / opponent / "summary.json").read_text(encoding="utf-8")) for opponent in OPPONENTS} for seed, root in roots.items()}


def rate(item: dict, metric: str) -> float:
    if metric == "win":
        return float(item["win_rate"])
    if metric == "timeout":
        return float(item["timeout_rate"])
    return int(item.get("learner_death_causes", {}).get(metric, 0)) / int(item["episodes"])


def values(data: dict[int, dict[str, dict]], metric: str) -> list[float]:
    return [100 * rate(data[seed]["evasive_fire"], metric) for seed in SEEDS]


def plot(before: dict[int, dict[str, dict]], after: dict[int, dict[str, dict]]) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    x = np.arange(len(SEEDS))
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2), sharex=True, sharey=True)
    for axis, (metric, title) in zip(axes.flat, (("win", "Win"), ("timeout", "Timeout"), ("shotdown", "Shot down"), ("crash", "Crash"))):
        old, new = values(before, metric), values(after, metric)
        axis.plot(x, old, "o-", color="#009E73", lw=2, ms=6, label="R1 timeout=0 (1.5M)")
        axis.plot(x, new, "s--", color="#CC79A7", lw=2, ms=6, label="R1 timeout=±0.25 (+0.5M)")
        for index in range(len(SEEDS)):
            axis.plot([x[index], x[index]], [old[index], new[index]], color="#999999", lw=0.8, alpha=0.6)
        axis.set_title(title)
        axis.set_xticks(x, [f"seed={seed}" for seed in SEEDS])
        axis.set_ylabel("Episode rate (%)")
        axis.set_ylim(-3, 103)
        axis.grid(alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.suptitle("Evasive opponent: timeout-terminal-utility continuation", y=1.045, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT / "figure-07-timeout025-ablation.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT / "figure-07-timeout025-ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def exact_table(before: dict[int, dict[str, dict]], after: dict[int, dict[str, dict]]) -> str:
    lines = [
        "| Contract | Seed | Opponent | Win | Timeout | Shot down | Crash | Legal launch | Resolved hit |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, data in (("timeout=0", before), ("timeout=±0.25", after)):
        for seed in SEEDS:
            for opponent in OPPONENTS:
                item = data[seed][opponent]
                n = int(item["episodes"])
                deaths = item.get("learner_death_causes", {})
                lines.append(
                    f"| {label} | {seed} | {LABELS[opponent]} | {100*item['win_rate']:.2f}% | {100*item['timeout_rate']:.2f}% | "
                    f"{100*int(deaths.get('shotdown', 0))/n:.2f}% | {100*int(deaths.get('crash', 0))/n:.2f}% | "
                    f"{100*item['legal_launch_acceptance_rate']:.2f}% | {100*item['resolved_weapon_hit_rate']:.2f}% |"
                )
    return "\n".join(lines)


def write_report(before: dict[int, dict[str, dict]], after: dict[int, dict[str, dict]]) -> None:
    old = {metric: values(before, metric) for metric in ("win", "timeout", "shotdown", "crash")}
    new = {metric: values(after, metric) for metric in ("win", "timeout", "shotdown", "crash")}
    easy = {opponent: statistics.median([100 * rate(after[seed][opponent], "win") for seed in SEEDS]) for opponent in ("pursuit_fire", "hold", "standoff_fire")}
    report = f"""# 超时终局效用 ±0.25 消融

## 判定

**该候选被拒绝。** 在保留 ±90° 动作接口并从同一批 1.5M checkpoint 追加 0.5M 后，Evasive 胜率中位数从 {statistics.median(old['win']):.2f}% 降到 {statistics.median(new['win']):.2f}%，超时率中位数从 {statistics.median(old['timeout']):.2f}% 升到 {statistics.median(new['timeout']):.2f}%。三种子的胜率配对变化为 {', '.join(f'{new['win'][i]-old['win'][i]:+.2f} pp' for i in range(3))}，超时率变化为 {', '.join(f'{new['timeout'][i]-old['timeout'][i]:+.2f} pp' for i in range(3))}。

被击落率中位数从 {statistics.median(old['shotdown']):.2f}% 降至 {statistics.median(new['shotdown']):.2f}%，说明策略确实学会了保命；但它通过拖延而不是完成拦截来实现。因此问题不是奖励没有生效，而是 ±0.25 的目标更容易被“避免死亡直到时限”满足。

所有合法发射接受率仍为 100%，易对手胜率中位数为 {easy['pursuit_fire']:.2f}%/{easy['hold']:.2f}%/{easy['standoff_fire']:.2f}%。正式候选应回退到 **±90° + R1 timeout=0 的 1.5M checkpoint**，不能继续使用本轮 2M checkpoint。

## 精确结果

{exact_table(before, after)}

## 下一步

不要继续扫描 0.25、0.5、1.0 等超时权重，这会把平台调参变成奖励搜索。下一项更有诊断价值的单变量是显式威胁响应技能：在 26D 观测已有 warning/urgency 的前提下，为高层策略提供来袭导弹相对方位驱动的 break-left/break-right 技能，仍使用原 R1 timeout=0。先做动作可达性与低空安全单元测试，再跑三种子 0.5M。

## 统计边界

这是配对 continuation 筛选；后者比前者多 0.5M 训练，因此结论是“继续使用该奖励不会改善候选”，不是对奖励系数的无偏因果效应估计。
"""
    (OUTPUT / "timeout025-analysis.md").write_text(report, encoding="utf-8")
    catalog_path = OUTPUT / "figure-catalog.md"
    catalog = catalog_path.read_text(encoding="utf-8")
    if "## Figure 7" not in catalog:
        catalog += """

## Figure 7 — Timeout-terminal-utility continuation

- Purpose: test whether aligning tactical-timeout training utility with evaluation adjudication reduces stalling.
- Data: paired seeds 17, 85, and 127; ±90° checkpoint before and after a 0.5M continuation with timeout utility ±0.25.
- Main observation: shotdown decreases but timeout increases sharply and median win declines, indicating survival-through-stalling.
- Caveat: the after checkpoint has 0.5M additional updates, so the figure rejects this continuation recipe rather than estimating a pure coefficient effect.
"""
        catalog_path.write_text(catalog, encoding="utf-8")


def main() -> None:
    before, after = load(WIDE), load(TIMEOUT)
    plot(before, after)
    write_report(before, after)
    print(json.dumps({"seeds": SEEDS, "output": str(OUTPUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
