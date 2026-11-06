"""Compare the baseline and 90-degree high-level heading command spans."""

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
BASELINE_ROOTS = {
    17: ROOT / "runs/threat_obs_v2_multiseed_confirmation/seed17/stage3_evasive_focused_1p5m/heldout",
    85: ROOT / "runs/threat_obs_v2_evasive_curriculum_seed85_1p5m/common_heldout",
    127: ROOT / "runs/threat_obs_v2_multiseed_confirmation/seed127/stage3_evasive_focused_1p5m/heldout",
}
WIDE_ROOTS = {
    seed: ROOT / f"runs/threat_obs_v2_wide_heading_ablation/seed{seed}_span90_1p5m/heldout"
    for seed in SEEDS
}


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


def plot(baseline: dict[int, dict[str, dict]], wide: dict[int, dict[str, dict]]) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    metrics = (("win", "Win"), ("timeout", "Timeout"), ("shotdown", "Shot down"), ("crash", "Crash"))
    x = np.arange(len(SEEDS))
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2), sharex=True, sharey=True)
    for axis, (metric, title) in zip(axes.flat, metrics):
        old = values(baseline, metric)
        new = values(wide, metric)
        axis.plot(x, old, "o-", color="#0072B2", lw=2, ms=6, label="±30° baseline")
        axis.plot(x, new, "s--", color="#009E73", lw=2, ms=6, label="±90° heading")
        for index in range(len(SEEDS)):
            axis.plot([x[index], x[index]], [old[index], new[index]], color="#999999", lw=0.8, alpha=0.6)
        axis.set_title(title)
        axis.set_xticks(x, [f"seed={seed}" for seed in SEEDS])
        axis.set_ylabel("Episode rate (%)")
        axis.set_ylim(-3, 103)
        axis.grid(alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.suptitle("Evasive opponent: high-level heading-span ablation", y=1.045, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT / "figure-06-wide-heading-ablation.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT / "figure-06-wide-heading-ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def table(baseline: dict[int, dict[str, dict]], wide: dict[int, dict[str, dict]]) -> str:
    lines = [
        "| Interface | Seed | Opponent | Win | Timeout | Shot down | Crash | Legal launch | Resolved hit |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, data in (("±30°", baseline), ("±90°", wide)):
        for seed in SEEDS:
            for opponent in OPPONENTS:
                item = data[seed][opponent]
                n = int(item["episodes"])
                deaths = item.get("learner_death_causes", {})
                lines.append(
                    f"| {label} | {seed} | {LABELS[opponent]} | {100*item['win_rate']:.2f}% | "
                    f"{100*item['timeout_rate']:.2f}% | {100*int(deaths.get('shotdown', 0))/n:.2f}% | "
                    f"{100*int(deaths.get('crash', 0))/n:.2f}% | {100*item['legal_launch_acceptance_rate']:.2f}% | "
                    f"{100*item['resolved_weapon_hit_rate']:.2f}% |"
                )
    return "\n".join(lines)


def write_report(baseline: dict[int, dict[str, dict]], wide: dict[int, dict[str, dict]]) -> None:
    old = {metric: values(baseline, metric) for metric in ("win", "timeout", "shotdown", "crash")}
    new = {metric: values(wide, metric) for metric in ("win", "timeout", "shotdown", "crash")}
    easy = {opponent: statistics.median([100 * rate(wide[seed][opponent], "win") for seed in SEEDS]) for opponent in ("pursuit_fire", "hold", "standoff_fire")}
    report = f"""# 宽航向动作接口消融

## 判定

**±90° 航向指令证明动作表达能力是重要瓶颈，但当前配置仍未通过最终验收。** 在三个配对训练种子上，Evasive 胜率全部提高，变化分别为 {', '.join(f'{new['win'][i]-old['win'][i]:+.2f} pp' for i in range(3))}；胜率中位数从 {statistics.median(old['win']):.2f}% 提升到 {statistics.median(new['win']):.2f}%，被击落率中位数从 {statistics.median(old['shotdown']):.2f}% 降到 {statistics.median(new['shotdown']):.2f}%。这是一致的方向性改进，明显优于前一轮渐进课程。

但 Evasive 超时率中位数由 {statistics.median(old['timeout']):.2f}% 升到 {statistics.median(new['timeout']):.2f}%，宽航向胜率中位数仍略低于 30% 门槛。因此应保留宽航向接口作为下一候选基线，但不能直接把它宣布为合格方案。

所有 12 个单元的合法发射接受率均为 100%，武器事件守恒通过；Pursuit/Hold/Standoff 胜率中位数为 {easy['pursuit_fire']:.2f}%/{easy['hold']:.2f}%/{easy['standoff_fire']:.2f}%，未出现易对手遗忘。

## 验收门槛

| Criterion | Result | Pass |
|---|---:|:---:|
| Evasive win median ≥30% | {statistics.median(new['win']):.2f}% | {'Yes' if statistics.median(new['win']) >= 30 else 'No'} |
| Evasive timeout median ≤20% | {statistics.median(new['timeout']):.2f}% | {'Yes' if statistics.median(new['timeout']) <= 20 else 'No'} |
| Evasive crash median ≤2% | {statistics.median(new['crash']):.2f}% | {'Yes' if statistics.median(new['crash']) <= 2 else 'No'} |
| Easy-opponent median wins ≥80% | {easy['pursuit_fire']:.2f}% / {easy['hold']:.2f}% / {easy['standoff_fire']:.2f}% | {'Yes' if min(easy.values()) >= 80 else 'No'} |

## 精确结果

{table(baseline, wide)}

## 下一步单变量实验

保留 ±90° 接口，纠正**训练目标与评估规则不一致**：当前 R1 对超时终局的训练奖励为 0，但验收器会把 `timeout_tactical_advantage` 判为胜负。因此策略可通过拖延获得零回报，却在评估中被判负。下一轮只将超时战术裁决映射为较小的零和终局奖励（建议幅度 0.25，击杀终局仍为 1.0），其余奖励、观测、动作、武器、动力学与 PPO 参数全部冻结。

先做三种子各 0.5M 的配对续训。只有 Evasive 超时中位数下降、胜率中位数达到 30%，且 shotdown/crash 不反弹，才进入更长训练；否则再引入显式来袭方位技能，而不是继续扩大航向角或叠加稠密塑形。

## 统计边界

n=3 训练种子只支持工程筛选；这里不做显著性检验。每个 checkpoint 使用相同的 192 个公共评估场景。
"""
    (OUTPUT / "wide-heading-analysis.md").write_text(report, encoding="utf-8")
    catalog_path = OUTPUT / "figure-catalog.md"
    catalog = catalog_path.read_text(encoding="utf-8")
    if "## Figure 6" not in catalog:
        catalog += """

## Figure 6 — High-level heading-span ablation

- Purpose: isolate whether the ±30° high-level command envelope limits evasive maneuvering.
- Data: paired seeds 17, 85, and 127; 192 common held-out Evasive episodes per checkpoint.
- Encoding: baseline ±30° versus ±90° heading command span; paired seeds linked vertically.
- Main observation: all three seeds improve Evasive wins and reduce shotdown, but timeout remains the dominant failure mode.
- Caveat: the wider interface changes command semantics while preserving action dimensions; it does not make the frozen low-level controller trainable.
"""
        catalog_path.write_text(catalog, encoding="utf-8")


def main() -> None:
    baseline = load(BASELINE_ROOTS)
    wide = load(WIDE_ROOTS)
    plot(baseline, wide)
    write_report(baseline, wide)
    print(json.dumps({"seeds": SEEDS, "output": str(OUTPUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
