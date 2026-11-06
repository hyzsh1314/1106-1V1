"""Compare fixed and progressive evasive curricula on common held-out episodes."""

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
LABELS = {
    "pursuit_fire": "Pursuit",
    "hold": "Hold",
    "evasive_fire": "Evasive",
    "standoff_fire": "Standoff",
}
SEEDS = (17, 85, 127)
FIXED_ROOTS = {
    17: ROOT / "runs/threat_obs_v2_multiseed_confirmation/seed17/stage3_evasive_focused_1p5m/heldout",
    85: ROOT / "runs/threat_obs_v2_evasive_curriculum_seed85_1p5m/common_heldout",
    127: ROOT / "runs/threat_obs_v2_multiseed_confirmation/seed127/stage3_evasive_focused_1p5m/heldout",
}
RAMP_ROOTS = {
    seed: ROOT / f"runs/threat_obs_v2_progressive_curriculum/seed{seed}_ramp25to100_1p5m/heldout"
    for seed in SEEDS
}


def load_roots(roots: dict[int, Path]) -> dict[int, dict[str, dict]]:
    return {
        seed: {
            opponent: json.loads((root / opponent / "summary.json").read_text(encoding="utf-8"))
            for opponent in OPPONENTS
        }
        for seed, root in roots.items()
    }


def count(summary: dict, key: str) -> int:
    return int(summary.get("learner_death_causes", {}).get(key, 0))


def rate(summary: dict, metric: str) -> float:
    if metric == "win":
        return float(summary["win_rate"])
    if metric == "timeout":
        return float(summary["timeout_rate"])
    return count(summary, metric) / int(summary["episodes"])


def plot(fixed: dict[int, dict[str, dict]], ramp: dict[int, dict[str, dict]]) -> None:
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    )
    metrics = (("win", "Win"), ("timeout", "Timeout"), ("shotdown", "Shot down"), ("crash", "Crash"))
    x = np.arange(len(SEEDS))
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2), sharex=True, sharey=True)
    for axis, (metric, label) in zip(axes.flat, metrics):
        fixed_values = [100 * rate(fixed[seed]["evasive_fire"], metric) for seed in SEEDS]
        ramp_values = [100 * rate(ramp[seed]["evasive_fire"], metric) for seed in SEEDS]
        axis.plot(x, fixed_values, "o-", color="#0072B2", lw=2, ms=6, label="Fixed 100% evasive")
        axis.plot(x, ramp_values, "s--", color="#D55E00", lw=2, ms=6, label="Ramp 25% to 100%")
        for index in range(len(SEEDS)):
            axis.plot([x[index], x[index]], [fixed_values[index], ramp_values[index]], color="#999999", lw=0.7, alpha=0.55)
        axis.set_title(label)
        axis.set_xticks(x, [f"seed={seed}" for seed in SEEDS])
        axis.set_ylabel("Episode rate (%)")
        axis.set_ylim(-3, 103)
        axis.grid(alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.suptitle("Evasive opponent: paired curriculum comparison", y=1.045, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT / "figure-05-progressive-vs-fixed.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT / "figure-05-progressive-vs-fixed.pdf", bbox_inches="tight")
    plt.close(fig)


def exact_table(fixed: dict[int, dict[str, dict]], ramp: dict[int, dict[str, dict]]) -> str:
    lines = [
        "| Curriculum | Seed | Opponent | Win | Timeout | Shot down | Crash | Legal launch | Resolved hit |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for curriculum, data in (("Fixed 100%", fixed), ("Ramp 25→100%", ramp)):
        for seed in SEEDS:
            for opponent in OPPONENTS:
                item = data[seed][opponent]
                n = int(item["episodes"])
                lines.append(
                    f"| {curriculum} | {seed} | {LABELS[opponent]} | {100 * item['win_rate']:.2f}% | "
                    f"{100 * item['timeout_rate']:.2f}% | {100 * count(item, 'shotdown') / n:.2f}% | "
                    f"{100 * count(item, 'crash') / n:.2f}% | "
                    f"{100 * item['legal_launch_acceptance_rate']:.2f}% | "
                    f"{100 * item['resolved_weapon_hit_rate']:.2f}% |"
                )
    return "\n".join(lines)


def summary_row(data: dict[int, dict[str, dict]], opponent: str, metric: str) -> tuple[float, float, float, float]:
    values = [100 * rate(data[seed][opponent], metric) for seed in SEEDS]
    return statistics.mean(values), statistics.median(values), min(values), max(values)


def write_report(fixed: dict[int, dict[str, dict]], ramp: dict[int, dict[str, dict]]) -> None:
    fixed_win = [100 * rate(fixed[s]["evasive_fire"], "win") for s in SEEDS]
    ramp_win = [100 * rate(ramp[s]["evasive_fire"], "win") for s in SEEDS]
    fixed_timeout = [100 * rate(fixed[s]["evasive_fire"], "timeout") for s in SEEDS]
    ramp_timeout = [100 * rate(ramp[s]["evasive_fire"], "timeout") for s in SEEDS]
    fixed_crash = [100 * rate(fixed[s]["evasive_fire"], "crash") for s in SEEDS]
    ramp_crash = [100 * rate(ramp[s]["evasive_fire"], "crash") for s in SEEDS]
    easy_ramp = {
        opponent: statistics.median([100 * rate(ramp[s][opponent], "win") for s in SEEDS])
        for opponent in ("pursuit_fire", "hold", "standoff_fire")
    }
    criteria = {
        "evasive_win_median_ge_30": statistics.median(ramp_win) >= 30,
        "evasive_timeout_median_le_20": statistics.median(ramp_timeout) <= 20,
        "evasive_crash_median_le_2": statistics.median(ramp_crash) <= 2,
        "easy_opponent_medians_ge_80": all(value >= 80 for value in easy_ramp.values()),
    }
    report = f"""# 渐进规避课程配对验证

## 结论

**25%→100% Evasive 的线性渐进课程未通过验收，也没有优于固定 100% Evasive 课程。** 在完全相同的三个训练种子和公共 held-out 场景上，Evasive 胜率中位数从固定课程的 {statistics.median(fixed_win):.2f}% 变为 {statistics.median(ramp_win):.2f}%；超时率中位数从 {statistics.median(fixed_timeout):.2f}% 升至 {statistics.median(ramp_timeout):.2f}%；坠毁率中位数从 {statistics.median(fixed_crash):.2f}% 升至 {statistics.median(ramp_crash):.2f}%。三种子的 Evasive 胜率配对变化为 {', '.join(f'{r-f:+.2f} pp' for f, r in zip(fixed_win, ramp_win))}。

这排除了“100% 规避对手切换过于突然”作为当前主要瓶颈。所有单元合法发射接受率仍为 100%，武器事件守恒通过，因此此轮失败不能归因于导弹发射接口。训练日志同时确认低层控制网络参数变化为 0；结合 seed127 在渐进课程下 64/192 局低空坠毁，下一主因应定位为**冻结低层控制器与粗粒度高层动作是否限制威胁规避机动**。

## 预注册门槛

| Criterion | Result | Pass |
|---|---:|:---:|
| Evasive win median ≥30% | {statistics.median(ramp_win):.2f}% | {'Yes' if criteria['evasive_win_median_ge_30'] else 'No'} |
| Evasive timeout median ≤20% | {statistics.median(ramp_timeout):.2f}% | {'Yes' if criteria['evasive_timeout_median_le_20'] else 'No'} |
| Evasive crash median ≤2% | {statistics.median(ramp_crash):.2f}% | {'Yes' if criteria['evasive_crash_median_le_2'] else 'No'} |
| Pursuit/Hold/Standoff median win ≥80% | {easy_ramp['pursuit_fire']:.2f}% / {easy_ramp['hold']:.2f}% / {easy_ramp['standoff_fire']:.2f}% | {'Yes' if criteria['easy_opponent_medians_ge_80'] else 'No'} |

## 精确结果（每个单元 192 局）

{exact_table(fixed, ramp)}

## 下一步实验

冻结 R1 奖励、26D 观测、导弹和动力学，做一个最小的层级动作消融，而不是继续叠加奖励项：

1. **A0 基线**：当前冻结低层控制器。
2. **A1 可训练低层**：从相同 checkpoint 继续，允许低层控制器以高层学习率的 0.1 倍更新；增加动作平滑和过载/低空约束，防止破坏基本飞行稳定性。
3. **A2 显式规避技能**：保持低层冻结，但给高层增加基于来袭导弹方位的 break-left、break-right、dive、climb 四个离散技能，验证动作表达能力是否为瓶颈。

先用 seed 17/85/127 各 0.5M，固定公共 192 局评估；只有 Evasive 胜率中位数提高且 crash、shotdown 不恶化时，才扩展到正式长训练。A1 和 A2 必须分开跑，不能同时修改，以保留因果解释。

## 统计边界

训练种子 n=3，仅用于工程筛选和配对方向判断；不报告 p 值或显著性。每个 checkpoint 的 192 个 episode 使用相同公共评估种子。
"""
    (OUTPUT / "progressive-comparison.md").write_text(report, encoding="utf-8")

    catalog_path = OUTPUT / "figure-catalog.md"
    catalog = catalog_path.read_text(encoding="utf-8")
    if "## Figure 5" not in catalog:
        catalog += """

## Figure 5 — Progressive versus fixed evasive curriculum

- Purpose: paired comparison of the only curriculum variable changed after the three-seed confirmation.
- Data: seeds 17, 85, and 127; 192 common held-out Evasive episodes per checkpoint.
- Encoding: fixed 100% Evasive and linear 25%→100% Evasive curricula; gray segments link paired training seeds.
- Main observation: the ramp does not improve Evasive wins and increases timeout/crash failure modes for representative seeds.
- Caveat: n=3 supports rejecting this candidate schedule, not estimating a precise population effect.
"""
        catalog_path.write_text(catalog, encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fixed = load_roots(FIXED_ROOTS)
    ramp = load_roots(RAMP_ROOTS)
    plot(fixed, ramp)
    write_report(fixed, ramp)
    print(json.dumps({"seeds": SEEDS, "output": str(OUTPUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
