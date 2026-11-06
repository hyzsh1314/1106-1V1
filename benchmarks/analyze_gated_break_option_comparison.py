"""Compare baseline, ungated, and target-aspect-gated threat-break options."""

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
LABELS = {
    "pursuit_fire": "Pursuit",
    "hold": "Hold",
    "evasive_fire": "Evasive",
    "standoff_fire": "Standoff",
}
BASELINE_ROOTS = {
    seed: ROOT / f"runs/threat_obs_v2_wide_heading_ablation/seed{seed}_span90_1p5m/heldout"
    for seed in SEEDS
}
UNGATED_ROOTS = {
    seed: ROOT / f"runs/threat_obs_v2_break_option_ablation/seed{seed}_break_option_2m/heldout"
    for seed in SEEDS
}
GATED_ROOTS = {
    seed: ROOT / f"runs/threat_obs_v2_gated_break_option_ablation/seed{seed}_gated_break_option_2m/heldout"
    for seed in SEEDS
}


def load(roots: dict[int, Path]) -> dict[int, dict[str, dict]]:
    return {
        seed: {
            opponent: json.loads((root / opponent / "summary.json").read_text(encoding="utf-8"))
            for opponent in OPPONENTS
        }
        for seed, root in roots.items()
    }


def rate(item: dict, metric: str) -> float:
    if metric == "win":
        return float(item["win_rate"])
    if metric == "timeout":
        return float(item["timeout_rate"])
    return int(item.get("learner_death_causes", {}).get(metric, 0)) / int(item["episodes"])


def values(data: dict[int, dict[str, dict]], metric: str) -> list[float]:
    return [100 * rate(data[seed]["evasive_fire"], metric) for seed in SEEDS]


def plot(baseline, ungated, gated) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    metrics = (("win", "Win"), ("timeout", "Timeout"),
               ("shotdown", "Shot down"), ("crash", "Crash"))
    x = np.arange(len(SEEDS))
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2), sharex=True, sharey=True)
    series = (
        (baseline, "±90° baseline", "#0072B2", "o-"),
        (ungated, "Ungated recovery", "#D55E00", "s--"),
        (gated, "TA≤90° recovery", "#009E73", "^-."),
    )
    for axis, (metric, title) in zip(axes.flat, metrics):
        for data, label, color, style in series:
            axis.plot(x, values(data, metric), style, color=color,
                      lw=2, ms=6, label=label)
        axis.set_title(title)
        axis.set_xticks(x, [f"seed={seed}" for seed in SEEDS])
        axis.set_ylabel("Episode rate (%)")
        axis.set_ylim(-3, 103)
        axis.grid(alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.suptitle("Evasive opponent: target-aspect-gated break recovery",
                 y=1.045, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT / "figure-10-gated-break-option-ablation.png",
                dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT / "figure-10-gated-break-option-ablation.pdf",
                bbox_inches="tight")
    plt.close(fig)


def result_table(baseline, ungated, gated) -> str:
    lines = [
        "| Interface | Seed | Opponent | Win | Timeout | Shot down | Crash | Legal launch | Resolved hit |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, data in (("±90° baseline", baseline),
                        ("Ungated option", ungated),
                        ("TA≤90° gated option", gated)):
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


def write_report(baseline, ungated, gated) -> None:
    metric_names = ("win", "timeout", "shotdown", "crash")
    old = {metric: values(baseline, metric) for metric in metric_names}
    raw = {metric: values(ungated, metric) for metric in metric_names}
    new = {metric: values(gated, metric) for metric in metric_names}
    easy = {
        opponent: statistics.median([
            100 * rate(gated[seed][opponent], "win") for seed in SEEDS])
        for opponent in ("pursuit_fire", "hold", "standoff_fire")
    }
    all_conserved = all(
        bool(gated[seed][opponent].get("weapon_event_conservation_ok"))
        for seed in SEEDS for opponent in OPPONENTS)
    all_legal = all(
        float(gated[seed][opponent]["legal_launch_acceptance_rate"]) == 1.0
        for seed in SEEDS for opponent in OPPONENTS)
    win_pass = statistics.median(new["win"]) >= 30
    timeout_pass = statistics.median(new["timeout"]) <= 20
    crash_pass = statistics.median(new["crash"]) <= 2
    easy_pass = min(easy.values()) >= 80
    accepted = (win_pass and timeout_pass and crash_pass and easy_pass and
                all_conserved and all_legal)
    decision = ("通过全部预注册门槛，可进入5M确认。" if accepted else
                "未通过全部预注册门槛，不应替换正式基准配置。")
    report = f"""# TA门控状态化来袭规避 option 消融

## 判定

**{decision}** 门控 option 的 Evasive 胜率中位数为 {statistics.median(new['win']):.2f}%（基线 {statistics.median(old['win']):.2f}%，无门控 {statistics.median(raw['win']):.2f}%）；超时率中位数为 {statistics.median(new['timeout']):.2f}%（基线 {statistics.median(old['timeout']):.2f}%，无门控 {statistics.median(raw['timeout']):.2f}%）；被击落率中位数为 {statistics.median(new['shotdown']):.2f}%（基线 {statistics.median(old['shotdown']):.2f}%，无门控 {statistics.median(raw['shotdown']):.2f}%）。

本轮只改变恢复门控：告警清除后，仅在对手目标方位角 `TA≤90°`（对手机头未朝向我方闭合）时执行10步恢复；`TA>90°` 时立即把航向控制权还给PPO。奖励、武器、PPO、2秒最短规避、1秒告警清除滞回和2秒恢复上限均保持不变。Pursuit/Hold/Standoff 胜率中位数为 {easy['pursuit_fire']:.2f}%/{easy['hold']:.2f}%/{easy['standoff_fire']:.2f}%。

## 验收门槛

| Criterion | Result | Pass |
|---|---:|:---:|
| Evasive win median ≥30% | {statistics.median(new['win']):.2f}% | {'Yes' if win_pass else 'No'} |
| Evasive timeout median ≤20% | {statistics.median(new['timeout']):.2f}% | {'Yes' if timeout_pass else 'No'} |
| Evasive crash median ≤2% | {statistics.median(new['crash']):.2f}% | {'Yes' if crash_pass else 'No'} |
| Easy-opponent median wins ≥80% | {easy['pursuit_fire']:.2f}% / {easy['hold']:.2f}% / {easy['standoff_fire']:.2f}% | {'Yes' if easy_pass else 'No'} |
| Legal launch =100% and event conservation | {'Yes' if all_legal and all_conserved else 'No'} | {'Yes' if all_legal and all_conserved else 'No'} |

## 精确结果

{result_table(baseline, ungated, gated)}

## 实验边界

- 三个门控checkpoint从相同的±90° 1.5M checkpoint继续训练约0.5M动作；每个checkpoint使用相同规模的192个双边平衡公共场景。
- 与上一轮唯一的行为差异是恢复TA门控，但续训随机轨迹会随接口变化而分叉。本实验用于工程筛选，不声称n=3具有统计显著性。
- 第一版无效stateless批次继续排除；比较中的无门控数据仅使用已验证的stateful批次。
"""
    (OUTPUT / "gated-break-option-analysis.md").write_text(
        report, encoding="utf-8")
    catalog_path = OUTPUT / "figure-catalog.md"
    catalog = catalog_path.read_text(encoding="utf-8")
    if "## Figure 10" not in catalog:
        catalog += """

## Figure 10 — Target-aspect-gated threat-break recovery

- Purpose: test whether suppressing forced reacquisition against a nose-on opponent preserves the timeout benefit without the shotdown regression.
- Data: paired seeds 17, 85, and 127 for ±90° baseline, ungated stateful option, and TA≤90° gated option.
- Encoding: Evasive win, timeout, shotdown, and crash episode rates; 192 common-size side-balanced episodes per checkpoint.
- Decision: see `gated-break-option-analysis.md`; acceptance is based on pre-registered multi-opponent median gates.
"""
        catalog_path.write_text(catalog, encoding="utf-8")


def main() -> None:
    baseline = load(BASELINE_ROOTS)
    ungated = load(UNGATED_ROOTS)
    gated = load(GATED_ROOTS)
    plot(baseline, ungated, gated)
    write_report(baseline, ungated, gated)
    print(json.dumps({"seeds": SEEDS, "output": str(OUTPUT)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
