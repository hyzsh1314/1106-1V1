"""Compare the wide-heading baseline with threat-relative break-skill continuation."""

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
BREAK_ROOTS = {
    seed: ROOT / f"runs/threat_obs_v2_break_skill_ablation_v2/seed{seed}_break_skill_2m/heldout"
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


def plot(baseline: dict[int, dict[str, dict]], skill: dict[int, dict[str, dict]]) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    metrics = (("win", "Win"), ("timeout", "Timeout"), ("shotdown", "Shot down"), ("crash", "Crash"))
    x = np.arange(len(SEEDS))
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2), sharex=True, sharey=True)
    for axis, (metric, title) in zip(axes.flat, metrics):
        old = values(baseline, metric)
        new = values(skill, metric)
        axis.plot(x, old, "o-", color="#0072B2", lw=2, ms=6, label="±90° baseline")
        axis.plot(x, new, "s--", color="#D55E00", lw=2, ms=6, label="Threat-relative break")
        for index in range(len(SEEDS)):
            axis.plot([x[index], x[index]], [old[index], new[index]], color="#999999", lw=0.8, alpha=0.6)
        axis.set_title(title)
        axis.set_xticks(x, [f"seed={seed}" for seed in SEEDS])
        axis.set_ylabel("Episode rate (%)")
        axis.set_ylim(-3, 103)
        axis.grid(alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.suptitle("Evasive opponent: threat-relative break-skill continuation", y=1.045, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT / "figure-08-break-skill-ablation.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT / "figure-08-break-skill-ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def table(baseline: dict[int, dict[str, dict]], skill: dict[int, dict[str, dict]]) -> str:
    lines = [
        "| Interface | Seed | Opponent | Win | Timeout | Shot down | Crash | Legal launch | Resolved hit |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, data in (("±90° baseline", baseline), ("Threat-relative break", skill)):
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


def write_report(baseline: dict[int, dict[str, dict]], skill: dict[int, dict[str, dict]]) -> None:
    old = {metric: values(baseline, metric) for metric in ("win", "timeout", "shotdown", "crash")}
    new = {metric: values(skill, metric) for metric in ("win", "timeout", "shotdown", "crash")}
    easy = {
        opponent: statistics.median([100 * rate(skill[seed][opponent], "win") for seed in SEEDS])
        for opponent in ("pursuit_fire", "hold", "standoff_fire")
    }
    all_conserved = all(
        bool(skill[seed][opponent].get("weapon_event_conservation_ok"))
        for seed in SEEDS
        for opponent in OPPONENTS
    )
    all_legal = all(
        float(skill[seed][opponent]["legal_launch_acceptance_rate"]) == 1.0
        for seed in SEEDS
        for opponent in OPPONENTS
    )
    win_pass = statistics.median(new["win"]) >= 30
    timeout_pass = statistics.median(new["timeout"]) <= 20
    crash_pass = statistics.median(new["crash"]) <= 2
    easy_pass = min(easy.values()) >= 80
    accepted = win_pass and timeout_pass and crash_pass and easy_pass and all_conserved and all_legal
    decision = (
        "通过全部预注册门槛，可进入更长训练确认。"
        if accepted
        else "未通过全部预注册门槛，不应替换正式基准配置。"
    )
    report = f"""# 来袭相对 break 技能消融

## 判定

**{decision}** Evasive 胜率的三个配对变化为 {', '.join(f'{new['win'][i]-old['win'][i]:+.2f} pp' for i in range(3))}；中位数由 {statistics.median(old['win']):.2f}% 变为 {statistics.median(new['win']):.2f}%。超时率中位数由 {statistics.median(old['timeout']):.2f}% 变为 {statistics.median(new['timeout']):.2f}%，被击落率中位数由 {statistics.median(old['shotdown']):.2f}% 变为 {statistics.median(new['shotdown']):.2f}%。

该接口仅在来袭告警存在时重解释两个极端航向动作：动作 0 沿威胁反方向 break，动作 4 朝威胁方向切入；无告警及中间动作保持原语义。训练和公共评估中的 scripted Evasive 对手都已同步适配该语义。Pursuit/Hold/Standoff 胜率中位数为 {easy['pursuit_fire']:.2f}%/{easy['hold']:.2f}%/{easy['standoff_fire']:.2f}%。

## 验收门槛

| Criterion | Result | Pass |
|---|---:|:---:|
| Evasive win median ≥30% | {statistics.median(new['win']):.2f}% | {'Yes' if win_pass else 'No'} |
| Evasive timeout median ≤20% | {statistics.median(new['timeout']):.2f}% | {'Yes' if timeout_pass else 'No'} |
| Evasive crash median ≤2% | {statistics.median(new['crash']):.2f}% | {'Yes' if crash_pass else 'No'} |
| Easy-opponent median wins ≥80% | {easy['pursuit_fire']:.2f}% / {easy['hold']:.2f}% / {easy['standoff_fire']:.2f}% | {'Yes' if easy_pass else 'No'} |
| Legal launch =100% and event conservation | {'Yes' if all_legal and all_conserved else 'No'} | {'Yes' if all_legal and all_conserved else 'No'} |

## 精确结果

{table(baseline, skill)}

## 实验边界与质量控制

- `break_skill_ablation_v2` 是唯一纳入结论的数据。第一版运行时 scripted Evasive 对手未同步适配新动作语义，造成不公平且虚高的 seed=17 结果；该批次已停止并在 `runs/threat_obs_v2_break_skill_ablation/INVALID.md` 中永久标记排除。
- after checkpoint 在 ±90° baseline 上额外续训约 0.5M 动作，因此本实验检验的是“技能接口 + 0.5M 续训”配方，不是接口的纯静态因果效应。
- n=3 仅用于工程筛选，不进行显著性检验；每个 checkpoint 使用相同的 192 个双边平衡公共场景。

## 下一步

若超时仍是主导失败模式，应停止继续调整终局奖励幅度。下一轮应保持奖励、武器和 PPO 参数冻结，只检验带最短持续时间与退出滞回的威胁规避 option，使策略在避弹后必须恢复交战，而不是每个高层步独立切换或持续拖延。先用三种子各 0.5M 配对续训，并继续使用相同门槛。
"""
    (OUTPUT / "break-skill-analysis.md").write_text(report, encoding="utf-8")
    catalog_path = OUTPUT / "figure-catalog.md"
    catalog = catalog_path.read_text(encoding="utf-8")
    if "## Figure 8" not in catalog:
        catalog += """

## Figure 8 — Threat-relative break-skill continuation

- Purpose: test whether explicit incoming-threat-relative turn semantics improve evasive behavior.
- Data: paired seeds 17, 85, and 127; ±90° checkpoint before and after a 0.5M continuation with the break skill enabled.
- Encoding: paired Evasive win, timeout, shotdown, and crash episode rates; 192 common episodes per checkpoint.
- Main observation: see `break-skill-analysis.md`; the result is judged against pre-registered median gates rather than a selected seed.
- Caveat: the after checkpoint has 0.5M additional training, and the corrected v2 batch is the only valid batch.
"""
        catalog_path.write_text(catalog, encoding="utf-8")


def main() -> None:
    baseline = load(BASELINE_ROOTS)
    skill = load(BREAK_ROOTS)
    plot(baseline, skill)
    write_report(baseline, skill)
    print(json.dumps({"seeds": SEEDS, "output": str(OUTPUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
