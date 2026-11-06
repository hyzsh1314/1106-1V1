"""Analyze the staged ThreatObservation-v2 curriculum experiment.

The script reads immutable JSON/JSONL evaluation artifacts and writes a
reproducible analysis bundle (Markdown plus PNG/PDF figures).  Confidence
intervals describe episode-level evaluation uncertainty only; training has one
seed and is therefore summarized descriptively.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OPPONENTS = ("pursuit_fire", "hold", "evasive_fire", "standoff_fire")
OPPONENT_LABELS = {
    "pursuit_fire": "Pursuit",
    "hold": "Hold",
    "evasive_fire": "Evasive",
    "standoff_fire": "Standoff",
}
STAGES = (
    ("24D 0.5M", "runs/altitude_safety_1m_paired/update20-seed85-screen", False),
    ("V2 0.5M", "runs/threat_obs_v2_safety_screen_seed85_500k/eval", False),
    ("V2 1.0M", "runs/threat_obs_v2_safety_seed85_1m_continue/eval", False),
    ("Focused 1.5M", "runs/threat_obs_v2_evasive_curriculum_seed85_1p5m/eval", False),
    ("Blend 2.0M", "runs/threat_obs_v2_blended_curriculum_seed85_2m/eval", False),
)
METRIC_FILES = (
    "runs/threat_obs_v2_safety_screen_seed85_500k/metrics.jsonl",
    "runs/threat_obs_v2_safety_seed85_1m_continue/metrics.jsonl",
    "runs/threat_obs_v2_evasive_curriculum_seed85_1p5m/metrics.jsonl",
    "runs/threat_obs_v2_blended_curriculum_seed85_2m/metrics.jsonl",
)
SELECTED_RUN = "runs/threat_obs_v2_evasive_curriculum_seed85_1p5m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/threat_obs_v2_curriculum_analysis/analysis-output"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def crash_count(summary: dict) -> int:
    return int(summary.get("learner_death_causes", {}).get("crash", 0))


def load_stage_data(root: Path) -> dict[str, dict[str, dict]]:
    result: dict[str, dict[str, dict]] = {}
    for label, relative_root, _ in STAGES:
        result[label] = {}
        for opponent in OPPONENTS:
            result[label][opponent] = read_json(root / relative_root / opponent / "summary.json")
    return result


def load_metrics(root: Path) -> list[dict]:
    by_update: dict[int, dict] = {}
    for relative in METRIC_FILES:
        with (root / relative).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    by_update[int(record["update"])] = record
    return [by_update[index] for index in sorted(by_update)]


def moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    output = np.full(values.shape, np.nan, dtype=float)
    for index in range(values.size):
        start = max(0, index - window + 1)
        output[index] = np.mean(values[start : index + 1])
    return output


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "legend.frameon": False,
        }
    )


def plot_stage_progression(stage_data: dict[str, dict[str, dict]], output: Path) -> None:
    stage_labels = [stage[0] for stage in STAGES]
    x = np.arange(len(stage_labels))
    colors = {"win": "#0072B2", "timeout": "#E69F00", "crash": "#D55E00"}
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.8), sharex=True, sharey=True)
    for axis, opponent in zip(axes.flat, OPPONENTS):
        summaries = [stage_data[label][opponent] for label in stage_labels]
        values = {
            "win": [100.0 * item["win_rate"] for item in summaries],
            "timeout": [100.0 * item["timeout_rate"] for item in summaries],
            "crash": [100.0 * crash_count(item) / item["episodes"] for item in summaries],
        }
        axis.plot(x, values["win"], "o-", color=colors["win"], lw=2.0, label="Win")
        axis.plot(x, values["timeout"], "s--", color=colors["timeout"], lw=1.7, label="Timeout")
        axis.plot(x, values["crash"], "^:", color=colors["crash"], lw=1.7, label="Learner crash")
        axis.set_title(OPPONENT_LABELS[opponent])
        axis.set_ylim(-3, 103)
        axis.set_xticks(x, stage_labels, rotation=24, ha="right")
        axis.set_ylabel("Episode rate (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Fixed-opponent performance across staged training", y=1.045, fontsize=12)
    fig.tight_layout()
    save_figure(fig, output, "figure-01-fixed-opponent-progression")


def plot_training_dynamics(metrics: list[dict], output: Path) -> None:
    update = np.asarray([item["update"] for item in metrics], dtype=float)
    reward = 1000.0 * np.asarray([item["reward"] for item in metrics], dtype=float)
    entropy = np.asarray([item["metrics"]["entropy"] for item in metrics], dtype=float)
    kl = np.asarray([item["metrics"]["approx_kl"] for item in metrics], dtype=float)
    timeout = 100.0 * np.asarray(
        [item["timeouts"] / max(1, item["episodes_this_update"]) for item in metrics], dtype=float
    )
    panels = (
        (reward, "Rollout mean reward (×10³)", "#0072B2"),
        (entropy, "Policy entropy", "#009E73"),
        (kl, "Approximate KL", "#CC79A7"),
        (timeout, "Rollout timeout rate (%)", "#E69F00"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.4), sharex=True)
    for axis, (values, ylabel, color) in zip(axes.flat, panels):
        axis.plot(update, values, color=color, alpha=0.25, lw=1.0, label="Per update")
        axis.plot(update, moving_average(values), color=color, lw=2.0, label="5-update mean")
        for boundary in (21, 41, 61):
            axis.axvline(boundary + 0.5, color="#666666", lw=0.8, ls="--", alpha=0.65)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("PPO update")
    axes[0, 0].axhline(0.0, color="#333333", lw=0.8, alpha=0.6)
    axes[0, 0].legend(loc="best")
    axes[0, 0].text(
        0.02,
        0.04,
        "Near-zero reward is expected under zero-sum aggregation",
        transform=axes[0, 0].transAxes,
        fontsize=7.5,
    )
    fig.suptitle("Training diagnostics (V2 population → focused → blended)", fontsize=12)
    fig.tight_layout()
    save_figure(fig, output, "figure-02-training-dynamics")


def plot_selected_replication(primary: dict[str, dict], heldout: dict[str, dict], output: Path) -> None:
    x = np.arange(len(OPPONENTS))
    width = 0.34
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), sharex=True, sharey=True)
    definitions = (
        ("win", "Win rate", lambda item: int(item["wins"])),
        ("timeout", "Timeout rate", lambda item: int(item["timeouts"])),
        ("crash", "Learner crash rate", crash_count),
    )
    for axis, (_, title, counter) in zip(axes, definitions):
        for offset, dataset, label, color in (
            (-width / 2, primary, "Selection eval (n=96)", "#56B4E9"),
            (width / 2, heldout, "Held-out eval (n=192)", "#0072B2"),
        ):
            rates = []
            lower = []
            upper = []
            for opponent in OPPONENTS:
                item = dataset[opponent]
                count = counter(item)
                n = int(item["episodes"])
                lo, hi = wilson_interval(count, n)
                rate = count / n
                rates.append(100.0 * rate)
                lower.append(max(0.0, 100.0 * (rate - lo)))
                upper.append(max(0.0, 100.0 * (hi - rate)))
            axis.bar(x + offset, rates, width, color=color, label=label, alpha=0.9)
            axis.errorbar(
                x + offset,
                rates,
                yerr=np.asarray([lower, upper]),
                fmt="none",
                ecolor="#222222",
                elinewidth=0.9,
                capsize=2,
            )
        axis.set_title(title)
        axis.set_ylim(0, 105)
        axis.set_xticks(x, [OPPONENT_LABELS[item] for item in OPPONENTS], rotation=25, ha="right")
        axis.set_ylabel("Episode rate (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Selected 1.5M checkpoint: selection versus held-out seeds", y=1.12, fontsize=12)
    fig.tight_layout()
    save_figure(fig, output, "figure-03-selected-heldout-replication")


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def stage_table(stage_data: dict[str, dict[str, dict]]) -> str:
    lines = [
        "| Stage | Opponent | n | Win | Timeout | Crash | Legal launch |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stage, _, _ in STAGES:
        for opponent in OPPONENTS:
            item = stage_data[stage][opponent]
            n = int(item["episodes"])
            lines.append(
                f"| {stage} | {OPPONENT_LABELS[opponent]} | {n} | {percent(item['win_rate'])} | "
                f"{percent(item['timeout_rate'])} | {100.0 * crash_count(item) / n:.2f}% | "
                f"{percent(item['legal_launch_acceptance_rate'])} |"
            )
    return "\n".join(lines)


def evaluation_table(dataset_name: str, data: dict[str, dict]) -> str:
    lines = [
        f"### {dataset_name}",
        "",
        "| Opponent | Metric | Count/n | Rate | Episode-level Wilson 95% CI |",
        "|---|---|---:|---:|---:|",
    ]
    for opponent in OPPONENTS:
        item = data[opponent]
        n = int(item["episodes"])
        for metric, count in (
            ("Win", int(item["wins"])),
            ("Timeout", int(item["timeouts"])),
            ("Learner crash", crash_count(item)),
        ):
            lo, hi = wilson_interval(count, n)
            lines.append(
                f"| {OPPONENT_LABELS[opponent]} | {metric} | {count}/{n} | {100.0 * count / n:.2f}% | "
                f"[{100.0 * lo:.2f}%, {100.0 * hi:.2f}%] |"
            )
    return "\n".join(lines)


def side_audit(root: Path) -> str:
    csv_path = root / SELECTED_RUN / "heldout/evasive_fire/episodes.csv"
    groups: dict[str, list[dict[str, str]]] = {"ego": [], "enemy": []}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            groups[row["learner_side"]].append(row)
    lines = [
        "| Learner side | n | Wins | Losses | Draws | Timeouts | Crashes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for side in ("ego", "enemy"):
        rows = groups[side]
        lines.append(
            f"| {side} | {len(rows)} | {sum(r['label'] == 'win' for r in rows)} | "
            f"{sum(r['label'] == 'loss' for r in rows)} | {sum(r['label'] == 'draw' for r in rows)} | "
            f"{sum(r['reason'].startswith('timeout') for r in rows)} | "
            f"{sum(r['learner_death_cause'] == 'crash' for r in rows)} |"
        )
    return "\n".join(lines)


def write_reports(
    root: Path,
    output: Path,
    stage_data: dict[str, dict[str, dict]],
    primary: dict[str, dict],
    heldout: dict[str, dict],
    metrics: list[dict],
) -> None:
    v2_1m = stage_data["V2 1.0M"]
    focused = stage_data["Focused 1.5M"]
    blended = stage_data["Blend 2.0M"]
    evasive_gain = 100.0 * (focused["evasive_fire"]["win_rate"] - v2_1m["evasive_fire"]["win_rate"])
    standoff_gain = 100.0 * (focused["standoff_fire"]["win_rate"] - v2_1m["standoff_fire"]["win_rate"])
    blend_evasive = 100.0 * (blended["evasive_fire"]["win_rate"] - focused["evasive_fire"]["win_rate"])
    selected = root / SELECTED_RUN / "latest.pt"
    analysis = f"""# ThreatObservation V2 与对手课程实验分析

## 结论

当前应将 **Focused 1.5M** 保存为候选基准 checkpoint，而不是继续使用 2.0M blended checkpoint。它将 evasive_fire 胜率从 V2 1.0M 的 0.00% 提高到 31.25%（+{evasive_gain:.2f} 个百分点），并将 standoff_fire 从 18.75% 提高到 90.62%（+{standoff_gain:.2f} 个百分点）。随后恢复 75% population / 25% evasive 的混合训练，evasive_fire 反而下降 {abs(blend_evasive):.2f} 个百分点且超时率升至 60.42%，因此不支持“训练步数越多越好”。

独立 held-out 评估（每类 192 局，使用全新环境随机种子）复现了候选 checkpoint 的主要结果：Pursuit 100.00%、Hold 100.00%、Evasive 32.29%、Standoff 90.10%。Evasive 的 32.29% 与选择集的 31.25% 接近，说明该增益不是原 96 个评估局特有；但其 held-out 超时率仍为 55.21%，并出现 5/192（2.60%）学习方低高度坠毁，尚未达到“超时≤5%、坠毁≤2%”的最终验收标准。

奖励曲线接近零不等价于没有学习：本实现按红蓝零和聚合，正负回报会相互抵消。固定对手胜率、超时率、坠毁率和合法发射率比全局平均奖励更适合作为该平台的训练验收指标。图 2 仍保留奖励动态用于优化器诊断，但不把它当作性能收敛的单一证据。

## 证据链

1. 26D ThreatObservation V2 仅加入“来袭标志+紧迫度”后，Pursuit/Hold 很快提升，但 Evasive 仍为 0%，说明观测增强本身不足以产生规避能力。
2. 100% evasive_fire 的 0.5M focused curriculum 将 Evasive 提升至约 31%，同时保持其他三类对手 90% 左右或以上；课程分布是本轮有效干预。
3. Blended 继续训练发生回退，说明当前策略存在遗忘/分布折中，而不是稳定单调收敛。
4. 四类评估均保持合法发射接受率 100%，武器事件守恒检查通过；当前主要瓶颈已经从“发射流程”转向“规避场景的超时与低高度安全”。

## 阶段结果

{stage_table(stage_data)}

## Held-out 侧别审计（Evasive）

{side_audit(root)}

胜率在两侧分别为 28/96 和 34/96，差异不大；5 次学习方坠毁中 4 次发生在 learner=enemy 一侧，提示仍需用更多训练种子和侧别分层报告排查侧别相关风险，现有样本不足以断言系统性偏置。

## 下一步执行门槛

1. 冻结当前奖励 R1、26D 观测、动力学和武器参数，以 `{selected.as_posix()}` 作为候选，而不是 2.0M checkpoint。
2. 进行至少 3 个独立训练种子的同配置复现；每个种子都采用 1.0M population + 0.5M evasive-focused 两阶段课程，并在相同 held-out 协议下评估。
3. 预先设定主指标：Evasive 胜率中位数、Evasive 超时率、学习方坠毁率；门槛建议为胜率≥30%、超时≤20%、坠毁≤2%，且 Pursuit/Hold/Standoff 胜率均≥80%。
4. 若多种子仍保持约 30% 胜率但超时>20%，下一轮只研究“终局效率/规避后再攻击”的课程配比或层级策略，不再同时改奖励、导弹和动力学，以保持归因清晰。

## 统计边界

本轮训练只有 seed=85，不能进行训练种子层面的显著性检验，也不能据此宣称算法普遍优越。Wilson 区间仅量化固定 checkpoint 下有限评估局数的二项抽样不确定性；同一并行仿真批次内局间也可能存在弱相关，因此区间应视为近似描述。
"""
    (output / "analysis-report.md").write_text(analysis, encoding="utf-8")

    stats = f"""# Statistical appendix

## Analysis unit and uncertainty

- Training-level independent sample size: **1 seed (seed=85)**. All cross-stage comparisons are descriptive paired engineering comparisons, not inferential claims.
- Evaluation unit: one simulated episode, side-balanced within each opponent condition.
- Intervals: two-sided 95% Wilson score intervals for binomial episode outcomes. They do not capture training-seed variance.
- No null-hypothesis p-values are reported because the independent training-seed sample size is insufficient.

{evaluation_table('Selected checkpoint: selection evaluation', primary)}

{evaluation_table('Selected checkpoint: held-out evaluation', heldout)}

## Exact staged results

{stage_table(stage_data)}

## Optimizer endpoint

- Updates analyzed: {len(metrics)} (updates {metrics[0]['update']}–{metrics[-1]['update']}).
- Final approximate KL: {metrics[-1]['metrics']['approx_kl']:.6f}.
- Final entropy: {metrics[-1]['metrics']['entropy']:.4f}.
- Final value loss: {metrics[-1]['metrics']['value_loss']:.6f}.
- Weapon-event conservation: true in all fixed-opponent summaries.
"""
    (output / "stats-appendix.md").write_text(stats, encoding="utf-8")

    catalog = """# Figure catalog

## Figure 1 — Fixed-opponent performance across staged training

- Purpose: show which intervention changed each opponent-specific outcome.
- Data: five checkpoints, four scripted opponents, 96 side-balanced episodes per cell.
- Encoding: win, timeout, and learner-crash episode rates; all panels share a 0–100% scale.
- Main observation: V2 alone improves easy opponents, while the focused curriculum produces the first material evasive win rate and large standoff gain; blending then regresses evasive performance.
- Caveat: all checkpoints are from one training seed.

## Figure 2 — Training diagnostics

- Purpose: distinguish optimizer behavior from task performance.
- Data: all 81 PPO update records across the four V2 phases.
- Encoding: raw per-update values and trailing five-update means; dashed vertical lines separate 0.5M, 1.0M, focused 1.5M, and blended 2.0M phases.
- Main observation: entropy declines continuously and KL remains controlled, while rollout reward stays near zero by zero-sum construction and timeout behavior changes with the opponent curriculum.
- Caveat: rollout composition changes by phase, so curves are diagnostics rather than like-for-like evaluation.

## Figure 3 — Selected checkpoint replication

- Purpose: test whether checkpoint-selection results reproduce on unseen evaluation seeds.
- Data: 96 selection episodes and 192 held-out episodes per opponent.
- Encoding: rates with episode-level Wilson 95% intervals.
- Main observation: evasive win rate is stable near 32% and standoff near 90%, but evasive timeout and crash remain the principal failure modes.
- Caveat: both evaluations use the same trained checkpoint and therefore do not estimate training-seed variability.
"""
    (output / "figure-catalog.md").write_text(catalog, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    output = args.output_dir
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    stage_data = load_stage_data(root)
    metrics = load_metrics(root)
    primary = {
        opponent: read_json(root / SELECTED_RUN / "eval" / opponent / "summary.json")
        for opponent in OPPONENTS
    }
    heldout = {
        opponent: read_json(root / SELECTED_RUN / "heldout" / opponent / "summary.json")
        for opponent in OPPONENTS
    }
    plot_stage_progression(stage_data, output)
    plot_training_dynamics(metrics, output)
    plot_selected_replication(primary, heldout, output)
    write_reports(root, output, stage_data, primary, heldout, metrics)
    print(json.dumps({"output_dir": str(output), "updates": len(metrics)}, indent=2))


if __name__ == "__main__":
    main()
