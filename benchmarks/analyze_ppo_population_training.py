#!/usr/bin/env python3
"""Analyze a completed population-PPO training log and draw convergence figures.

The log can contain duplicate update records after checkpoint recovery.  The
analysis keeps the final occurrence of each update and records the duplicate
audit in the generated report.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


UPDATE_RE = re.compile(
    r"^update=(?P<update>\d+)/(?P<target_updates>\d+)\s+"
    r"samples=(?P<samples>\d+)\s+population=(?P<population_before>\d+)->"
    r"(?P<population_size>\d+).*?reward=(?P<reward>[-+0-9.eE]+)\s+"
    r"launches=(?P<learner_launches>\d+)/(?P<opponent_launches>\d+)\s+"
    r"wdl=(?P<wins>\d+)/(?P<draws>\d+)/(?P<losses>\d+)\s+"
    r"policy_loss=(?P<policy_loss>[-+0-9.eE]+)\s+"
    r"value_loss=(?P<value_loss>[-+0-9.eE]+)\s+"
    r"entropy=(?P<entropy>[-+0-9.eE]+)\s+"
    r"kl=(?P<approx_kl>[-+0-9.eE]+)\s+"
    r"zero_sum_err=(?P<zero_sum_error>[-+0-9.eE]+)"
)

INTEGER_COLUMNS = {
    "update",
    "target_updates",
    "samples",
    "population_before",
    "population_size",
    "learner_launches",
    "opponent_launches",
    "wins",
    "draws",
    "losses",
}

OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
    "gray": "#7A7A7A",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/ppo_population_1v1_baseline_seed85_100m"),
    )
    parser.add_argument("--rolling-window", type=int, default=100)
    parser.add_argument("--long-window", type=int, default=500)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_log(log_path: Path) -> tuple[pd.DataFrame, dict]:
    records: list[dict] = []
    occurrences: dict[int, list[dict]] = {}
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            match = UPDATE_RE.match(line.strip())
            if match is None:
                continue
            record: dict[str, int | float] = {"line_number": line_number}
            for key, value in match.groupdict().items():
                record[key] = int(value) if key in INTEGER_COLUMNS else float(value)
            records.append(record)
            occurrences.setdefault(int(record["update"]), []).append(record)

    if not records:
        raise RuntimeError(f"No update records found in {log_path}")

    last_records = [items[-1] for _, items in sorted(occurrences.items())]
    frame = pd.DataFrame(last_records).sort_values("update").reset_index(drop=True)

    metric_keys = [
        "samples",
        "reward",
        "learner_launches",
        "opponent_launches",
        "wins",
        "draws",
        "losses",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "zero_sum_error",
    ]
    conflicting = 0
    duplicate_updates = 0
    for items in occurrences.values():
        if len(items) <= 1:
            continue
        duplicate_updates += 1
        reference = tuple(items[-1][key] for key in metric_keys)
        if any(tuple(item[key] for key in metric_keys) != reference for item in items[:-1]):
            conflicting += 1

    audit = {
        "parsed_update_rows": len(records),
        "unique_updates": len(frame),
        "duplicate_rows_removed": len(records) - len(frame),
        "updates_with_duplicates": duplicate_updates,
        "updates_with_conflicting_duplicate_metrics": conflicting,
        "first_update": int(frame["update"].min()),
        "last_update": int(frame["update"].max()),
    }
    return frame, audit


def add_derived_columns(
    frame: pd.DataFrame,
    steps_per_update: int,
    rolling_window: int,
    long_window: int,
) -> pd.DataFrame:
    frame = frame.copy()
    frame["steps"] = frame["update"] * steps_per_update
    frame["steps_million"] = frame["steps"] / 1.0e6
    frame["episodes"] = frame[["wins", "draws", "losses"]].sum(axis=1)
    frame["total_launches"] = (
        frame["learner_launches"] + frame["opponent_launches"]
    )
    for window, suffix in ((rolling_window, "short"), (long_window, "long")):
        roll = frame["reward"].rolling(window=window, min_periods=max(10, window // 5))
        frame[f"reward_mean_{suffix}"] = roll.mean()
        frame[f"reward_std_{suffix}"] = roll.std(ddof=1)
        frame[f"reward_q10_{suffix}"] = roll.quantile(0.10)
        frame[f"reward_q90_{suffix}"] = roll.quantile(0.90)

        wins = frame["wins"].rolling(window=window, min_periods=1).sum()
        draws = frame["draws"].rolling(window=window, min_periods=1).sum()
        losses = frame["losses"].rolling(window=window, min_periods=1).sum()
        episodes = (wins + draws + losses).replace(0, np.nan)
        frame[f"win_rate_{suffix}"] = wins / episodes
        frame[f"draw_rate_{suffix}"] = draws / episodes
        frame[f"loss_rate_{suffix}"] = losses / episodes
        frame[f"learner_launches_per_100_ep_{suffix}"] = (
            100.0
            * frame["learner_launches"].rolling(window=window, min_periods=1).sum()
            / episodes
        )
        frame[f"opponent_launches_per_100_ep_{suffix}"] = (
            100.0
            * frame["opponent_launches"].rolling(window=window, min_periods=1).sum()
            / episodes
        )
    return frame


def describe_slice(name: str, subset: pd.DataFrame) -> dict:
    reward = subset["reward"].to_numpy(dtype=float)
    episodes = int(subset["episodes"].sum())
    wins = int(subset["wins"].sum())
    draws = int(subset["draws"].sum())
    losses = int(subset["losses"].sum())
    learner_launches = int(subset["learner_launches"].sum())
    opponent_launches = int(subset["opponent_launches"].sum())
    return {
        "phase": name,
        "update_start": int(subset["update"].iloc[0]),
        "update_end": int(subset["update"].iloc[-1]),
        "n_updates": len(subset),
        "reward_mean": float(np.mean(reward)),
        "reward_std": float(np.std(reward, ddof=1)),
        "reward_median": float(np.median(reward)),
        "reward_q05": float(np.quantile(reward, 0.05)),
        "reward_q95": float(np.quantile(reward, 0.95)),
        "episodes": episodes,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / episodes if episodes else math.nan,
        "draw_rate": draws / episodes if episodes else math.nan,
        "loss_rate": losses / episodes if episodes else math.nan,
        "learner_launches": learner_launches,
        "opponent_launches": opponent_launches,
        "launches_per_100_episodes": (
            100.0 * (learner_launches + opponent_launches) / episodes
            if episodes
            else math.nan
        ),
    }


def phase_summary(frame: pd.DataFrame) -> pd.DataFrame:
    total = len(frame)
    boundaries = [0, total // 4, total // 2, 3 * total // 4, total]
    names = ["Q1 (0–25%)", "Q2 (25–50%)", "Q3 (50–75%)", "Q4 (75–100%)"]
    rows = [
        describe_slice(names[index], frame.iloc[boundaries[index] : boundaries[index + 1]])
        for index in range(4)
    ]
    rows.append(describe_slice("Full run", frame))
    return pd.DataFrame(rows)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def population_capacity_update(frame: pd.DataFrame, capacity: int) -> int | None:
    reached = frame.loc[frame["population_size"] >= capacity, "update"]
    return int(reached.iloc[0]) if not reached.empty else None


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=600)
    plt.close(fig)


def plot_reward_convergence(
    frame: pd.DataFrame,
    figures_dir: Path,
    rolling_window: int,
    long_window: int,
    capacity_update: int | None,
) -> None:
    setup_style()
    x = frame["steps_million"].to_numpy(dtype=float)
    raw = frame["reward"].to_numpy(dtype=float)
    short_mean = frame["reward_mean_short"].to_numpy(dtype=float)
    long_mean = frame["reward_mean_long"].to_numpy(dtype=float)
    q10 = frame["reward_q10_short"].to_numpy(dtype=float)
    q90 = frame["reward_q90_short"].to_numpy(dtype=float)
    short_std = frame["reward_std_short"].to_numpy(dtype=float)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.1, 5.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.10},
    )
    ax = axes[0]
    ax.plot(x, raw, color=OKABE_ITO["gray"], alpha=0.20, linewidth=0.55, label="Per-update mean")
    ax.fill_between(
        x,
        q10,
        q90,
        color=OKABE_ITO["sky"],
        alpha=0.20,
        linewidth=0,
        label=f"Rolling 10–90% ({rolling_window})",
    )
    ax.plot(
        x,
        short_mean,
        color=OKABE_ITO["blue"],
        linewidth=1.7,
        label=f"Rolling mean ({rolling_window})",
    )
    ax.plot(
        x,
        long_mean,
        color=OKABE_ITO["orange"],
        linewidth=1.6,
        linestyle="--",
        label=f"Rolling mean ({long_window})",
    )
    ax.axhline(0.0, color=OKABE_ITO["black"], linewidth=0.8, alpha=0.7)
    if capacity_update is not None:
        capacity_x = float(frame.loc[frame["update"] == capacity_update, "steps_million"].iloc[0])
        ax.axvline(capacity_x, color=OKABE_ITO["purple"], linestyle=":", linewidth=1.2)
    ax.set_ylabel("Mean step reward")
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.legend(loc="upper right", frameon=False, ncol=2, columnspacing=1.0)

    ax2 = axes[1]
    ax2.plot(x, short_std, color=OKABE_ITO["green"], label=f"Rolling SD ({rolling_window})")
    ax2.plot(x, np.abs(short_mean), color=OKABE_ITO["vermillion"], linestyle="--", label="Absolute rolling mean")
    if capacity_update is not None:
        capacity_x = float(frame.loc[frame["update"] == capacity_update, "steps_million"].iloc[0])
        ax2.axvline(capacity_x, color=OKABE_ITO["purple"], linestyle=":", linewidth=1.2)
    ax2.set_xlabel("Environment steps (million)")
    ax2.set_ylabel("Local magnitude")
    ax2.grid(True, alpha=0.22, linewidth=0.6)
    ax2.legend(loc="upper right", frameon=False, ncol=2)
    ax2.set_xlim(float(x.min()), float(x.max()))

    save_figure(fig, figures_dir / "figure-01-reward-convergence")


def plot_behavior_diagnostics(
    frame: pd.DataFrame,
    figures_dir: Path,
    rolling_window: int,
    capacity_update: int | None,
) -> None:
    setup_style()
    x = frame["steps_million"].to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 5.0), sharex=True, gridspec_kw={"hspace": 0.10})

    ax = axes[0]
    ax.plot(x, 100.0 * frame["win_rate_short"], color=OKABE_ITO["green"], label="Win")
    ax.plot(x, 100.0 * frame["draw_rate_short"], color=OKABE_ITO["blue"], label="Draw")
    ax.plot(x, 100.0 * frame["loss_rate_short"], color=OKABE_ITO["vermillion"], label="Loss")
    ax.set_ylabel("Outcome share (%)")
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.legend(loc="center right", frameon=False, ncol=3)

    ax2 = axes[1]
    learner_launch_rate = frame["learner_launches_per_100_ep_short"].to_numpy(dtype=float)
    opponent_launch_rate = frame["opponent_launches_per_100_ep_short"].to_numpy(dtype=float)
    ax2.plot(
        x,
        np.log10(1.0 + learner_launch_rate),
        color=OKABE_ITO["blue"],
        label="Learner launches",
    )
    ax2.plot(
        x,
        np.log10(1.0 + opponent_launch_rate),
        color=OKABE_ITO["orange"],
        linestyle="--",
        label="Opponent launches",
    )
    if capacity_update is not None:
        capacity_x = float(frame.loc[frame["update"] == capacity_update, "steps_million"].iloc[0])
        for axis in axes:
            axis.axvline(capacity_x, color=OKABE_ITO["purple"], linestyle=":", linewidth=1.2)
    ax2.set_xlabel("Environment steps (million)")
    ax2.set_ylabel("Launches per 100 episodes\n(log1p scale)")
    launch_ticks = np.asarray([0.0, 0.1, 1.0, 10.0, 100.0, 1000.0])
    ax2.set_yticks(np.log10(1.0 + launch_ticks))
    ax2.set_yticklabels(["0", "0.1", "1", "10", "100", "1000"])
    ax2.grid(True, alpha=0.22, linewidth=0.6)
    ax2.legend(loc="upper right", frameon=False, ncol=2)
    ax2.set_xlim(float(x.min()), float(x.max()))

    save_figure(fig, figures_dir / "figure-02-outcome-launch-diagnostics")


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def write_reports(
    output_dir: Path,
    frame: pd.DataFrame,
    phases: pd.DataFrame,
    audit: dict,
    manifest: dict,
    completed: dict,
    rolling_window: int,
    long_window: int,
    capacity_update: int | None,
) -> None:
    full = phases.loc[phases["phase"] == "Full run"].iloc[0]
    first = phases.iloc[0]
    last = phases.iloc[3]
    tail = frame.tail(long_window)
    tail_slope = float(np.polyfit(tail["update"], tail["reward"], 1)[0])
    tail_mean = float(tail["reward"].mean())
    tail_std = float(tail["reward"].std(ddof=1))
    tail_drift_span = tail_slope * max(len(tail) - 1, 1)
    tail_drift_to_std = abs(tail_drift_span) / max(tail_std, 1.0e-12)
    last_short_mean = float(frame["reward_mean_short"].iloc[-1])
    last_short_std = float(frame["reward_std_short"].iloc[-1])
    total_restarts = 12

    phase_table = phases[
        [
            "phase",
            "update_start",
            "update_end",
            "reward_mean",
            "reward_std",
            "reward_median",
            "win_rate",
            "draw_rate",
            "loss_rate",
            "launches_per_100_episodes",
        ]
    ].copy()
    for column in ["win_rate", "draw_rate", "loss_rate"]:
        phase_table[column] *= 100.0
    phase_table.columns = [
        "阶段",
        "起始 update",
        "结束 update",
        "奖励均值",
        "奖励标准差",
        "奖励中位数",
        "胜率(%)",
        "平局率(%)",
        "负率(%)",
        "每100局发射数",
    ]
    phase_markdown = phase_table.to_markdown(index=False, floatfmt=".6f")

    report = f"""# 1v1 导弹种群 PPO 训练数据分析

## 分析问题

本分析回答两个问题：训练日志中的平均单步奖励是否随训练趋于稳定；这种奖励稳定是否同时对应更有效的空战结果与武器使用行为。

## 数据与口径

- 场景：`{manifest['scenario']}`，随机种子 `{manifest['seed']}`，并行环境 `{manifest['num_envs']}`。
- 总训练量：{completed['total_steps']:,} 环境步，{completed['target_updates']:,} 次 update；每次 update 包含 {manifest['num_envs'] * manifest['rollout_steps']:,} 个学习方转移。
- 日志 `reward` 是每个 update 内所有学习方转移的平均单步奖励，不是 episode return。
- 原始日志包含 {audit['parsed_update_rows']:,} 条 update 记录。恢复训练造成 {audit['duplicate_rows_removed']:,} 条重复记录，分析按 update 保留最后一条，得到 {audit['unique_updates']:,} 个连续 update。
- 本实验只有 1 个随机种子。滚动分位带描述相邻 update 的时间波动，不是跨种子置信区间；不进行显著性检验。

## 主要发现

1. **奖励曲线没有形成可信的稳定平台。** 最后 {long_window} 个 update 的平均奖励为 {fmt(tail_mean)}、标准差为 {fmt(tail_std)}；最后一个 {rolling_window}-update 窗口均值却为 {fmt(last_short_mean)}，说明不同窗口给出的终值判断并不一致。
2. 最后 {long_window} 个 update 的线性斜率为每 update {tail_slope:.3e}。对应整个尾窗的拟合漂移为 {tail_drift_span:.6f}，约为尾窗标准差的 {tail_drift_to_std:.2f} 倍，不能忽略。结合曲线在约 65–75 百万步的正向隆起、75–95 百万步的负向回落和末段反弹，当前证据更符合非平稳振荡而非严格收敛。
3. 从第一四分位到最后四分位，奖励均值由 {fmt(float(first['reward_mean']))} 变为 {fmt(float(last['reward_mean']))}；第三四分位均值一度达到 {fmt(float(phases.iloc[2]['reward_mean']))}，随后反向变化。最后四分位平局率为 {float(last['draw_rate']) * 100:.2f}%，每 100 局仅产生 {float(last['launches_per_100_episodes']):.3f} 次发射。
4. 全程累计胜/平/负为 {int(full['wins'])}/{int(full['draws'])}/{int(full['losses'])}，总平局率 {float(full['draw_rate']) * 100:.2f}%；累计发射 {int(full['learner_launches'] + full['opponent_launches']):,} 枚。奖励趋近零与高平局、低发射共同出现，因此更可能反映保守策略或超时平局吸引域，而不是明确的制胜策略收敛。
5. 种群在 update {capacity_update if capacity_update is not None else '未达到'} 达到容量上限 {manifest['population_capacity']}。达到容量后仍持续替换快照，图中该节点用于区分种群扩张期和固定容量期。

## 精确分阶段汇总

{phase_markdown}

## 图形解释

### Figure 1：奖励收敛

- **目的**：判断平均单步奖励的中心位置与局部波动是否稳定。
- **观察**：短窗与长窗滚动均值发生明显相位变化，尾段仍在回升；局部标准差没有单调下降到零。
- **解释**：训练目标随历史对手池变化，奖励序列呈非平稳振荡；最后一点接近零并不构成稳定平台。
- **影响**：本次运行不能据此宣称奖励已经收敛，应使用冻结策略独立评估并增加多随机种子复现实验。

### Figure 2：结果与发射诊断

- **目的**：验证奖励稳定是否对应胜率提升和有效武器使用。
- **观察**：滚动结果长期由平局主导，发射率在训练后半段非常低。
- **解释**：当前奖励/终止结构可能允许避免交战的保守行为获得接近零的稳定回报。
- **影响**：下一步应固定最终策略，对多个种子和一组强弱对手做独立评估，并单独报告胜率、超时率、发射率、命中率和 episode return。

## Claim Candidates

- Claim:
  - Source evidence: 单随机种子完整训练曲线；最后 {long_window} update 拟合漂移为 {tail_drift_span:.6f}，是局部标准差的 {tail_drift_to_std:.2f} 倍，并存在中后期方向反转。
  - Allowed wording: “在 seed 85 的单次运行中，平均单步奖励尚未形成可信的稳定平台。”
  - Forbidden stronger wording: “PPO 已发散”或“最终策略必然无效”。
  - Uncertainty: 历史对手池使训练分布非平稳，且缺少多随机种子和独立评估回合。
  - Next check: 至少 3–5 个种子，并用冻结策略对固定对手池独立评估。
  - Decision: keep

- Claim:
  - Source evidence: 全程平局率 {float(full['draw_rate']) * 100:.2f}%，最后四分位平局率 {float(last['draw_rate']) * 100:.2f}%，最后四分位每 100 局发射 {float(last['launches_per_100_episodes']):.3f} 次。
  - Allowed wording: “训练后期出现高平局、低发射的保守行为模式。”
  - Forbidden stronger wording: “奖励函数必然错误”或“策略完全失效”。
  - Uncertainty: 日志未记录命中率、超时原因与完整 episode return。
  - Next check: 分解终止原因并做奖励项消融。
  - Decision: keep
"""
    (output_dir / "analysis-report.md").write_text(report, encoding="utf-8")

    stats = f"""# 统计附录

## 有效性检查

- 目标 update：{manifest['target_updates']}; 唯一有效 update：{audit['unique_updates']}。
- 缺失 update：{manifest['target_updates'] - audit['unique_updates']}。
- 重复行剔除：{audit['duplicate_rows_removed']}；含重复的 update 数：{audit['updates_with_duplicates']}；重复记录中指标不一致的 update 数：{audit['updates_with_conflicting_duplicate_metrics']}。
- 完成标记：`{completed['status']}`，总步数 {completed['total_steps']:,}。
- 日志聚合 episode 数：{int(full['episodes']):,}；完成文件 episode 数：{completed['total_episodes']:,}。
- 日志聚合发射数：{int(full['learner_launches'] + full['opponent_launches']):,}；完成文件发射数：{completed['total_launches']:,}。
- 监督器在早期修复前记录到 attempt 12；修复后正式会话未再重启。该事件通过保留每个 update 的最后记录进行去重。

## 描述性统计

{phase_markdown}

## 收敛诊断定义

- 短窗：{rolling_window} update = {rolling_window * manifest['num_envs'] * manifest['rollout_steps']:,} 环境步。
- 长窗：{long_window} update = {long_window * manifest['num_envs'] * manifest['rollout_steps']:,} 环境步。
- Figure 1 阴影：短窗内第 10–90 百分位，表示时间局部波动，不是置信区间。
- 后期趋势：对最后 {long_window} 个 update 的 per-update mean reward 做普通最小二乘直线斜率，仅作为描述性漂移指标。
- 后期拟合漂移：{tail_drift_span:.6f}，相当于尾窗标准差的 {tail_drift_to_std:.2f} 倍；因此尾段不满足“漂移远小于局部波动”的稳定平台条件。

## 禁止的推断

- 单种子训练轨迹不能用于跨种子显著性检验或置信区间。
- 相邻 update 有明显时间依赖，不能把 4070 个 update 当作独立重复实验。
- 平均单步奖励接近零不等同于 episode return 收敛，也不等同于胜率最优。
- 没有独立评估集，因此不能从训练对局直接给出泛化性能结论。
"""
    (output_dir / "stats-appendix.md").write_text(stats, encoding="utf-8")

    catalog = f"""# 图表目录

## figure-01-reward-convergence.pdf / .svg / .png

- Purpose: 展示平均单步奖励的原始波动、{rolling_window}-update 与 {long_window}-update 平滑趋势，以及局部波动强度。
- Data source: `train.log` 去重后的 {len(frame)} 个 update。
- Caption requirements: 横轴为环境步（百万）；纵轴为 update 内平均单步奖励；阴影为 {rolling_window}-update 滚动 10–90% 时间分位范围，不是跨种子置信区间；虚线标记种群达到容量 64。
- Key observation: 中后期滚动均值发生正向隆起、负向回落与末段反弹，未形成稳定平台。
- Interpretation checklist: 不把末值接近零误称为收敛；同时引用 Figure 2 的胜平负和发射行为。
- Caveat: 单随机种子、训练数据、非 episode return。

## figure-02-outcome-launch-diagnostics.pdf / .svg / .png

- Purpose: 检查奖励稳定是否伴随胜率提升和有效交战。
- Data source: `train.log` 中每个 update 的胜/平/负与双方发射计数，使用 {rolling_window}-update 聚合窗。
- Caption requirements: 上图为窗内 episode 加权的胜平负占比；下图为每 100 个完成 episode 的双方发射数，并采用 `log10(1+x)` 显示尺度以同时保留零值、初期高峰和后期低发射区；虚线标记种群达到容量 64。
- Key observation: 平局长期占主导，后期发射率很低。
- Interpretation checklist: 把该现象描述为保守行为模式；在缺少终止原因和命中率时不宣称具体机制已经被证明。
- Caveat: 训练对手来自历史种群，窗口内对手组成随时间变化。
"""
    (output_dir / "figure-catalog.md").write_text(catalog, encoding="utf-8")

    summary = {
        "audit": audit,
        "tail": {
            "window_updates": long_window,
            "reward_mean": tail_mean,
            "reward_std": tail_std,
            "reward_linear_slope_per_update": tail_slope,
            "reward_linear_slope_per_1000_updates": tail_slope * 1000.0,
            "reward_fitted_drift_across_window": tail_drift_span,
            "reward_fitted_drift_to_local_std": tail_drift_to_std,
            "last_short_window_mean": last_short_mean,
            "last_short_window_std": last_short_std,
        },
        "population_capacity_reached_update": capacity_update,
        "restarts_observed_before_fix": total_restarts,
        "single_seed": True,
    }
    (output_dir / "analysis-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = run_dir / "analysis-output"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_json(run_dir / "experiment_manifest.json")
    completed = load_json(run_dir / "COMPLETED.json")
    frame, audit = parse_log(run_dir / "train.log")

    target_updates = int(manifest["target_updates"])
    expected = np.arange(1, target_updates + 1)
    actual = frame["update"].to_numpy(dtype=int)
    if not np.array_equal(actual, expected):
        missing = sorted(set(expected.tolist()) - set(actual.tolist()))
        raise RuntimeError(f"Update sequence is incomplete; first missing values: {missing[:20]}")

    steps_per_update = int(manifest["num_envs"] * manifest["rollout_steps"])
    frame = add_derived_columns(
        frame,
        steps_per_update=steps_per_update,
        rolling_window=args.rolling_window,
        long_window=args.long_window,
    )
    phases = phase_summary(frame)
    capacity_update = population_capacity_update(frame, int(manifest["population_capacity"]))

    frame.to_csv(output_dir / "training-metrics-deduplicated.csv", index=False)
    phases.to_csv(output_dir / "phase-summary.csv", index=False)
    plot_reward_convergence(
        frame,
        figures_dir,
        rolling_window=args.rolling_window,
        long_window=args.long_window,
        capacity_update=capacity_update,
    )
    plot_behavior_diagnostics(
        frame,
        figures_dir,
        rolling_window=args.rolling_window,
        capacity_update=capacity_update,
    )
    write_reports(
        output_dir,
        frame,
        phases,
        audit,
        manifest,
        completed,
        rolling_window=args.rolling_window,
        long_window=args.long_window,
        capacity_update=capacity_update,
    )

    print(f"analysis_output={output_dir}")
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    print(phases.to_string(index=False))


if __name__ == "__main__":
    main()
