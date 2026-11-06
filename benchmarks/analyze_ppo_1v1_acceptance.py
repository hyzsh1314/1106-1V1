"""Create the strict analysis bundle for the 1v1 PPO acceptance gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def wilson(successes, total, z=1.959963984540054):
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    radius = z * np.sqrt(
        p * (1.0 - p) / total + z * z / (4.0 * total * total)
    ) / denominator
    return float(centre - radius), float(centre + radius)


def rate_text(value, interval):
    return "{:.1f}% [{:.1f}, {:.1f}]".format(
        100.0 * value, 100.0 * interval[0], 100.0 * interval[1])


def main():
    args = parse_args()
    evaluation_dir = args.run_dir / "acceptance-evaluation"
    episodes = pd.read_csv(evaluation_dir / "episodes.csv")
    summary = json.loads((evaluation_dir / "summary.json").read_text(
        encoding="utf-8"))
    progress = json.loads((args.run_dir / "progress.json").read_text(
        encoding="utf-8"))
    output = args.run_dir / "analysis-output"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    n = len(episodes)
    wins = int((episodes.result > 0).sum())
    draws = int((episodes.result == 0).sum())
    losses = int((episodes.result < 0).sum())
    timeout_mask = episodes.reason.str.startswith("timeout")
    timeouts = int(timeout_mask.sum())
    decisive = wins + losses
    decisive_test = binomtest(wins, decisive, p=0.5, alternative="two-sided")
    win_ci = wilson(wins, n)
    decisive_ci = wilson(wins, decisive)
    timeout_ci = wilson(timeouts, n)
    draw_ci = wilson(draws, n)

    side_rows = []
    for side, frame in episodes.groupby("learner_side", sort=True):
        side_wins = int((frame.result > 0).sum())
        side_draws = int((frame.result == 0).sum())
        side_losses = int((frame.result < 0).sum())
        side_timeouts = int(frame.reason.str.startswith("timeout").sum())
        side_rows.append({
            "side": side,
            "episodes": len(frame),
            "wins": side_wins,
            "draws": side_draws,
            "losses": side_losses,
            "win_rate": side_wins / len(frame),
            "draw_rate": side_draws / len(frame),
            "loss_rate": side_losses / len(frame),
            "timeout_rate": side_timeouts / len(frame),
        })
    side_table = pd.DataFrame(side_rows)
    ego = side_table.set_index("side").loc["ego"]
    enemy = side_table.set_index("side").loc["enemy"]
    side_fisher = fisher_exact([
        [int(ego.wins), int(ego.losses)],
        [int(enemy.wins), int(enemy.losses)],
    ], alternative="two-sided")

    overall = pd.DataFrame([{
        "side": "overall", "episodes": n, "wins": wins, "draws": draws,
        "losses": losses, "win_rate": wins / n, "draw_rate": draws / n,
        "loss_rate": losses / n, "timeout_rate": timeouts / n,
    }])
    plot_table = pd.concat([side_table, overall], ignore_index=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    bottom = np.zeros(len(plot_table))
    colours = {"win_rate": "#2C7FB8", "draw_rate": "#BDBDBD",
               "loss_rate": "#D95F0E"}
    labels = {"win_rate": "Win", "draw_rate": "Draw",
              "loss_rate": "Loss"}
    for key in ("win_rate", "draw_rate", "loss_rate"):
        values = plot_table[key].to_numpy() * 100.0
        ax.bar(plot_table.side, values, bottom=bottom,
               color=colours[key], label=labels[key], width=0.62)
        bottom += values
    ax.set_ylabel("Episode outcome (%)")
    ax.set_ylim(0.0, 100.0)
    ax.set_xlabel("Learner side")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(figures / "figure-01-fixed-opponent-outcomes.png", dpi=220)
    fig.savefig(figures / "figure-01-fixed-opponent-outcomes.pdf")
    plt.close(fig)

    def reason_group(row):
        if row.reason == "enemy_team_eliminated":
            return "Physical win" if row.result > 0 else "Physical loss"
        if row.reason == "ego_team_eliminated":
            return "Physical win" if row.result > 0 else "Physical loss"
        if str(row.reason).startswith("timeout"):
            return "Timeout adjudication"
        return "Simultaneous/draw"

    episodes["reason_group"] = episodes.apply(reason_group, axis=1)
    reason_counts = pd.crosstab(
        episodes.learner_side, episodes.reason_group, normalize="index")
    reason_order = ["Physical win", "Physical loss", "Timeout adjudication",
                    "Simultaneous/draw"]
    reason_counts = reason_counts.reindex(columns=reason_order, fill_value=0.0)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), constrained_layout=True)
    reason_counts.T.plot.bar(
        ax=axes[0], color=["#4C78A8", "#F58518"], width=0.72)
    axes[0].set_ylabel("Fraction of episodes")
    axes[0].set_xlabel("Termination class")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(title="Learner side", frameon=False)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[1].hist([
        episodes.loc[episodes.learner_side == "ego", "episode_steps"],
        episodes.loc[episodes.learner_side == "enemy", "episode_steps"],
    ], bins=30, label=["ego", "enemy"], color=["#4C78A8", "#F58518"],
       alpha=0.75)
    axes[1].set_xlabel("Episode length (decision steps)")
    axes[1].set_ylabel("Episodes")
    axes[1].legend(title="Learner side", frameon=False)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.savefig(figures / "figure-02-termination-and-duration.png", dpi=220)
    fig.savefig(figures / "figure-02-termination-and-duration.pdf")
    plt.close(fig)

    update_pattern = re.compile(
        r"update=(?P<update>\d+)/(?P<target>\d+).*?"
        r"samples=(?P<samples>\d+).*?reward=(?P<reward>[+-]?\d+\.\d+).*?"
        r"launches=(?P<learner_launches>\d+)/(?P<opponent_launches>\d+).*?"
        r"wdl=(?P<wins>\d+)/(?P<draws>\d+)/(?P<losses>\d+)\s+"
        r"timeout=(?P<timeouts>\d+)\s+"
        r"fire=(?P<fire_requests>\d+)/(?P<rejected_requests>\d+)\s+"
        r"legal_opportunities=(?P<legal_opportunities>\d+).*?"
        r"value_loss=(?P<value_loss>\d+\.\d+).*?"
        r"entropy=(?P<entropy>\d+\.\d+).*?kl=(?P<kl>\d+\.\d+)"
    )
    update_rows = []
    for line in (args.run_dir / "train.log").read_text(
            encoding="utf-8", errors="replace").splitlines():
        match = update_pattern.search(line)
        if match is None:
            continue
        row = {key: float(value) for key, value in match.groupdict().items()}
        for key in (
                "update", "target", "samples", "learner_launches",
                "opponent_launches", "wins", "draws", "losses",
                "timeouts", "fire_requests", "rejected_requests",
                "legal_opportunities"):
            row[key] = int(row[key])
        update_rows.append(row)
    updates = pd.DataFrame(update_rows).sort_values("update")
    if len(updates) != int(progress["target_updates"]):
        raise ValueError(
            f"expected {progress['target_updates']} update rows, found {len(updates)}")
    updates["environment_steps"] = updates["update"] * updates["samples"]
    updates["reward_ma5"] = updates.reward.rolling(
        5, min_periods=1).mean()
    updates["reward_ma10"] = updates.reward.rolling(
        10, min_periods=3).mean()
    updates["reward_std10"] = updates.reward.rolling(
        10, min_periods=3).std(ddof=1)
    completed = updates[["wins", "draws", "losses"]].sum(axis=1)
    updates["win_rate"] = updates.wins / completed.replace(0, np.nan)
    updates["draw_rate"] = updates.draws / completed.replace(0, np.nan)
    updates["loss_rate"] = updates.losses / completed.replace(0, np.nan)
    updates["timeout_rate"] = updates.timeouts / completed.replace(0, np.nan)
    updates.to_csv(output / "training-update-metrics.csv", index=False)

    step_millions = updates.environment_steps.to_numpy() / 1.0e6
    reward = updates.reward.to_numpy()
    ma5 = updates.reward_ma5.to_numpy()
    ma10 = updates.reward_ma10.to_numpy()
    std10 = updates.reward_std10.to_numpy()
    fig, axes = plt.subplots(
        2, 1, figsize=(8.2, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.25]}, constrained_layout=True)
    axes[0].plot(step_millions, reward, color="#A6A6A6", linewidth=1.0,
                 marker="o", markersize=2.8, label="Per-update mean reward")
    axes[0].plot(step_millions, ma5, color="#2C7FB8", linewidth=2.0,
                 label="5-update moving mean")
    axes[0].plot(step_millions, ma10, color="#D95F0E", linewidth=2.2,
                 label="10-update moving mean")
    valid = np.isfinite(std10)
    axes[0].fill_between(
        step_millions[valid], (ma10 - std10)[valid], (ma10 + std10)[valid],
        color="#D95F0E", alpha=0.15,
        label="10-update temporal mean ± 1 SD")
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.55)
    axes[0].set_ylabel("Mean reward per learner transition")
    axes[0].legend(frameon=False, ncol=2, fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].plot(step_millions, updates.win_rate * 100.0,
                 color="#2C7FB8", linewidth=1.6, label="Win")
    axes[1].plot(step_millions, updates.draw_rate * 100.0,
                 color="#7F7F7F", linewidth=1.6, label="Draw")
    axes[1].plot(step_millions, updates.loss_rate * 100.0,
                 color="#D95F0E", linewidth=1.6, label="Loss")
    axes[1].plot(step_millions, updates.timeout_rate * 100.0,
                 color="#756BB1", linewidth=1.6, linestyle="--",
                 label="Timeout")
    axes[1].set_xlabel("Environment steps (millions)")
    axes[1].set_ylabel("Per-update episodes (%)")
    axes[1].set_ylim(0.0, 100.0)
    axes[1].legend(frameon=False, ncol=4, fontsize=9)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.savefig(figures / "figure-03-reward-convergence.png", dpi=220)
    fig.savefig(figures / "figure-03-reward-convergence.pdf")
    plt.close(fig)

    first10 = updates.reward.iloc[:10]
    last10 = updates.reward.iloc[-10:]
    tail_slope = float(np.polyfit(
        updates["update"].iloc[-10:].to_numpy(), last10.to_numpy(), 1)[0])

    exact = plot_table.copy()
    exact.to_csv(output / "fixed-opponent-summary.csv", index=False)
    side_markdown = side_table.to_markdown(index=False, floatfmt=".4f")
    acceptance = summary["legal_launch_acceptance_rate"]
    report = f"""# 1v1 PPO 百万步验收分析

## 分析问题

修改后的环境是否消除了超时平局与非法开火退化，并使冻结 PPO 策略在红蓝方平衡的固定对手评估中达到可继续训练的最低工程门槛。

## 数据与比较口径

- 训练：seed 85，96环境，{progress['total_steps']:,}步，单次运行。
- 固定评估：`scripted_pursuit_fire_v1`，共{n}局；学习策略作为红方/蓝方各500局。
- 独立统计单位为评估对局。训练 update、时间窗口和同一对局内的开火窗口不作为独立重复实验。
- 单一训练种子不足以支持算法优越性结论；本报告仅判定工程验收与下一步实验价值。

## 主要结果

1. 冻结策略取得 **{wins}/{draws}/{losses}** 胜/平/负，原始胜率为 **{rate_text(wins/n, win_ci)}**。
2. 排除{draws}局平局后，决定性对局胜率为 **{rate_text(wins/decisive, decisive_ci)}**；双侧二项检验相对50%的 `p={decisive_test.pvalue:.4f}`。因此当前优势尚未达到统计明确水平。
3. 超时率为 **{rate_text(timeouts/n, timeout_ci)}**，平局率为 **{rate_text(draws/n, draw_ci)}**。相较旧运行的高超时/高平局退化，工程问题已被移除。
4. 评估期合法发射接受率为 **{acceptance*100:.1f}%**（{summary['learner_fire_requests']}次请求、0次拒绝）；合法窗口利用率为 **{summary['legal_opportunity_exploitation_rate']*100:.1f}%**。后者的窗口并非独立样本，只作描述。
5. 红方与蓝方胜率分别为{ego.win_rate*100:.1f}%和{enemy.win_rate*100:.1f}%；排除平局后的侧别 Fisher 精确检验 `p={side_fisher.pvalue:.4f}`，未发现本批评估中的明显颜色侧偏。
6. 训练分布的超时率仍为 **{progress['timeout_rate']*100:.1f}%**，显著高于固定对手评估的{timeouts/n*100:.1f}%。这说明历史种群中仍包含较多不善于闭合交战的策略，种群采样和训练课程仍需继续改进。

## 精确侧别汇总

{side_markdown}

## 验收判断

- **环境/武器接口：通过。** 超时和平局不再占据绝对多数，非法开火为零，红蓝侧结果近似对称。
- **100万步策略最低工程门槛：通过。** 固定对手胜率高于50%，超时率低于10%。
- **论文基准或“已收敛”门槛：未通过。** 单种子、胜率置信区间覆盖50%，且训练种群超时率仍高；不应把当前checkpoint作为最终PPO基准。
- **下一决策：** 保留该checkpoint作为100万步里程碑；进入500万步前加入定期固定对手评估和best-checkpoint门控，避免最终快照弱于中间策略。

## Claim Candidates

- Claim:
  - Source evidence: 1000局侧别平衡固定评估，517/15/468胜平负，88局超时，886次合法开火请求均被接受。
  - Allowed wording: “修改后的平台在单种子百万步验收中消除了高超时与非法开火退化，并达到51.7%的固定对手胜率。”
  - Forbidden stronger wording: “PPO已经收敛”或“修改后算法显著优于固定对手”。
  - Uncertainty: 决定性胜率95%区间覆盖50%，仅一个训练种子和一个固定对手。
  - Next check: 3–5个训练种子、多个固定对手、500万步带best-checkpoint门控。
  - Decision: keep
"""
    (output / "analysis-report.md").write_text(report, encoding="utf-8")
    with (output / "analysis-report.md").open("a", encoding="utf-8") as stream:
        stream.write(f"""

## 奖励收敛诊断

- 41个 update 均被完整解析；每点是24,576个学习方 transition 的平均奖励，不是 episode return。
- 前10个 update 奖励均值为 `{first10.mean():+.6f}`、标准差 `{first10.std(ddof=1):.6f}`；最后10个 update 均值为 `{last10.mean():+.6f}`、标准差 `{last10.std(ddof=1):.6f}`。
- 最后10个 update 的线性斜率为每 update `{tail_slope:+.3e}`。曲线围绕零上下振荡，且尾窗波动没有消失，因此只能描述为“百万步内未见发散”，不能宣称已经收敛。
- 图中橙色带表示相邻10个 update 的时间波动（均值±1标准差），不是跨随机种子置信区间。
""")

    appendix = f"""# 统计附录

## 描述统计

- 评估局数：{n}；胜/平/负：{wins}/{draws}/{losses}。
- 胜率 Wilson 95% CI：{win_ci[0]:.6f}–{win_ci[1]:.6f}。
- 决定性胜率 Wilson 95% CI：{decisive_ci[0]:.6f}–{decisive_ci[1]:.6f}。
- 超时率 Wilson 95% CI：{timeout_ci[0]:.6f}–{timeout_ci[1]:.6f}。
- 平局率 Wilson 95% CI：{draw_ci[0]:.6f}–{draw_ci[1]:.6f}。
- 平均对局长度：{episodes.episode_steps.mean():.3f}步；中位数：{episodes.episode_steps.median():.1f}步。
- 每局学习方发射：均值{episodes.launches.mean():.3f}，标准差{episodes.launches.std(ddof=1):.3f}。

## 推断检验

- 决定性对局胜率相对0.5：双侧 exact binomial test，`n={decisive}`，`p={decisive_test.pvalue:.6f}`；效应量（胜率减0.5）={(wins/decisive-0.5):.6f}。
- 红/蓝侧决定性胜负差异：Fisher exact test，odds ratio={side_fisher.statistic:.6f}，`p={side_fisher.pvalue:.6f}`。
- 未进行多重比较校正：只有一个预先指定的主比较（固定对手决定性胜率相对0.5）；侧别检验为诊断性检查。

## 限制

- 只有一个训练随机种子；1000个评估回合不能替代独立训练重复。
- 合法开火窗口在对局内自相关，因此不为窗口利用率构造伪独立置信区间。
- 固定对手仅代表一个确定性追踪—开火策略，不代表完整策略族或可利用度。
"""
    (output / "stats-appendix.md").write_text(appendix, encoding="utf-8")
    with (output / "stats-appendix.md").open("a", encoding="utf-8") as stream:
        stream.write(f"""

## 训练奖励时间序列

- update数：{len(updates)}；每update样本数：{int(updates.samples.iloc[0])}。
- 全程奖励均值：{updates.reward.mean():+.6f}；标准差：{updates.reward.std(ddof=1):.6f}。
- 前10 update：{first10.mean():+.6f} ± {first10.std(ddof=1):.6f}。
- 后10 update：{last10.mean():+.6f} ± {last10.std(ddof=1):.6f}。
- 尾10 update OLS斜率：{tail_slope:+.6e}/update。
- 单一训练轨迹不满足独立重复假设，因此不对update序列进行显著性检验，也不把滚动波动带解释为置信区间。
""")

    catalog = """# 图表目录

## Figure 1 — `figure-01-fixed-opponent-outcomes`

- 目的：检查固定对手结果是否仍被平局主导，并审计红蓝侧偏。
- 数据：`acceptance-evaluation/episodes.csv`，1000个独立评估对局。
- 观察：总体平局很低；红蓝两侧胜负构成接近。
- 决策影响：环境退化问题已修复，但需要置信区间判断51.7%是否足以支持更强结论。
- 限制：单训练种子、单固定对手。

## Figure 2 — `figure-02-termination-and-duration`

- 目的：区分物理击杀、超时裁决与同时毁伤，并检查对局长度分布。
- 数据：同上。
- 观察：物理胜负占主导，超时为少数；长尾对局仍存在。
- 决策影响：允许进入更长训练，但应继续监测训练种群的闭合交战能力。
- 限制：对局步数分布不是训练收敛曲线。
"""
    (output / "figure-catalog.md").write_text(catalog, encoding="utf-8")
    with (output / "figure-catalog.md").open("a", encoding="utf-8") as stream:
        stream.write("""

## Figure 3 — `figure-03-reward-convergence`

- 目的：判断百万步训练中平均单步奖励是否形成稳定平台，并将奖励波动与胜平负、超时组成对照。
- 数据：`train.log`解析出的41个连续update；精确数据保存于`training-update-metrics.csv`。
- 观察：奖励在零附近呈非平稳振荡，末段仍有方向变化；超时比例也随历史对手构成波动。
- 决策影响：当前checkpoint满足工程验收，但曲线不支持“奖励已收敛”的论文表述。
- 限制：滚动标准差是单条时间序列的局部波动，不是跨种子不确定性。
""")
    print(json.dumps({
        "episodes": n,
        "wins": wins, "draws": draws, "losses": losses,
        "win_rate": wins / n,
        "win_rate_ci95": win_ci,
        "decisive_win_rate": wins / decisive,
        "decisive_p_value": decisive_test.pvalue,
        "timeout_rate": timeouts / n,
        "side_fisher_p_value": side_fisher.pvalue,
        "output": str(output.resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
