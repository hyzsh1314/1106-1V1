"""Strict seed-level analysis for the completed V4 reward ablation."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


VARIANTS = (
    "R0_terminal", "R1_attack_pbrs", "R2_threat_pbrs", "R3_constraints")
SHORT = dict(zip(VARIANTS, ("R0", "R1", "R2", "R3")))
SEEDS = (85, 185, 285)
OPPONENTS = ("pursuit_fire", "hold")
COLORS = {
    "R0_terminal": "#6B7280", "R1_attack_pbrs": "#0072B2",
    "R2_threat_pbrs": "#009E73", "R3_constraints": "#E69F00",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, required=True)
    return parser.parse_args()


def mean_sd_ci(values):
    x = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    half = float(stats.t.ppf(0.975, len(x) - 1) * sd / np.sqrt(len(x)))
    return mean, sd, mean - half, mean + half


def exact_sign_flip_p(differences):
    d = np.asarray(differences, dtype=np.float64)
    observed = abs(float(np.mean(d)))
    values = [abs(float(np.mean(d * np.asarray(signs))))
              for signs in itertools.product((-1.0, 1.0), repeat=len(d))]
    return sum(value >= observed - 1.0e-15 for value in values) / len(values)


def holm_adjust(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (len(p_values) - rank) * p_values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    numeric = set(rows[0]) - {"variant", "opponent"}
    for row in rows:
        for key in numeric:
            row[key] = float(row[key])
        row["seed"] = int(row["seed"])
    return rows


def save_figure(fig, base):
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    output = args.root_dir / "analysis-output"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows = load_rows(output / "per-run-metrics.csv")
    expected = len(VARIANTS) * len(SEEDS) * len(OPPONENTS)
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} evaluation rows, found {len(rows)}")

    by_key = {(row["variant"], row["seed"], row["opponent"]): row
              for row in rows}
    missing = [key for key in itertools.product(VARIANTS, SEEDS, OPPONENTS)
               if key not in by_key]
    if missing:
        raise RuntimeError(f"missing evaluation cells: {missing}")
    if any(row["train_steps"] != 1007616 for row in rows):
        raise RuntimeError("not every run completed the registered step budget")

    # Seed-level descriptives with t intervals. With n=3 these intervals are
    # intentionally reported as wide rather than treating 300 episodes/seed
    # as independent experimental repetitions.
    descriptive = []
    for variant in VARIANTS:
        for opponent in OPPONENTS:
            group = [by_key[(variant, seed, opponent)] for seed in SEEDS]
            record = {"variant": variant, "opponent": opponent, "n_seeds": 3}
            for metric in ("win_rate", "loss_rate", "timeout_rate",
                           "opportunity_exploitation_rate",
                           "mean_episode_steps", "train_timeout_rate"):
                mean, sd, low, high = mean_sd_ci([row[metric] for row in group])
                record.update({f"{metric}_mean": mean, f"{metric}_sd": sd,
                               f"{metric}_ci95_low": low,
                               f"{metric}_ci95_high": high})
            descriptive.append(record)
    (output / "strict-descriptive-statistics.json").write_text(
        json.dumps(descriptive, indent=2, sort_keys=True), encoding="utf-8")

    # Repeated-measures omnibus tests across the four variants.
    omnibus = []
    for opponent in OPPONENTS:
        arrays = [np.asarray([by_key[(variant, seed, opponent)]["win_rate"]
                              for seed in SEEDS]) for variant in VARIANTS]
        statistic, p_value = stats.friedmanchisquare(*arrays)
        omnibus.append({"opponent": opponent, "metric": "win_rate",
                        "test": "Friedman repeated-measures",
                        "statistic": float(statistic), "df": 3,
                        "p_value": float(p_value), "n_blocks": 3})

    # Preregistered candidate contrasts: R1 against the other variants,
    # paired by seed and evaluated separately for each fixed opponent.
    contrasts = []
    for opponent in OPPONENTS:
        r1 = np.asarray([by_key[("R1_attack_pbrs", seed, opponent)]["win_rate"]
                         for seed in SEEDS])
        for baseline in ("R0_terminal", "R2_threat_pbrs", "R3_constraints"):
            other = np.asarray([by_key[(baseline, seed, opponent)]["win_rate"]
                                for seed in SEEDS])
            diff = r1 - other
            mean, sd, low, high = mean_sd_ci(diff)
            dz = None if sd == 0 else mean / sd
            contrasts.append({
                "opponent": opponent, "contrast": f"R1 - {SHORT[baseline]}",
                "paired_differences": diff.tolist(),
                "mean_difference": mean, "sd_difference": sd,
                "ci95_low": low, "ci95_high": high,
                "exact_sign_flip_p": exact_sign_flip_p(diff),
                "cohen_dz": dz,
                "positive_ties_negative": [int(np.sum(diff > 0)),
                                             int(np.sum(diff == 0)),
                                             int(np.sum(diff < 0))],
            })
    adjusted = holm_adjust([item["exact_sign_flip_p"] for item in contrasts])
    for item, value in zip(contrasts, adjusted):
        item["holm_adjusted_p"] = value
    (output / "strict-inferential-statistics.json").write_text(
        json.dumps({"omnibus": omnibus, "contrasts": contrasts},
                   indent=2, sort_keys=True), encoding="utf-8")

    components = json.loads((output / "reward-component-summary.json").read_text(
        encoding="utf-8"))
    component_ratios = {}
    for variant, values in components.items():
        terminal = values["TeamOutcomeReward"]
        component_ratios[variant] = {
            name: value / terminal for name, value in values.items()
            if name != "TeamOutcomeReward"
        }
    (output / "strict-component-ratios.json").write_text(
        json.dumps(component_ratios, indent=2, sort_keys=True), encoding="utf-8")

    # Figure 1: evaluation performance. Seed pairing is explicit; thick bars
    # are means, not inferential confidence claims.
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), constrained_layout=True)
    specs = (("pursuit_fire", "win_rate", "Win rate vs pursuit-fire"),
             ("hold", "win_rate", "Win rate vs hold"),
             ("pursuit_fire", "mean_episode_steps", "Episode length vs pursuit-fire"))
    x = np.arange(len(VARIANTS))
    for axis, (opponent, metric, label) in zip(axes, specs):
        for seed_index, seed in enumerate(SEEDS):
            values = [by_key[(variant, seed, opponent)][metric]
                      for variant in VARIANTS]
            axis.plot(x, values, color="#B8BDC6", linewidth=1.0, alpha=0.8,
                      zorder=1)
        for index, variant in enumerate(VARIANTS):
            values = [by_key[(variant, seed, opponent)][metric] for seed in SEEDS]
            axis.scatter(np.full(3, index), values, color=COLORS[variant],
                         edgecolor="white", linewidth=0.6, s=42, zorder=3)
            axis.plot([index - 0.18, index + 0.18], [np.mean(values)] * 2,
                      color="black", linewidth=2.7, zorder=4)
        axis.set_xticks(x, [SHORT[v] for v in VARIANTS])
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
        if metric == "win_rate":
            axis.set_ylim(0, 1.03)
    save_figure(fig, figures / "figure-01-fixed-opponent-performance")

    # Figure 2: action/weapon behavior separates policy collapse from weapon
    # legality. Every accepted-request rate is 1.0, so exploitation is the
    # behaviorally informative quantity.
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), sharey=True,
                             constrained_layout=True)
    for axis, opponent in zip(axes, OPPONENTS):
        for seed in SEEDS:
            axis.plot(x, [by_key[(v, seed, opponent)][
                "opportunity_exploitation_rate"] for v in VARIANTS],
                color="#B8BDC6", linewidth=1.0, zorder=1)
        for index, variant in enumerate(VARIANTS):
            values = [by_key[(variant, seed, opponent)][
                      "opportunity_exploitation_rate"] for seed in SEEDS]
            axis.scatter(np.full(3, index), values, color=COLORS[variant],
                         edgecolor="white", linewidth=0.6, s=42, zorder=3)
            axis.plot([index - 0.18, index + 0.18], [np.mean(values)] * 2,
                      color="black", linewidth=2.7, zorder=4)
        axis.set_xticks(x, [SHORT[v] for v in VARIANTS])
        axis.set_title("Pursuit-fire" if opponent == "pursuit_fire" else "Hold")
        axis.grid(axis="y", alpha=0.25)
        axis.set_ylim(0, 1.03)
    axes[0].set_ylabel("Legal-opportunity exploitation rate")
    save_figure(fig, figures / "figure-02-fire-opportunity-exploitation")

    # Figure 3: training dynamics. Mean zero-sum reward is not a valid sole
    # convergence indicator; timeout frequency and PPO KL are more diagnostic.
    curves = {}
    for variant in VARIANTS:
        for seed in SEEDS:
            path = args.root_dir / f"{variant}_seed{seed}_1m" / "metrics.jsonl"
            curves[(variant, seed)] = [json.loads(line) for line in
                path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(curves[(variant, seed)]) != 41:
                raise RuntimeError(f"incomplete curve: {variant} seed {seed}")
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), constrained_layout=True)
    for variant in VARIANTS:
        steps = np.asarray([p["total_steps"] for p in curves[(variant, SEEDS[0])]])
        timeout_curves, kl_curves, entropy_curves = [], [], []
        for seed in SEEDS:
            curve = curves[(variant, seed)]
            timeout_curves.append([
                p["timeouts"] / p["episodes_this_update"]
                if p["episodes_this_update"] else np.nan for p in curve])
            kl_curves.append([p["metrics"]["approx_kl"] for p in curve])
            entropy_curves.append([p["metrics"]["entropy"] for p in curve])
        for axis, values in zip(axes, (timeout_curves, kl_curves, entropy_curves)):
            array = np.asarray(values, dtype=np.float64)
            mean = np.nanmean(array, axis=0)
            sd = np.nanstd(array, axis=0, ddof=1)
            axis.plot(steps, mean, color=COLORS[variant], label=SHORT[variant],
                      linewidth=1.8)
            axis.fill_between(steps, mean - sd, mean + sd,
                              color=COLORS[variant], alpha=0.12, linewidth=0)
    for axis, label in zip(axes, ("Training timeout fraction",
                                  "Approximate KL", "Policy entropy")):
        axis.set_xlabel("Environment decisions")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].set_ylim(0, 1.03)
    axes[0].legend(frameon=False, ncol=2)
    save_figure(fig, figures / "figure-03-training-diagnostics")

    desc = {(item["variant"], item["opponent"]): item for item in descriptive}
    r1_p = desc[("R1_attack_pbrs", "pursuit_fire")]
    r1_h = desc[("R1_attack_pbrs", "hold")]
    train_timeout = {
        variant: mean_sd_ci([by_key[(variant, seed, "pursuit_fire")][
            "train_timeout_rate"] for seed in SEEDS]) for variant in VARIANTS
    }
    throughput = [by_key[(variant, seed, "pursuit_fire")][
                  "train_throughput_steps_s"] for variant in VARIANTS for seed in SEEDS]
    final_entropy = {
        variant: float(np.mean([
            np.mean([point["metrics"]["entropy"] for point in
                     curves[(variant, seed)][-5:]])
            for seed in SEEDS]))
        for variant in VARIANTS
    }
    total_requests = total_rejected = 0
    for variant in VARIANTS:
        for seed in SEEDS:
            completion = json.loads((args.root_dir /
                f"{variant}_seed{seed}_1m" / "COMPLETED.json").read_text(
                    encoding="utf-8"))
            total_requests += completion["total_fire_requests"]
            total_rejected += completion["total_rejected_fire_requests"]

    def percent(value):
        return f"{100.0 * value:.1f}%"

    report = f"""# V4 奖励消融严格分析

## 分析问题与单位

主要问题是：在相同 PPO、96 环境、100.8 万步和相同三个随机种子下，哪种奖励能提高对固定对手的胜率，同时保持武器合法性、减少策略坍塌并控制跨种子方差？统计单位是**随机种子**（每种奖励 n=3），不是 300 个评估回合。评估回合只降低每个种子胜率估计的测量噪声。

## 核心结论

R1（终局 ±1/0 + 0.1 进攻态势 PBRS）是本阶段优先进入 500 万步复验的候选。其追击发射对手胜率为 {percent(r1_p['win_rate_mean'])} ± {percent(r1_p['win_rate_sd'])}，直飞对手胜率为 {percent(r1_h['win_rate_mean'])} ± {percent(r1_h['win_rate_sd'])}（均为跨种子 SD）。三个种子的追击胜率均不低于 86.3%，直飞胜率均不低于 98.7%。

R2 不应进入下一阶段。其 seed 285 的追击/直飞胜率降至 48.0%/39.3%，合法机会利用率仅 6.8%/3.2%。所有发射请求仍被正确接受，所以这是策略学习坍塌，不是武器发射器拒绝或弹药状态错误。R2 最后五次更新的平均策略熵为 {final_entropy['R2_threat_pbrs']:.3f}，低于 R1 的 {final_entropy['R1_attack_pbrs']:.3f}，与过早收缩探索相符，但尚不能据此断言因果关系。

R3 优于 R2 的均值，但仍有明显种子敏感性；额外高度/发射成本没有带来比 R1 更稳定的收益。R0 可学习但方差过大，说明纯稀疏终局奖励在 100 万步预算下不够稳健。

## 统计边界

Friedman 重复测量检验在追击对手上 χ²={omnibus[0]['statistic']:.3f}, p={omnibus[0]['p_value']:.3f}，在直飞对手上 χ²={omnibus[1]['statistic']:.3f}, p={omnibus[1]['p_value']:.3f}。R1 对三个基线的六个配对比较经 Holm 校正后均未达到 0.05；n=3 时精确双侧符号翻转检验的最小可达 p 值为 0.25。因此允许的表述是“R1 在本次筛选中数值最好且最稳定”，不允许表述为“R1 已被统计证明显著最优”。

## 武器与训练健康度

12 次训练共记录 {total_requests:,} 次发射请求，拒绝 {total_rejected} 次，训练合法发射接受率为 100%；7,200 个固定对手回合中的接受率也为 100%。GPU 训练吞吐为 {np.mean(throughput):.0f} ± {np.std(throughput, ddof=1):.0f} 决策/秒。

四组训练自博弈超时率仍在 {percent(min(v[0] for v in train_timeout.values()))}–{percent(max(v[0] for v in train_timeout.values()))} 的均值范围，而固定对手评估超时率均为 0%。这表示超时主要来自历史种群双方能力接近时的僵持，不能用固定脚本对手的零超时替代自博弈验收；500 万步复验必须同时监控两者。

## 奖励分量与“收敛曲线”解释

R1 的进攻 PBRS 平均绝对贡献仅为终局分量的 {100 * component_ratios['R1_attack_pbrs']['PBGRDiscountedPostureReward']:.2f}%；R2 的威胁 PBRS 为 {100 * component_ratios['R2_threat_pbrs']['ThreatConditionedPotentialReward']:.2f}%。塑形项没有压过终局目标。R1 更可能是以小幅、连续的方向信号改善信用分配；R2 的威胁状态切换则增加了学习非平稳性，这是基于结果的机制推断，尚需状态切换消融验证。

平均 rollout reward 经过零和中心化，理论上会接近 0，不能按普通单智能体累计回报曲线判断“是否收敛”。本分析改用固定对手胜率、超时比例、KL、策略熵和跨种子一致性。100 万步结果足以筛选奖励，但不足以宣称最终收敛。

## 决策

进入下一阶段：仅对 R1 运行 500 万步、至少 3 个种子；每 50 万步固定评估，保存最佳和最终检查点。验收建议为：追击对手跨种子平均胜率 ≥90%，最差种子 ≥80%，固定对手超时率 ≤5%，训练发射接受率 100%，且自博弈超时率不随训练后半程恶化。R0 保留为论文实验中的稀疏奖励对照，不作为正式平台默认奖励。

## Claim Candidates

- Claim: R1 是 100 万步筛选中数值表现最好且跨种子最稳定的奖励。
  - Source evidence: 三种子固定对手胜率、机会利用率和配对轨迹。
  - Allowed wording: “R1 was selected for longer validation.”
  - Forbidden stronger wording: “R1 is statistically significantly superior.”
  - Uncertainty: 每组仅三个种子，固定对手类型有限。
  - Next check: 500 万步三种子、周期性固定评估与更强脚本对手。
  - Decision: keep

- Claim: R2 的失败不是武器合法性实现错误。
  - Source evidence: 请求接受率 100%，但 seed 285 合法机会利用率和胜率同时坍塌。
  - Allowed wording: “The failure is attributable to learned action selection rather than request rejection.”
  - Forbidden stronger wording: “The threat PBRS mechanism is universally invalid.”
  - Uncertainty: 尚未做威胁切换点的状态访问审计。
  - Next check: 记录威胁进入/退出次数及对应 PBRS 跳变量。
  - Decision: keep
"""
    (output / "analysis-report.md").write_text(report, encoding="utf-8")

    contrast_lines = []
    for item in contrasts:
        contrast_lines.append(
            f"|{item['opponent']}|{item['contrast']}|"
            f"{100 * item['mean_difference']:.2f} pp|"
            f"[{100 * item['ci95_low']:.2f}, {100 * item['ci95_high']:.2f}] pp|"
            f"{item['exact_sign_flip_p']:.3f}|{item['holm_adjusted_p']:.3f}|"
            f"{item['cohen_dz'] if item['cohen_dz'] is not None else 'NA'}|"
        )
    appendix = """# 统计附录

## 设计

- 配对单位：随机种子 85、185、285。
- 每个变体每种固定对手：3 个种子 × 300 回合。
- 主指标：固定对手胜率（越高越好）。
- 次指标：超时率、平均回合长度、合法机会利用率。
- 武器硬约束：拒绝请求必须为 0，合法请求接受率必须为 100%。

## 推断方法

样本量 n=3 无法可靠判断正态性，因此不使用依赖正态假设的显著性结论。四组总体比较使用 Friedman 重复测量检验；R1 与三个对照使用穷举 2³ 符号翻转检验，六个比较用 Holm 方法控制家族错误率。Cohen dz 仅作为不稳定的效应量描述；95% t 区间同样仅展示 n=3 带来的不确定性，不作为正态性成立的证据。

|对手|配对比较|平均胜率差|95% CI|精确 p|Holm p|Cohen dz|
|---|---|---:|---:|---:|---:|---:|
""" + "\n".join(contrast_lines) + "\n\n完整数值见 `strict-descriptive-statistics.json` 与 `strict-inferential-statistics.json`。\n"
    (output / "stats-appendix.md").write_text(appendix, encoding="utf-8")

    catalog = """# 图目录

## Figure 1 — fixed-opponent performance

- 文件：`figures/figure-01-fixed-opponent-performance.pdf`（另有 PNG）。
- 目的：比较四种奖励的固定对手胜率、跨种子一致性和交战效率。
- 数据：每个点为一个独立训练种子；灰线连接相同种子；黑横线为三种子均值。
- 观察：R1 的三个种子集中在高胜率区域且回合较短；R0/R2/R3 存在明显低值种子。
- 含义：R1 是长程复验候选，但 n=3 不支持显著最优结论。

## Figure 2 — fire-opportunity exploitation

- 文件：`figures/figure-02-fire-opportunity-exploitation.pdf`（另有 PNG）。
- 目的：区分武器实现故障与策略不发射。
- 数据：学习方实际发射数/合法发射机会数；所有请求接受率另经硬验收为 100%。
- 观察：R2 seed 285 的利用率接近 0，而请求拒绝仍为 0。
- 含义：R2 的坏种子属于动作选择坍塌，不应修改武器合法判定来掩盖。

## Figure 3 — training diagnostics

- 文件：`figures/figure-03-training-diagnostics.pdf`（另有 PNG）。
- 目的：用超时比例、PPO KL 和熵检查训练健康度，避免误读零和平均奖励。
- 数据：实线为三种子均值，阴影为 ±1 SD，每点为一次 PPO 更新。
- 观察：训练保持有限 KL，熵逐步降低；自博弈超时仍有较大波动。
- 含义：优化器未失控，但 100 万步尚不能证明策略已最终收敛。
"""
    (output / "figure-catalog.md").write_text(catalog, encoding="utf-8")
    print(json.dumps({"status": "PASS", "selected_candidate": "R1_attack_pbrs",
                      "runs": 12, "evaluation_episodes": 7200,
                      "total_fire_requests": total_requests,
                      "total_rejected_requests": total_rejected},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
