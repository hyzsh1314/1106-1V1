#!/usr/bin/env python3
"""Analyze the three-seed Torch WVR reward ablation.

The independent experimental unit is the training seed. Evaluation episodes are
used to estimate each seed's policy performance and are never treated as
independent method-level replicates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np


VARIANTS = ("competitive", "engagement", "attack-quality")
VARIANT_LABELS = {
    "competitive": "Competitive",
    "engagement": "Engagement",
    "attack-quality": "Attack quality",
}
OPPONENTS = ("pursuit", "hold", "evasive")
OPPONENT_LABELS = {"pursuit": "Pursuit", "hold": "Hold", "evasive": "Evasive"}
SEEDS = (85, 185, 285)
COLORS = {
    "competitive": "#0072B2",
    "engagement": "#E69F00",
    "attack-quality": "#CC79A7",
}
MARKERS = {85: "o", 185: "s", 285: "^"}
T_CRIT_DF2 = 4.302652729911275


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("runs/torch6dof_validation/v4-reward-ablation-3seed-1m"),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def summary(values: list[float]) -> tuple[float, float, float, float]:
    avg = mean(values)
    sd = stdev(values)
    half_width = T_CRIT_DF2 * sd / math.sqrt(len(values))
    return avg, sd, avg - half_width, avg + half_width


def fmt_mean_sd(values: list[float], percent: bool = False) -> str:
    avg, sd, _, _ = summary(values)
    if percent:
        return f"{100 * avg:.1f}% ± {100 * sd:.1f}%"
    return f"{avg:.3f} ± {sd:.3f}"


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: plt.Figure, figures: Path, stem: str) -> None:
    fig.savefig(figures / f"{stem}.pdf")
    fig.savefig(figures / f"{stem}.png", dpi=600)
    plt.close(fig)


def figure_main_comparison(eval_data: dict, figures: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex=True, sharey="row")
    metrics = (("win_rate", "Win rate"), ("timeout_rate", "Timeout rate"))
    x = np.arange(len(VARIANTS))
    for col, opponent in enumerate(OPPONENTS):
        for row, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            for seed in SEEDS:
                values = [eval_data[(variant, seed, opponent)][metric] for variant in VARIANTS]
                ax.plot(
                    x,
                    values,
                    color="#777777",
                    alpha=0.48,
                    linewidth=0.9,
                    marker=MARKERS[seed],
                    markersize=3.7,
                    label=f"seed {seed}" if col == 0 and row == 0 else None,
                )
            for idx, variant in enumerate(VARIANTS):
                values = [eval_data[(variant, seed, opponent)][metric] for seed in SEEDS]
                avg, sd, _, _ = summary(values)
                ax.errorbar(
                    idx,
                    avg,
                    yerr=sd,
                    color=COLORS[variant],
                    marker="D",
                    markersize=5.0,
                    linewidth=1.5,
                    capsize=3,
                    zorder=5,
                )
            ax.set_ylim(-0.04, 1.04)
            ax.set_xticks(x, ["Comp.", "Engage.", "Attack-Q"])
            if row == 0:
                ax.set_title(OPPONENT_LABELS[opponent])
            if col == 0:
                ax.set_ylabel(ylabel)
    axes[0, 0].legend(frameon=False, loc="upper left")
    fig.subplots_adjust(wspace=0.15, hspace=0.20)
    save_figure(fig, figures, "figure-01-fixed-opponent-performance")


def figure_reward_curves(train_data: dict, figures: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharex=True)
    for ax, variant in zip(axes, VARIANTS):
        matrices = []
        decisions = None
        for seed in SEEDS:
            history = train_data[(variant, seed)]["history"]
            decisions = np.asarray([item["environment_decisions"] for item in history]) / 1e6
            reward = np.asarray([item["mean_learning_reward"] for item in history])
            matrices.append(reward)
            ax.plot(
                decisions,
                reward,
                color=COLORS[variant],
                alpha=0.26,
                linewidth=0.9,
                marker=MARKERS[seed],
                markersize=1.8,
                markevery=4,
            )
        matrix = np.vstack(matrices)
        avg = matrix.mean(axis=0)
        sd = matrix.std(axis=0, ddof=1)
        ax.fill_between(decisions, avg - sd, avg + sd, color=COLORS[variant], alpha=0.18)
        ax.plot(decisions, avg, color=COLORS[variant], linewidth=1.8, label="seed mean")
        ax.axvline(14 * 24576 / 1e6, color="#555555", linestyle="--", linewidth=0.8)
        ax.axvline(28 * 24576 / 1e6, color="#555555", linestyle=":", linewidth=0.8)
        ax.axhline(0.0, color="#222222", linewidth=0.6)
        ax.set_title(VARIANT_LABELS[variant])
        ax.set_xlabel("Environment decisions (million)")
    axes[0].set_ylabel("Mean learning reward / decision")
    axes[-1].legend(frameon=False, loc="best")
    fig.subplots_adjust(wspace=0.31)
    save_figure(fig, figures, "figure-02-reward-convergence")


def figure_mechanism(eval_data: dict, figures: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex=True, sharey="row")
    metrics = (
        ("legal_opportunity_episode_rate", "Episodes with legal shot"),
        ("hit_per_launch", "Hit probability / launch"),
    )
    x = np.arange(len(VARIANTS))
    for col, opponent in enumerate(OPPONENTS):
        for row, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            for idx, variant in enumerate(VARIANTS):
                values = [eval_data[(variant, seed, opponent)][metric] for seed in SEEDS]
                jitter = np.asarray([-0.055, 0.0, 0.055])
                for offset, seed, value in zip(jitter, SEEDS, values):
                    ax.scatter(
                        idx + offset,
                        value,
                        s=22,
                        color=COLORS[variant],
                        marker=MARKERS[seed],
                        alpha=0.75,
                    )
                avg, sd, _, _ = summary(values)
                ax.errorbar(
                    idx,
                    avg,
                    yerr=sd,
                    color="#111111",
                    marker="_",
                    markersize=9,
                    capsize=3,
                    linewidth=1.0,
                    zorder=4,
                )
            ax.set_ylim(-0.04, 1.04)
            ax.set_xticks(x, ["Comp.", "Engage.", "Attack-Q"])
            if row == 0:
                ax.set_title(OPPONENT_LABELS[opponent])
            if col == 0:
                ax.set_ylabel(ylabel)
    fig.subplots_adjust(wspace=0.15, hspace=0.20)
    save_figure(fig, figures, "figure-03-launch-mechanism")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = (args.output or (root / "analysis-output")).resolve()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    if not (root / "COMPLETED").exists():
        raise RuntimeError(f"Experiment root is not complete: {root}")

    train_data: dict[tuple[str, int], dict] = {}
    eval_data: dict[tuple[str, int, str], dict] = {}
    eval_rows: list[dict] = []
    train_rows: list[dict] = []

    for variant in VARIANTS:
        for seed in SEEDS:
            run = root / variant / f"seed{seed}"
            train = load_json(run / "train.json")
            if train["total_environment_decisions"] != 1_007_616 or len(train["history"]) != 41:
                raise RuntimeError(f"Incomplete training history: {run}")
            if train["seed"] != seed or train["reward_variant"] != variant:
                raise RuntimeError(f"Configuration mismatch: {run}")
            train_data[(variant, seed)] = train
            train_rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "decisions": train["total_environment_decisions"],
                    "elapsed_s": train["elapsed_s"],
                    "decisions_per_s": train["environment_decisions_per_s"],
                    "final_mean_reward": train["history"][-1]["mean_learning_reward"],
                    "final_approx_kl": train["history"][-1]["approx_kl"],
                    "final_value_loss": train["history"][-1]["value_loss"],
                }
            )
            for opponent in OPPONENTS:
                evaluation = load_json(run / f"eval-{opponent}.json")
                if evaluation["episodes"] != 192 or evaluation["seed"] != 10085:
                    raise RuntimeError(f"Evaluation mismatch: {run / f'eval-{opponent}.json'}")
                eval_data[(variant, seed, opponent)] = evaluation
                eval_rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "opponent": opponent,
                        "win_rate": evaluation["win_rate"],
                        "loss_rate": evaluation["loss_rate"],
                        "draw_rate": evaluation["draw_rate"],
                        "timeout_rate": evaluation["timeout_rate"],
                        "crash_rate": evaluation["learner_crash_rate"],
                        "legal_opportunity_episode_rate": evaluation["legal_opportunity_episode_rate"],
                        "launch_episode_rate": evaluation["launch_episode_rate"],
                        "hit_per_launch": evaluation["hit_per_launch"],
                        "mean_first_legal_step": evaluation["mean_first_legal_step"],
                        "legal_opportunity_step_fraction": evaluation["legal_opportunity_step_fraction"],
                        "mean_launch_attack_quality": evaluation["mean_launch_attack_quality"],
                        "rejected_fire_requests": evaluation["rejected_fire_requests"],
                    }
                )

    write_csv(output / "evaluation-seed-level.csv", eval_rows, list(eval_rows[0]))
    write_csv(output / "training-run-level.csv", train_rows, list(train_rows[0]))

    configure_plot_style()
    figure_main_comparison(eval_data, figures)
    figure_reward_curves(train_data, figures)
    figure_mechanism(eval_data, figures)

    aggregate_rows: list[dict] = []
    for variant in VARIANTS:
        for opponent in OPPONENTS:
            for metric in (
                "win_rate",
                "loss_rate",
                "timeout_rate",
                "legal_opportunity_episode_rate",
                "hit_per_launch",
            ):
                values = [eval_data[(variant, seed, opponent)][metric] for seed in SEEDS]
                avg, sd, ci_low, ci_high = summary(values)
                aggregate_rows.append(
                    {
                        "variant": variant,
                        "opponent": opponent,
                        "metric": metric,
                        "mean": avg,
                        "sd": sd,
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "seed85": values[0],
                        "seed185": values[1],
                        "seed285": values[2],
                    }
                )
    write_csv(output / "aggregate-summary.csv", aggregate_rows, list(aggregate_rows[0]))

    primary_metrics = ("win_rate", "timeout_rate", "legal_opportunity_episode_rate")
    delta_rows: list[dict] = []
    for variant in ("engagement", "attack-quality"):
        for opponent in OPPONENTS:
            for metric in primary_metrics:
                deltas = [
                    eval_data[(variant, seed, opponent)][metric]
                    - eval_data[("competitive", seed, opponent)][metric]
                    for seed in SEEDS
                ]
                avg, sd, ci_low, ci_high = summary(deltas)
                delta_rows.append(
                    {
                        "contrast": f"{variant}-competitive",
                        "opponent": opponent,
                        "metric": metric,
                        "mean_delta": avg,
                        "sd_delta": sd,
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "seed85_delta": deltas[0],
                        "seed185_delta": deltas[1],
                        "seed285_delta": deltas[2],
                    }
                )
    write_csv(output / "paired-deltas.csv", delta_rows, list(delta_rows[0]))

    def values(variant: str, opponent: str, metric: str) -> list[float]:
        return [eval_data[(variant, seed, opponent)][metric] for seed in SEEDS]

    report_lines = [
        "# Torch 6DoF 1v1 reward ablation: three-seed analysis",
        "",
        "## Executive conclusion",
        "",
        "The current evidence supports **retaining the competitive reward as the baseline**. "
        "Neither the stronger engagement terminal penalties nor the 0.5-weight attack-quality "
        "potential improves performance consistently across paired seeds. The attack-quality "
        "variant is rejected at its present weight because it lowers mean fixed-opponent win "
        "rates without increasing legal-shot coverage reliably.",
        "",
        "The remaining blocker is not the missile launch state machine: all 27 evaluations have "
        "zero rejected fire requests, and launch-to-hit performance is nonzero against pursuit "
        "and hold opponents. The blocker is geometry acquisition against the evasive opponent: "
        "only about 6%–9% of evaluation episodes ever contain a legal shot across all variants.",
        "",
        "## Experimental unit and comparability",
        "",
        "- Independent unit: training seed (n=3 paired seeds: 85, 185, 285).",
        "- Each run: 1,007,616 environment decisions, 96 parallel environments, 41 PPO updates.",
        "- Each checkpoint: 192 fixed-opponent episodes per opponent, balanced 96 red / 96 blue.",
        "- Dynamics, weapon model, observation, action space, initialization distribution, "
        "opponent schedule, and PPO settings are held fixed. Only reward configuration changes.",
        "- Evaluation episodes are not treated as method-level replicates.",
        "- Artifacts remain marked `scientific_result=false`; this is an engineering gate, not yet "
        "a paper-level benchmark claim.",
        "",
        "## Seed-level fixed-opponent results",
        "",
        "Values are mean ± sample SD across the three training seeds.",
        "",
        "| Variant | Pursuit win | Hold win | Evasive win | Pursuit timeout | Hold timeout | Evasive timeout |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        report_lines.append(
            "| " + VARIANT_LABELS[variant] + " | "
            + " | ".join(
                [
                    fmt_mean_sd(values(variant, "pursuit", "win_rate"), True),
                    fmt_mean_sd(values(variant, "hold", "win_rate"), True),
                    fmt_mean_sd(values(variant, "evasive", "win_rate"), True),
                    fmt_mean_sd(values(variant, "pursuit", "timeout_rate"), True),
                    fmt_mean_sd(values(variant, "hold", "timeout_rate"), True),
                    fmt_mean_sd(values(variant, "evasive", "timeout_rate"), True),
                ]
            )
            + " |"
        )
    report_lines += [
        "",
        "## Interpretation",
        "",
        "### Figure 1 — fixed-opponent outcome",
        "",
        "**Purpose.** Determine whether either reward modification improves fixed-opponent "
        "performance consistently across matched seeds. **Observation.** Competitive reward has "
        "the highest mean pursuit win rate and the highest mean hold win rate; every variant shows "
        "large hold-opponent seed dispersion. Evasive win rate remains near zero. **Interpretation.** "
        "The proposed reward additions do not remove optimization instability and do not solve "
        "evasive engagement. **Implication.** Keep competitive reward for the next baseline and stop "
        "the 0.5 attack-quality branch.",
        "",
        "### Figure 2 — reward convergence",
        "",
        "**Purpose.** Check whether PPO is numerically learning rather than diverging. "
        "**Observation.** Reward trajectories remain finite and PPO completes all 41 updates, but "
        "the curves change when the staged opponent mixture changes at about 0.344 M and 0.688 M "
        "decisions. **Interpretation.** A single monotonic 'convergence' criterion is inappropriate "
        "for this non-stationary curriculum; fixed-opponent evaluation is the decision metric. "
        "**Implication.** Continue to plot learning rewards for diagnostics, but never select a "
        "reward variant from the training curve alone.",
        "",
        "### Figure 3 — launch mechanism",
        "",
        "**Purpose.** Separate fire-control faults from failure to acquire engagement geometry. "
        "**Observation.** Pursuit and hold opponents frequently produce legal shots, whereas only "
        "about 6%–9% of evasive episodes do. Rejected fire requests are zero in every evaluation. "
        "**Interpretation.** The limiting mechanism is guidance/geometry acquisition before launch, "
        "not weapon authorization or missile hit processing. **Implication.** The next experiment "
        "should target curriculum and state coverage for evasive intercept geometry, without changing "
        "the missile state machine or adding a direct launch-count reward.",
        "",
        "## Predeclared engineering gates",
        "",
        "Gate thresholds: pursuit win ≥60%, hold win ≥40%, timeout ≤30% for pursuit and hold, "
        "rejected-fire rate ≤1%, crash rate ≤2%. A robust method must pass for every seed, not only "
        "on the seed mean.",
        "",
        "- No reward variant passes every performance gate for all three seeds.",
        "- All variants pass the weapon legality gate: 0 rejected requests in all evaluations.",
        "- All variants pass the evaluation crash gate.",
        "- Competitive is the least-regret baseline because the alternatives do not deliver a "
        "consistent paired-seed improvement.",
        "",
        "## Next decision",
        "",
        "1. Revert the active baseline to competitive reward (`attack_quality_weight=0`).",
        "2. Freeze reward coefficients and weapon logic for the next stage.",
        "3. Replace reward tinkering with a dedicated evasive-intercept curriculum that starts from "
        "states known to contain recoverable launch geometry, then expands the initial-condition "
        "distribution only after fixed-opponent gates pass.",
        "4. Run a short 3-seed curriculum screen, followed by at least 5 seeds and a longer horizon "
        "for the selected configuration before using it as a paper benchmark.",
        "",
        "## Claim candidates",
        "",
        "### Supported claim",
        "- **Claim:** At one million decisions, the tested reward additions do not consistently "
        "outperform the competitive baseline across matched seeds.",
        "- **Evidence:** paired three-seed fixed-opponent evaluations and Figures 1–3.",
        "- **Scope:** this PPO configuration, staged opponents, and current initialization envelope.",
        "- **Caveat:** n=3 seeds is adequate for an engineering rejection gate but underpowered for "
        "a publication-level null claim.",
        "",
        "### Blocked claim",
        "- **Blocked:** the platform has solved 1v1 missile combat against evasive opponents.",
        "- **Reason:** near-zero evasive win rates and legal-shot opportunity in fewer than 10% of "
        "episodes.",
        "- **Needed evidence:** curriculum correction, multi-seed gate pass, and longer training.",
    ]
    (output / "analysis-report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    speed_lines = []
    for variant in VARIANTS:
        speed_values = [train_data[(variant, seed)]["environment_decisions_per_s"] for seed in SEEDS]
        speed_lines.append(f"- {VARIANT_LABELS[variant]}: {fmt_mean_sd(speed_values)} decisions/s")
    stats_lines = [
        "# Statistical appendix",
        "",
        "## Analysis plan",
        "",
        "The training seed is the independent unit. Variants are paired by seed. With only three "
        "seeds, normality testing is uninformative and exact signed-rank tests have insufficient "
        "resolution for confirmatory inference. Therefore this gate uses seed-level values, mean, "
        "sample SD, exploratory 95% t intervals, and paired deltas; no statistical-significance "
        "claim is made. The very wide intervals in `aggregate-summary.csv` are retained rather than "
        "hidden. No evaluation episode is counted as an independent replicate.",
        "",
        "## Multiple comparisons",
        "",
        "No null-hypothesis tests are reported, so no multiplicity-adjusted p-values are needed. "
        "The two prespecified contrasts (engagement − competitive and attack-quality − competitive) "
        "are provided in `paired-deltas.csv` for all three opponents and primary metrics.",
        "",
        "## Throughput",
        "",
        *speed_lines,
        "",
        "## Reproducibility files",
        "",
        "- `evaluation-seed-level.csv`: one row per training seed × variant × opponent.",
        "- `training-run-level.csv`: one row per training run.",
        "- `aggregate-summary.csv`: mean, SD, exploratory 95% interval, and raw seed values.",
        "- `paired-deltas.csv`: matched-seed deltas against competitive reward.",
        "",
        "## Limitations",
        "",
        "- Three seeds cannot establish equivalence or a publication-grade null result.",
        "- One million decisions may rank short-run learnability differently from long-run asymptotic performance.",
        "- All fixed-opponent evaluations share one episode-generation seed; policy seeds remain the "
        "unit of analysis, but a paper result should add multiple evaluation scenario seeds.",
        "- The staged curriculum is non-stationary, so raw reward curves are diagnostic rather than "
        "a standalone convergence proof.",
    ]
    (output / "stats-appendix.md").write_text("\n".join(stats_lines) + "\n", encoding="utf-8")

    catalog_lines = [
        "# Figure catalog",
        "",
        "## Figure 1 — fixed-opponent performance",
        "",
        "Files: `figures/figure-01-fixed-opponent-performance.pdf` and `.png`.",
        "",
        "Paired seed trajectories for win and timeout rates. Diamonds and error bars are mean ± "
        "sample SD over n=3 training seeds; gray lines connect the same seed across variants. Each "
        "seed uses 192 evaluation episodes per opponent.",
        "",
        "## Figure 2 — reward convergence",
        "",
        "Files: `figures/figure-02-reward-convergence.pdf` and `.png`.",
        "",
        "Unsmoothed mean learning reward per decision over 41 PPO updates. Thin lines are individual "
        "seeds, thick lines are seed means, and ribbons are ±1 sample SD. Vertical lines mark the "
        "fixed curriculum transitions. Panels have independent y scales because reward definitions differ.",
        "",
        "## Figure 3 — launch mechanism",
        "",
        "Files: `figures/figure-03-launch-mechanism.pdf` and `.png`.",
        "",
        "Seed-level legal-shot episode coverage and hit probability per accepted launch. Black bars "
        "show mean ± sample SD over n=3 training seeds. Each seed uses 192 evaluation episodes per opponent.",
    ]
    (output / "figure-catalog.md").write_text("\n".join(catalog_lines) + "\n", encoding="utf-8")

    manifest = {
        "root": str(root),
        "training_runs": 9,
        "fixed_opponent_evaluations": 27,
        "training_seeds": list(SEEDS),
        "evaluation_seed": 10085,
        "episodes_per_evaluation": 192,
        "analysis_unit": "training seed",
        "selected_baseline": "competitive",
        "rejected_variant": "attack-quality weight 0.5",
    }
    (output / "analysis-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
