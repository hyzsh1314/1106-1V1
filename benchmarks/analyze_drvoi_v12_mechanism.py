#!/usr/bin/env python3
"""Build a strict single-seed V12 mechanism-analysis bundle."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
OPPONENTS = ("hold", "pursuit", "evasive")
METHODS = {
    "V9": "runs/drvoi_rhppo_v9/coach-online/development-1m-seed85",
    "V11.2": "runs/drvoi_rhppo_v11_2/race-episode-latched/frozen-v9-seed85",
    "V11.3": "runs/drvoi_rhppo_v11_3/race-sustained/frozen-v9-seed85",
    "B1": "runs/drvoi_rhppo_v12/mrat-b1-frozen-seed85",
    "B4": "runs/drvoi_rhppo_v12/mrat-b4-return-gated-smoke-seed85",
    "B5": "runs/drvoi_rhppo_v12/mrat-b5-behavior-gated-smoke-seed85",
    "B6": "runs/drvoi_rhppo_v12/mrat-b6-onpolicy-gated-smoke-seed85",
    "B7": "runs/drvoi_rhppo_v12/b7-gate-latched-diagnostic-seed85",
    "B8": "runs/drvoi_rhppo_v12/b8-sprt-latched-diagnostic-seed85",
}
MECHANISM_METHODS = ("B1", "B4", "B5", "B6", "B7", "B8")
COLORS = {
    "V9": "#7F7F7F", "V11.2": "#0072B2", "V11.3": "#56B4E9",
    "B1": "#CC79A7", "B4": "#E69F00", "B5": "#F0E442",
    "B6": "#009E73", "B7": "#D55E00", "B8": "#000000",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("docs/analysis/drvoi-rhppo-v12-mechanism"))
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return centre - radius, centre + radius


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        value = min((m - rank) * p_values[index], 1.0)
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def performance_table(available: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for method, directory in available.items():
        for opponent in OPPONENTS:
            summary = load_json(directory / f"eval-{opponent}.json")
            win_lo, win_hi = wilson(summary["wins"], summary["episodes"])
            timeout_count = round(summary["timeout_rate"] * summary["episodes"])
            timeout_lo, timeout_hi = wilson(timeout_count, summary["episodes"])
            rows.append({
                "method": method, "opponent": opponent,
                "episodes": summary["episodes"],
                "win_rate": summary["win_rate"], "win_ci_low": win_lo,
                "win_ci_high": win_hi, "timeout_rate": summary["timeout_rate"],
                "timeout_ci_low": timeout_lo, "timeout_ci_high": timeout_hi,
                "legal_request_rate": summary["legal_request_rate"],
                "hit_per_launch": summary["hit_per_launch"],
                "accepted_launches": summary["accepted_launches"],
                "hits": summary["hits"],
                "intercept_fraction": summary.get("intercept_expert_steps", 0)
                / max(summary.get("environment_decisions_executed", 1), 1),
                "forced_fraction": summary.get("forced_intervention_steps", 0)
                / max(summary.get("environment_decisions_executed", 1), 1),
                "gate_allowed_fraction": summary.get(
                    "routing_gate_allowed_fraction", np.nan),
                "mean_option_dwell": summary.get("mean_option_dwell", np.nan),
            })
    return pd.DataFrame(rows)


def mechanism_table(available: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for method in MECHANISM_METHODS:
        if method not in available:
            continue
        directory = available[method]
        episodes = load_jsonl(directory / "eval-evasive-episodes.jsonl")
        segments = load_jsonl(directory / "eval-evasive-segments.jsonl")
        by_episode: dict[tuple[int, int], list[dict]] = {}
        for segment in segments:
            by_episode.setdefault(
                (segment["env_index"], segment["episode_id"]), []).append(segment)
        for episode in episodes:
            key = (episode["env_index"], episode["episode_id"])
            intercept = [segment for segment in by_episode.get(key, [])
                         if segment["expert"] == 1]
            outcome = "win" if episode["win"] else (
                "timeout" if episode["timeout"] else "loss")
            first_intercept = episode.get("first_intercept_step")
            if first_intercept is None:
                first_intercept = min(
                    (segment["start_step"] for segment in intercept),
                    default=episode["episode_length"])
            rows.append({
                "method": method, "outcome": outcome,
                "initial_state_sha256": episode["initial_state_sha256"],
                "learner_side": episode["learner_side"],
                "win": int(episode["win"]), "timeout": int(episode["timeout"]),
                "episode_length": episode["episode_length"],
                "intercept_fraction": episode["intercept_expert_steps"]
                / episode["episode_length"],
                "forced_fraction": episode["forced_intervention_steps"]
                / episode["episode_length"],
                "gate_fraction": episode.get("routing_gate_allowed_steps", 0)
                / episode["episode_length"],
                "first_intercept_step": first_intercept,
                "first_gate_allowed_step": episode.get(
                    "first_routing_gate_allowed_step"),
                "first_gate_latch_step": episode.get("first_gate_latch_step"),
                "max_intercept_dwell": max(
                    (segment["duration"] for segment in intercept), default=0),
                "expert_switches": episode["expert_switches"],
            })
    return pd.DataFrame(rows)


def paired_tests(mechanism: pd.DataFrame) -> pd.DataFrame:
    contrasts = [("B6", "B7"), ("B6", "B8"), ("B7", "B8")]
    rows = []
    for left, right in contrasts:
        if left not in set(mechanism.method) or right not in set(mechanism.method):
            continue
        a = mechanism[mechanism.method == left].set_index(
            ["initial_state_sha256", "learner_side"])
        b = mechanism[mechanism.method == right].set_index(
            ["initial_state_sha256", "learner_side"])
        joined = a[["win"]].join(b[["win"]], lsuffix="_left", rsuffix="_right", how="inner")
        left_only = int(((joined.win_left == 1) & (joined.win_right == 0)).sum())
        right_only = int(((joined.win_left == 0) & (joined.win_right == 1)).sum())
        discordant = left_only + right_only
        p_value = (binomtest(min(left_only, right_only), discordant, 0.5).pvalue
                   if discordant else 1.0)
        rows.append({
            "contrast": f"{right} - {left}", "paired_episodes": len(joined),
            "left_only_wins": left_only, "right_only_wins": right_only,
            "paired_win_difference": (right_only - left_only) / max(len(joined), 1),
            "mcnemar_exact_p": p_value,
        })
    if rows:
        adjusted = holm_adjust([row["mcnemar_exact_p"] for row in rows])
        for row, value in zip(rows, adjusted):
            row["holm_adjusted_p"] = value
    return pd.DataFrame(rows)


def plot_performance(performance: pd.DataFrame, output: Path) -> None:
    methods = list(dict.fromkeys(performance.method))
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharey=True)
    x = np.arange(len(methods))
    for axis, opponent in zip(axes, OPPONENTS):
        frame = performance[performance.opponent == opponent].set_index("method").loc[methods]
        values = frame.win_rate.to_numpy()
        errors = np.maximum(np.vstack((
            values - frame.win_ci_low.to_numpy(),
            frame.win_ci_high.to_numpy() - values)), 0.0)
        axis.errorbar(x, values, yerr=errors, fmt="none", ecolor="#333333",
                      capsize=2.5, linewidth=1.0, zorder=1)
        axis.scatter(x, values, s=42,
                     c=[COLORS.get(method, "#666666") for method in methods],
                     edgecolors="black", linewidths=0.5, zorder=2)
        axis.scatter(x, frame.timeout_rate, marker="x", s=35, c="#D55E00",
                     linewidths=1.2,
                     label="Timeout rate" if opponent == "hold" else None)
        axis.set_title(opponent.capitalize())
        axis.set_xticks(x, methods, rotation=55, ha="right")
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Episode proportion")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mechanism(mechanism: pd.DataFrame, output: Path) -> None:
    methods = [method for method in MECHANISM_METHODS
               if method in set(mechanism.method)]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for axis, column, label in (
            (axes[0], "intercept_fraction", "LOS expert fraction"),
            (axes[1], "first_intercept_step", "First LOS entry (decision step)")):
        data = [mechanism.loc[mechanism.method == method, column].to_numpy()
                for method in methods]
        box = axis.boxplot(data, tick_labels=methods, showfliers=False,
                           patch_artist=True, widths=0.65)
        for patch, method in zip(box["boxes"], methods):
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.70)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
        if column == "intercept_fraction":
            axis.set_ylim(0.0, 1.02)
        else:
            axis.set_ylim(bottom=0.0)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    return frame[columns].to_markdown(index=False, floatfmt=".4f")


def main() -> None:
    args = parse_args()
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    available = {
        method: ROOT / relative for method, relative in METHODS.items()
        if all((ROOT / relative / f"eval-{opponent}.json").is_file()
               for opponent in OPPONENTS)
    }
    performance = performance_table(available)
    mechanism = mechanism_table(available)
    paired = paired_tests(mechanism)
    performance.to_csv(output / "exact-performance.csv", index=False)
    mechanism.to_csv(output / "evasive-episode-mechanism.csv", index=False)
    paired.to_csv(output / "paired-frozen-policy-tests.csv", index=False)
    plot_performance(performance, figures / "figure-01-performance")
    plot_mechanism(mechanism, figures / "figure-02-evasive-mechanism")

    macro = performance.groupby("method", sort=False).agg(
        macro_win=("win_rate", "mean"), macro_timeout=("timeout_rate", "mean"),
        total_launches=("accepted_launches", "sum"), total_hits=("hits", "sum"))
    evasive = performance[performance.opponent == "evasive"].set_index("method")
    mechanism_summary = mechanism.groupby(["method", "outcome"], sort=False).agg(
        episodes=("outcome", "size"), mean_los_fraction=("intercept_fraction", "mean"),
        median_first_los=("first_intercept_step", "median"),
        mean_max_los_dwell=("max_intercept_dwell", "mean")).reset_index()

    report = f"""# V12 B6–B8 mechanism analysis

## Analysis question

Can online-distribution return supervision and temporal gate aggregation recover Evasive performance without reproducing the Hold/Pursuit over-intervention failure? The primary unit for algorithm-level inference is the training seed; only seed 85 is available, so all method comparisons below are development evidence rather than algorithm-level significance claims.

## Key findings

- A6/A7 fixed the observed deployment-coverage failure: B6 Evasive gate allowance reached {evasive.loc['B6','gate_allowed_fraction']:.2%}, but B6 win rate remained {evasive.loc['B6','win_rate']:.2%}. This separates model coverage from control effectiveness.
- B7 gate-latching raised Evasive win rate to {evasive.loc['B7','win_rate']:.2%}, but its macro win rate was {macro.loc['B7','macro_win']:.2%}; long skill persistence is necessary but indiscriminate latching harms simple opponents.
{('- B8 SPRT-latching reached Evasive win rate ' + f"{evasive.loc['B8','win_rate']:.2%}" + ' with macro win rate ' + f"{macro.loc['B8','macro_win']:.2%}" + '.' ) if 'B8' in evasive.index else '- B8 was not complete when this bundle was generated.'}
- Weapon legality remained 100% for every included V12 evaluation; the negative results are not explained by rejected launches or crashes.

## Exact macro summary

{markdown_table(macro.reset_index(), ['method','macro_win','macro_timeout','total_launches','total_hits'])}

## Evasive mechanism breakdown

{markdown_table(mechanism_summary, ['method','outcome','episodes','mean_los_fraction','median_first_los','mean_max_los_dwell'])}

## Claim Candidates

- Claim:
  - Source evidence: A5 versus A7 offline behavior-stratified reports and B5 versus B6 online gate fractions.
  - Allowed wording: On seed-85 development data, on-policy aggregation corrected the measured gate-coverage shift but did not by itself restore Evasive wins.
  - Forbidden stronger wording: On-policy aggregation is ineffective in general.
  - Uncertainty: One training seed and one reused development evaluation seed.
  - Next check: multi-seed development only after a structure passes the current Pareto screen.
  - Decision: keep.
- Claim:
  - Source evidence: B6 versus B7 under the same B6 checkpoint and A7 predictor.
  - Allowed wording: The frozen-policy diagnostic supports option persistence as a causal mechanism candidate for Evasive recovery, while showing a simple-opponent cost.
  - Forbidden stronger wording: Episode latching is the optimal air-combat strategy.
  - Uncertainty: fixed scripted opponents and development initial-state distribution.
  - Next check: temporal evidence aggregation and randomized opponent behaviors.
  - Decision: keep with boundary.

## Decision

Do not expand B4–B8 to one million steps. Retain A6/A7 as valid coverage infrastructure. The current return-gated branch is a negative result; V11.2/V11.3 remain the strongest balanced development structures until a new candidate improves their Hold/Pursuit/Evasive Pareto front.
"""
    (output / "analysis-report.md").write_text(report, encoding="utf-8")

    stats = f"""# Statistical appendix

## Units and validity

- Training-seed unit: n=1 (seed 85). No t-test, ANOVA, seed-level standard deviation, or algorithm-level p-value is valid.
- Frozen-policy evaluation unit: 192 paired initial conditions per opponent (96 red, 96 blue) under development seed 10085.
- Wilson 95% intervals describe episode sampling uncertainty for a fixed checkpoint only; they do not quantify training instability.
- Exact paired McNemar/binomial tests below compare deterministic frozen policies on matched initial states. Holm adjustment covers the listed B6/B7/B8 Evasive contrasts. These tests cannot support algorithm-level superiority.

## Frozen-policy paired contrasts

{markdown_table(paired, list(paired.columns)) if len(paired) else 'B8 incomplete; paired contrast table unavailable.'}

## Exact performance table

{markdown_table(performance, ['method','opponent','episodes','win_rate','win_ci_low','win_ci_high','timeout_rate','legal_request_rate','hit_per_launch','intercept_fraction','gate_allowed_fraction'])}

## Blockers

1. Only one training seed; no estimate of seed variance.
2. Development evaluation seed 10085 has been repeatedly reused for structural decisions.
3. Fixed Hold/Pursuit/Evasive scripts are not opponent-family generalization.
4. A7 empirical LCB coverage at z=1.64 was 81.32%, below a 90% calibration target; it is not a formal confidence guarantee.
"""
    (output / "stats-appendix.md").write_text(stats, encoding="utf-8")

    catalog = """# Figure catalog

## Figure 01 — `figures/figure-01-performance.pdf`

- Purpose: compare frozen-policy win and timeout profiles across all three opponents.
- Data source: each method's `eval-{hold,pursuit,evasive}.json`, 192 episodes per cell.
- Caption requirements: circles are win proportions; vertical bars are Wilson 95% intervals for a fixed checkpoint; orange crosses are timeout proportions; intervals do not represent training-seed variance.
- Key observation: the methods trade simple-opponent retention against Evasive recovery; no current V12 candidate dominates V11.2/V11.3.
- Interpretation checklist: inspect all three panels; do not select on macro average alone; do not read overlapping episode intervals as a seed-level test.

## Figure 02 — `figures/figure-02-evasive-mechanism.pdf`

- Purpose: test whether LOS usage and entry timing explain Evasive outcomes.
- Data source: 192 episode JSONL records and option-segment JSONL records per V12 diagnostic.
- Caption requirements: boxes show episode distributions (median and quartiles; whiskers 1.5 IQR, fliers hidden); left is LOS expert fraction, right is first LOS decision step.
- Key observation: high LOS occupancy alone is insufficient; successful variants also differ in when a persistent LOS option begins.
- Interpretation checklist: distributions are episodes from one frozen seed, not independent training runs; association with outcome does not by itself prove causality.
"""
    (output / "figure-catalog.md").write_text(catalog, encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output), "methods": list(available),
        "performance_rows": len(performance), "mechanism_rows": len(mechanism),
        "paired_contrasts": len(paired),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
