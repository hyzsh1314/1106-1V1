"""Compare matched no-option, short-pulse, and long-option continuations."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 85, 127)
OPPONENTS = ("pursuit_fire", "hold", "evasive_fire", "standoff_fire")
LABELS = ("Pursuit", "Hold", "Evasive", "Standoff")
COHORTS = {
    "No option": (
        ROOT / "runs/threat_obs_v2_no_option_matched_2m",
        "seed{seed}_no_option_2m"),
    "Short pulse": (
        ROOT / "runs/threat_break_pulse_matched_2m",
        "seed{seed}_pulse_2m"),
    "Long gated option": (
        ROOT / "runs/threat_obs_v2_gated_break_option_ablation",
        "seed{seed}_gated_break_option_2m"),
}
OUTPUT = ROOT / "runs/threat_break_pulse_matched_2m/analysis-output"


def load() -> dict[str, dict[int, dict[str, dict]]]:
    result = {}
    for cohort, (root, pattern) in COHORTS.items():
        result[cohort] = {}
        for seed in SEEDS:
            result[cohort][seed] = {}
            for opponent in OPPONENTS:
                path = (root / pattern.format(seed=seed) / "heldout" /
                        opponent / "summary.json")
                if not path.exists():
                    raise FileNotFoundError(path)
                result[cohort][seed][opponent] = json.loads(
                    path.read_text(encoding="utf-8"))
    return result


def percent(summary: dict, key: str) -> float:
    return 100.0 * float(summary.get(key, 0.0))


def crash_percent(summary: dict) -> float:
    return (100.0 * int(summary.get("learner_death_causes", {}).get(
        "crash", 0)) / int(summary["episodes"]))


def main() -> None:
    data = load()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for cohort in COHORTS:
        for seed in SEEDS:
            for opponent in OPPONENTS:
                summary = data[cohort][seed][opponent]
                rows.append({
                    "cohort": cohort,
                    "seed": seed,
                    "opponent": opponent,
                    "kill_win_pct": percent(summary, "kill_win_rate"),
                    "timeout_pct": percent(summary, "timeout_rate"),
                    "crash_pct": crash_percent(summary),
                    "legal_launch_pct": percent(
                        summary, "legal_launch_acceptance_rate"),
                    "event_conservation": bool(
                        summary["weapon_event_conservation_ok"]),
                })
    with (OUTPUT / "short-pulse-metrics.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    styles = {
        "No option": ("o-", "#0072B2"),
        "Short pulse": ("^-", "#009E73"),
        "Long gated option": ("s--", "#D55E00"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9), sharey=True)
    x = range(len(OPPONENTS))
    for axis, seed in zip(axes, SEEDS):
        for cohort in COHORTS:
            values = [percent(
                data[cohort][seed][opponent], "kill_win_rate")
                for opponent in OPPONENTS]
            style, color = styles[cohort]
            axis.plot(x, values, style, color=color, lw=2, ms=6,
                      label=cohort)
        axis.axhline(80, color="0.7", ls=":", lw=1)
        axis.set_title(f"Seed {seed}")
        axis.set_xticks(list(x), LABELS, rotation=25, ha="right")
        axis.set_ylim(0, 105)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Kill win rate (%)")
    axes[-1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Matched 2M threat-response action ablation")
    fig.tight_layout()
    fig.savefig(OUTPUT / "figure-12-short-pulse-ablation.png", dpi=220)
    fig.savefig(OUTPUT / "figure-12-short-pulse-ablation.pdf")
    plt.close(fig)

    table = []
    accepted = {}
    for cohort in COHORTS:
        evasive_kill = statistics.median(
            percent(data[cohort][seed]["evasive_fire"], "kill_win_rate")
            for seed in SEEDS)
        evasive_timeout = statistics.median(
            percent(data[cohort][seed]["evasive_fire"], "timeout_rate")
            for seed in SEEDS)
        evasive_crash = statistics.median(
            crash_percent(data[cohort][seed]["evasive_fire"])
            for seed in SEEDS)
        easy_worst = min(
            percent(data[cohort][seed][opponent], "kill_win_rate")
            for seed in SEEDS for opponent in
            ("pursuit_fire", "hold", "standoff_fire"))
        legal = all(
            percent(data[cohort][seed][opponent],
                    "legal_launch_acceptance_rate") == 100.0 and
            data[cohort][seed][opponent]["weapon_event_conservation_ok"]
            for seed in SEEDS for opponent in OPPONENTS)
        accepted[cohort] = (
            evasive_kill >= 30.0 and evasive_timeout <= 20.0 and
            evasive_crash <= 2.0 and easy_worst >= 80.0 and legal)
        table.append((cohort, evasive_kill, evasive_timeout, evasive_crash,
                      easy_worst, legal))

    lines = [
        "# Short-pulse matched 2M ablation", "", "## Results", "",
        "| Cohort | Evasive kill median | Evasive timeout median | "
        "Evasive crash median | Worst seed x easy opponent | Weapon contract |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cohort, kill, timeout, crash, easy, legal in table:
        lines.append(
            f"| {cohort} | {kill:.2f}% | {timeout:.2f}% | {crash:.2f}% | "
            f"{easy:.2f}% | {'Pass' if legal else 'Fail'} |")
    pulse_changes = []
    for seed in SEEDS:
        old = percent(data["No option"][seed]["evasive_fire"],
                      "kill_win_rate")
        new = percent(data["Short pulse"][seed]["evasive_fire"],
                      "kill_win_rate")
        pulse_changes.append(f"seed {seed}: {old:.2f}% -> {new:.2f}% "
                             f"({new-old:+.2f} pp)")
    lines.extend([
        "", "Paired no-option to short-pulse Evasive changes: " +
        "; ".join(pulse_changes) + ".", "", "## Decision", "",
        "The short pulse is **" +
        ("accepted" if accepted["Short pulse"] else "rejected") +
        "** under the preregistered all-seed gate: Evasive kill median >=30%, "
        "Evasive timeout <=20%, Evasive crash <=2%, every seed's Pursuit/Hold/"
        "Standoff kill win >=80%, and a perfect weapon contract.", "",
        "R1, weapon logic, dynamics, observation, and training budget are "
        "identical across the matched comparison.",
    ])
    (OUTPUT / "short-pulse-analysis.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
