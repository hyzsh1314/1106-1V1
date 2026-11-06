"""Analyze the matched 2M no-option control against the gated option."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 85, 127)
OPPONENTS = ("pursuit_fire", "hold", "evasive_fire", "standoff_fire")
LABELS = {
    "pursuit_fire": "Pursuit",
    "hold": "Hold",
    "evasive_fire": "Evasive",
    "standoff_fire": "Standoff",
}
CONTROL = ROOT / "runs/threat_obs_v2_no_option_matched_2m"
OPTION = ROOT / "runs/threat_obs_v2_gated_break_option_ablation"
OUTPUT = CONTROL / "analysis-output"


def summary_path(root: Path, seed: int, opponent: str) -> Path:
    suffix = (f"seed{seed}_no_option_2m" if root == CONTROL else
              f"seed{seed}_gated_break_option_2m")
    return root / suffix / "heldout" / opponent / "summary.json"


def load_cohort(root: Path) -> dict[int, dict[str, dict]]:
    data = {}
    for seed in SEEDS:
        data[seed] = {}
        for opponent in OPPONENTS:
            path = summary_path(root, seed, opponent)
            if not path.exists():
                raise FileNotFoundError(path)
            data[seed][opponent] = json.loads(path.read_text(encoding="utf-8"))
    return data


def pct(summary: dict, key: str) -> float:
    return 100.0 * float(summary.get(key, 0.0))


def crash_pct(summary: dict) -> float:
    crashes = int(summary.get("learner_death_causes", {}).get("crash", 0))
    return 100.0 * crashes / int(summary["episodes"])


def main() -> None:
    control = load_cohort(CONTROL)
    option = load_cohort(OPTION)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in SEEDS:
        for opponent in OPPONENTS:
            for cohort, data in (("No option 2M", control),
                                 ("Gated option 2M", option)):
                summary = data[seed][opponent]
                rows.append({
                    "cohort": cohort,
                    "seed": seed,
                    "opponent": opponent,
                    "kill_win_pct": pct(summary, "kill_win_rate"),
                    "timeout_pct": pct(summary, "timeout_rate"),
                    "crash_pct": crash_pct(summary),
                    "legal_launch_pct": pct(
                        summary, "legal_launch_acceptance_rate"),
                    "event_conservation": bool(
                        summary["weapon_event_conservation_ok"]),
                })
    with (OUTPUT / "matched-control-metrics.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharey=True)
    x = range(len(OPPONENTS))
    for axis, seed in zip(axes, SEEDS):
        control_values = [
            pct(control[seed][opponent], "kill_win_rate")
            for opponent in OPPONENTS]
        option_values = [
            pct(option[seed][opponent], "kill_win_rate")
            for opponent in OPPONENTS]
        axis.plot(x, control_values, "o-", lw=2, label="No option 2M")
        axis.plot(x, option_values, "s--", lw=2, label="Gated option 2M")
        axis.axhline(80.0, color="0.7", ls=":", lw=1)
        axis.set_title(f"Seed {seed}")
        axis.set_xticks(list(x), [LABELS[name] for name in OPPONENTS],
                        rotation=25, ha="right")
        axis.set_ylim(0, 105)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Kill win rate (%)")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.suptitle("Matched 2M continuation: gated option versus no option")
    fig.tight_layout()
    fig.savefig(OUTPUT / "figure-11-matched-option-control.png", dpi=220)
    fig.savefig(OUTPUT / "figure-11-matched-option-control.pdf")
    plt.close(fig)

    def med(data, opponent, metric):
        function = crash_pct if metric == "crash" else lambda item: pct(
            item, metric)
        return statistics.median(
            function(data[seed][opponent]) for seed in SEEDS)

    all_legal = all(
        row["legal_launch_pct"] == 100.0 and row["event_conservation"]
        for row in rows)
    control_easy_worst = min(
        pct(control[seed][opponent], "kill_win_rate")
        for seed in SEEDS
        for opponent in ("pursuit_fire", "hold", "standoff_fire"))
    option_easy_worst = min(
        pct(option[seed][opponent], "kill_win_rate")
        for seed in SEEDS
        for opponent in ("pursuit_fire", "hold", "standoff_fire"))
    control_evasive = med(control, "evasive_fire", "kill_win_rate")
    option_evasive = med(option, "evasive_fire", "kill_win_rate")
    control_timeout = med(control, "evasive_fire", "timeout_rate")
    option_timeout = med(option, "evasive_fire", "timeout_rate")
    control_crash = med(control, "evasive_fire", "crash")
    option_crash = med(option, "evasive_fire", "crash")

    per_seed = []
    for seed in SEEDS:
        c = pct(control[seed]["evasive_fire"], "kill_win_rate")
        o = pct(option[seed]["evasive_fire"], "kill_win_rate")
        per_seed.append(f"seed {seed}: {c:.2f}% -> {o:.2f}% ({o-c:+.2f} pp)")

    report = f"""# Matched 2M no-option control

## Design

Both cohorts resume the same 1.5M, ±90-degree, R1 checkpoints and receive the
same additional 0.5M steps against 100% Evasive.  They differ only in whether
the TA-gated stateful break option is enabled.  Each checkpoint is evaluated
for 192 side-balanced episodes per opponent with identical evaluation seeds.

## Results

| Metric | No option 2M | Gated option 2M |
|---|---:|---:|
| Evasive kill-win median | {control_evasive:.2f}% | {option_evasive:.2f}% |
| Evasive timeout median | {control_timeout:.2f}% | {option_timeout:.2f}% |
| Evasive crash median | {control_crash:.2f}% | {option_crash:.2f}% |
| Worst seed x easy-opponent kill win | {control_easy_worst:.2f}% | {option_easy_worst:.2f}% |
| Legal launch 100% and event conservation | {'Yes' if all_legal else 'No'} | {'Yes' if all_legal else 'No'} |

Paired Evasive changes: {'; '.join(per_seed)}.

## Decision

The matched control removes the previous training-budget confound.  The gated
option is accepted only if Evasive improves without reducing any seed's
Pursuit, Hold, or Standoff kill-win rate below 80%.  Under this preregistered
gate the option is **{'accepted' if option_easy_worst >= 80.0 and option_evasive >= 30.0 and option_timeout <= 20.0 else 'rejected'}**.

The no-option continuation is **{'accepted' if control_easy_worst >= 80.0 and control_evasive >= 30.0 and control_timeout <= 20.0 and control_crash <= 2.0 else 'not accepted'}** as the final curriculum.  Reward, weapon, and dynamics remain frozen; any next intervention should target a short, interruptible threat-response action rather than changing R1.
"""
    (OUTPUT / "matched-control-analysis.md").write_text(
        report, encoding="utf-8")


if __name__ == "__main__":
    main()
