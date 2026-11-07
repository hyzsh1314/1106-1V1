#!/usr/bin/env python3
"""Build the strict descriptive V9--V11.3 analysis bundle figures."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "analysis" / "drvoi-rhppo-v11"
METHODS = {
    "V9": ROOT / "runs/drvoi_rhppo_v9/coach-online/development-1m-seed85",
    "V9.1": ROOT / "runs/drvoi_rhppo_v9_1/coach-highlow/development-klstop-1m-seed85",
    "V10": ROOT / "runs/drvoi_rhppo_v10/barc/development-1m-seed85",
    "V11": ROOT / "runs/drvoi_rhppo_v11/race/frozen-v9-seed85",
    "V11.1": ROOT / "runs/drvoi_rhppo_v11_1/race-hysteretic/frozen-v9-seed85",
    "V11.2": ROOT / "runs/drvoi_rhppo_v11_2/race-episode-latched/frozen-v9-seed85",
    "V11.3": ROOT / "runs/drvoi_rhppo_v11_3/race-sustained/frozen-v9-seed85",
}
OPPONENTS = ("hold", "pursuit", "evasive")
COLORS = (
    "#0072B2", "#56B4E9", "#E69F00", "#CC79A7",
    "#009E73", "#D55E00", "#000000",
)
HATCHES = ("", "//", "xx", "..", "\\\\", "++", "oo")


def wilson(successes: int, total: int, z: float = 1.959963984540054):
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def load_rows():
    rows = []
    for method, directory in METHODS.items():
        for opponent in OPPONENTS:
            report = json.loads(
                (directory / f"eval-{opponent}.json").read_text(
                    encoding="utf-8"))
            lower, upper = wilson(int(report["wins"]), int(report["episodes"]))
            rows.append({
                "method": method,
                "opponent": opponent,
                "episodes": int(report["episodes"]),
                "wins": int(report["wins"]),
                "losses": int(report["losses"]),
                "draws": int(report["draws"]),
                "win_rate": float(report["win_rate"]),
                "wilson_low": lower,
                "wilson_high": upper,
                "timeout_rate": float(report["timeout_rate"]),
                "hit_per_launch": float(report["hit_per_launch"]),
                "accepted_launches": int(report["accepted_launches"]),
                "action_intervention_fraction": report.get(
                    "action_intervention_fraction"),
            })
    return rows


def save_csv(rows):
    fields = list(rows[0])
    with (OUTPUT / "exact-summary.csv").open(
            "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main_comparison(rows):
    figure, axis = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    x = np.arange(len(OPPONENTS), dtype=float)
    width = 0.11
    for index, (method, color, hatch) in enumerate(
            zip(METHODS, COLORS, HATCHES)):
        selected = [
            next(row for row in rows
                 if row["method"] == method and row["opponent"] == opponent)
            for opponent in OPPONENTS]
        values = np.asarray([row["win_rate"] for row in selected]) * 100.0
        lower = np.asarray([row["wilson_low"] for row in selected]) * 100.0
        upper = np.asarray([row["wilson_high"] for row in selected]) * 100.0
        positions = x + (index - 3.0) * width
        axis.bar(
            positions, values, width=width, color=color, hatch=hatch,
            edgecolor="black", linewidth=0.45, label=method)
        axis.errorbar(
            positions, values, yerr=np.maximum(
                np.vstack((values - lower, upper - values)), 0.0),
            fmt="none", ecolor="black", elinewidth=0.7, capsize=1.5)
    axis.set_xticks(x, ("Hold", "Pursuit", "Evasive"))
    axis.set_ylabel("Win rate (%)")
    axis.set_ylim(0.0, 100.0)
    axis.grid(axis="y", alpha=0.25, linewidth=0.7)
    axis.legend(ncol=4, fontsize=7.5, frameon=False, loc="upper center")
    for suffix in ("pdf", "png"):
        figure.savefig(
            OUTPUT / "figures" / f"figure-01-fixed-opponent-win-rates.{suffix}",
            dpi=300)
    plt.close(figure)


def tradeoff(rows):
    figure, axis = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    for method, color in zip(METHODS, COLORS):
        selected = {
            row["opponent"]: row for row in rows if row["method"] == method}
        easy = 50.0 * (
            selected["hold"]["win_rate"]
            + selected["pursuit"]["win_rate"])
        hard = 100.0 * selected["evasive"]["win_rate"]
        axis.scatter(easy, hard, s=55, color=color, edgecolor="black",
                     linewidth=0.5, zorder=3)
        axis.annotate(method, (easy, hard), xytext=(4, 4),
                      textcoords="offset points", fontsize=8)
    axis.set_xlabel("Mean Hold/Pursuit win rate (%)")
    axis.set_ylabel("Evasive win rate (%)")
    axis.set_xlim(45.0, 90.0)
    axis.set_ylim(-2.0, 30.0)
    axis.grid(alpha=0.25, linewidth=0.7)
    for suffix in ("pdf", "png"):
        figure.savefig(
            OUTPUT / "figures" / f"figure-02-easy-hard-tradeoff.{suffix}",
            dpi=300)
    plt.close(figure)


def intervention_breakdown(rows):
    methods = ("V11", "V11.1", "V11.2", "V11.3")
    figure, axis = plt.subplots(figsize=(6.2, 3.5), constrained_layout=True)
    x = np.arange(len(OPPONENTS), dtype=float)
    width = 0.18
    for index, method in enumerate(methods):
        values = [
            100.0 * float(next(
                row["action_intervention_fraction"] for row in rows
                if row["method"] == method and row["opponent"] == opponent))
            for opponent in OPPONENTS]
        axis.bar(
            x + (index - 1.5) * width, values, width=width,
            label=method, color=COLORS[index + 3], edgecolor="black",
            linewidth=0.45, hatch=HATCHES[index + 3])
    axis.set_xticks(x, ("Hold", "Pursuit", "Evasive"))
    axis.set_ylabel("Action intervention (%)")
    axis.set_ylim(0.0, 100.0)
    axis.grid(axis="y", alpha=0.25, linewidth=0.7)
    axis.legend(ncol=4, fontsize=8, frameon=False, loc="upper center")
    for suffix in ("pdf", "png"):
        figure.savefig(
            OUTPUT / "figures" / f"figure-03-intervention-fraction.{suffix}",
            dpi=300)
    plt.close(figure)


def main():
    (OUTPUT / "figures").mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    save_csv(rows)
    main_comparison(rows)
    tradeoff(rows)
    intervention_breakdown(rows)


if __name__ == "__main__":
    main()
