"""Create a publication-ready reward diagnostic for the 1.0076M Torch run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=5)
    return parser.parse_args()


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full_like(values, np.nan, dtype=float)
    for index in range(values.size):
        start = max(0, index - window + 1)
        result[index] = np.mean(values[start:index + 1])
    return result


def main():
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    history = payload.get("update_history", [])
    if not history:
        raise ValueError("input does not contain update_history")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(history[0])
    csv_path = args.output_dir / "reward-history.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)

    steps = np.asarray([row["environment_decisions"] for row in history]) / 1e6
    absolute = np.asarray([row["mean_abs_reward"] for row in history])
    red = np.asarray([row["mean_red_reward"] for row in history])
    blue = np.asarray([row["mean_blue_reward"] for row in history])
    potential = np.asarray([
        row["mean_posture_potential"] for row in history])
    for name, values in {
            "steps": steps, "absolute reward": absolute,
            "red reward": red, "blue reward": blue,
            "posture potential": potential}.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"non-finite {name} values")

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 6.5,
    })
    blue_dark = "#2D5F8B"
    blue_light = "#91B7D5"
    red_color = "#C85C5C"
    neutral = "#707070"
    green = "#4E8A72"

    fig, axes = plt.subplots(
        3, 1, figsize=(183 / 25.4, 135 / 25.4), sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.0, 1.0], "hspace": 0.18})
    ax = axes[0]
    ax.plot(steps, absolute, color=blue_light, lw=0.8, marker="o",
            ms=2.3, label="Per-update mean")
    ax.plot(steps, moving_average(absolute, args.window), color=blue_dark,
            lw=1.7, label=f"{args.window}-update moving mean")
    ax.set_yscale("log")
    ax.set_ylabel("Mean |PBRS reward|\nper transition")
    ax.grid(axis="y", color="#E6E6E6", lw=0.6)
    ax.legend(loc="upper right")
    reduction = 100.0 * (1.0 - absolute[-1] / absolute[0])
    ax.annotate(
        f"{reduction:.1f}% decrease", xy=(steps[-1], absolute[-1]),
        xytext=(0.61, 0.62), textcoords="axes fraction", color=blue_dark,
        arrowprops={"arrowstyle": "-", "color": blue_dark, "lw": 0.8})

    ax = axes[1]
    scale = 1e6
    ax.axhline(0.0, color=neutral, lw=0.7, ls="--")
    ax.plot(steps, red * scale, color=red_color, lw=1.0,
            marker="o", ms=2.0, label="Red")
    ax.plot(steps, blue * scale, color=blue_dark, lw=1.0,
            marker="s", ms=1.8, label="Blue")
    ax.set_ylabel("Signed mean reward\n($\\times 10^{-6}$)")
    ax.grid(axis="y", color="#E6E6E6", lw=0.6)
    ax.legend(loc="upper right", ncol=2)
    ax.text(0.01, 0.08, "Zero-sum: Red + Blue = 0",
            transform=ax.transAxes, color=neutral)

    ax = axes[2]
    ax.plot(steps, potential, color="#9BC3AE", lw=0.8,
            marker="o", ms=2.3, label="Per-update mean")
    ax.plot(steps, moving_average(potential, args.window), color=green,
            lw=1.7, label=f"{args.window}-update moving mean")
    ax.set_ylabel("Mean posture\npotential")
    ax.set_xlabel("Environment decisions ($\\times 10^6$)")
    ax.grid(axis="y", color="#E6E6E6", lw=0.6)
    ax.legend(loc="upper right")

    for label, axis in zip(("a", "b", "c"), axes):
        axis.text(-0.105, 1.04, label, transform=axis.transAxes,
                  fontsize=8, fontweight="bold", va="bottom")
        axis.margins(x=0.01)
    fig.align_ylabels(axes)
    stem = args.output_dir / "torch-ppo-1m-reward-convergence"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "source": str(args.input),
        "updates": len(history),
        "environment_decisions": int(history[-1]["environment_decisions"]),
        "moving_average_window_updates": args.window,
        "initial_mean_abs_reward": float(absolute[0]),
        "final_mean_abs_reward": float(absolute[-1]),
        "absolute_reward_reduction_percent": float(reduction),
        "last_10_mean_abs_reward": float(np.mean(absolute[-10:])),
        "initial_mean_posture_potential": float(potential[0]),
        "final_mean_posture_potential": float(potential[-1]),
        "zero_sum_max_error": float(np.max(np.abs(red + blue))),
        "interpretation": (
            "Reward magnitude rapidly collapses and then plateaus while the "
            "posture potential also declines; this is stabilization of a "
            "low-change, low-potential regime, not evidence of improved "
            "combat performance or reward maximization."),
        "single_seed": True,
        "uncertainty_band": None,
    }
    (args.output_dir / "reward-curve-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
