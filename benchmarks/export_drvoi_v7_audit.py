#!/usr/bin/env python3
"""Export compact, tracked V7 audit artifacts and hashes from local runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path


SMALL_ARTIFACTS = {
    "validation.json": "p1-response-formal/validation.json",
    "response_gate.json": "p1-response-formal/response_gate.json",
    "calibration_gate.json": "p1-response-formal/calibration_gate.json",
    "run_manifest.json": "p1-response-formal/manifest.json",
    "b0_summary.json": "b0-1m/summary.json",
}


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path,
                        default=Path("runs/drvoi_rhppo_v7"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/drvoi_rhppo_v7"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for target, source in SMALL_ARTIFACTS.items():
        shutil.copy2(args.run_root / source, args.output / target)

    validation = json.loads((args.output / "validation.json").read_text(
        encoding="utf-8"))
    results = list(validation["sensitivity"]) + list(
        validation["reactive_positive_control"])
    sensitivity_rows = []
    opponent_rows = []
    for result in results:
        row = {
            "dataset": result["dataset"], "branch": result["branch"],
            "delta": result["delta"], "context": result["context"],
            "train_samples": result["train_samples"],
            "test_samples": result["test_samples"],
            "test_episodes": result["test_episodes"],
            "delta_nll_per_dim": result["mean_delta_nll_per_dimension"],
            "delta_nll_ci_low": result["delta_nll_ci95"][0],
            "delta_nll_ci_high": result["delta_nll_ci95"][1],
            "permutation_gain_per_dim": result["permutation_gain_per_dimension"],
            "permutation_ci_low": result["permutation_gain_ci95"][0],
            "permutation_ci_high": result["permutation_gain_ci95"][1],
            "best_epoch": result["training"]["best_epoch"],
            "epochs_executed": result["training"]["epochs_executed"],
            "raw_gradient_max": result["training"]["raw_gradient_norm_max"],
            "gradient_fault": result["training"]["gradient_fault_over_1000"],
        }
        sensitivity_rows.append(row)
        for opponent, value in result["delta_nll_by_opponent"].items():
            opponent_rows.append({
                "dataset": result["dataset"], "branch": result["branch"],
                "delta": result["delta"], "context": result["context"],
                "opponent_id": opponent, "delta_nll_per_dim": value,
                "permutation_gain_per_dim": result[
                    "permutation_gain_by_opponent"][opponent],
            })
    write_csv(args.output / "sensitivity_summary.csv",
              list(sensitivity_rows[0]), sensitivity_rows)
    write_csv(args.output / "opponent_summary.csv",
              list(opponent_rows[0]), opponent_rows)

    calibration = validation["primary"]["calibration"]
    dimension_rows = []
    for index in range(len(calibration["nll_per_dimension"])):
        dimension_rows.append({
            "dimension": index,
            "nll": calibration["nll_per_dimension"][index],
            "rmse": calibration["rmse_per_dimension"][index],
            "coverage_50": calibration["coverage_50_by_dimension"][index],
            "coverage_90": calibration["coverage_90_by_dimension"][index],
            "coverage_95": calibration["coverage_95_by_dimension"][index],
            "pit_ks": calibration["pit_ks_by_dimension"][index],
        })
    write_csv(args.output / "main_calibration_by_dimension.csv",
              list(dimension_rows[0]), dimension_rows)

    data_rows = []
    for report_path in sorted((args.run_root / "p1-response-formal/data").glob(
            "*.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        data_rows.append({
            "dataset": report["dataset"], "collection_seed": report["seed"],
            "b0_seed": report["b0_seed"], "episodes": report["episodes"],
            "transitions": report["transitions"],
            "collector_sha256": report["collector_sha256"],
            "protocol_config_sha256": report["protocol_config_sha256"],
            "all_end_done": report["complete_episode_invariants"]["all_end_done"],
            "no_early_done": report["complete_episode_invariants"]["no_early_done"],
            "unique_episode_ids": report[
                "complete_episode_invariants"]["unique_episode_ids"],
        })
    write_csv(args.output / "data_by_collection_seed.csv",
              list(data_rows[0]), data_rows)

    inventory_paths = []
    inventory_paths.extend(sorted(
        (args.run_root / "p1-response-formal/data").glob("*.pt")))
    inventory_paths.extend(sorted(
        (args.run_root / "p1-response-formal/data").glob("*.json")))
    inventory_paths.append(args.run_root / "p1-response-formal/validation.pt")
    inventory_paths.extend(sorted(
        (args.run_root / "b0-1m").glob("seed*/latest.pt")))
    inventory = []
    for path in inventory_paths:
        inventory.append({
            "path": path.as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256(path), "tracked": False,
        })
    for path in sorted(args.output.glob("*")):
        if path.name == "artifact_manifest.json":
            continue
        inventory.append({
            "path": path.as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256(path), "tracked": True,
        })
    manifest = {
        "schema": "drvoi-rhppo-v7-audit-artifacts-1",
        "fixed_commit": "73733ddcbfe48b04b9c267d4f86e8bda976d2896",
        "scientific_result": False,
        "gate_outcome": "P1R_failed_P1C_passed",
        "artifacts": inventory,
    }
    (args.output / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output), "artifacts": len(inventory),
        "untracked_bytes_hashed": sum(
            item["bytes"] for item in inventory if not item["tracked"]),
    }, indent=2))


if __name__ == "__main__":
    main()
