#!/usr/bin/env python3
"""Freeze a compact, hashed audit of the passed V7.1 development gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


T_CRITICAL_95_DF4 = 2.7764451051977987


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def t_interval(values):
    count = len(values)
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    half = T_CRITICAL_95_DF4 * math.sqrt(variance / count)
    return mean, [mean - half, mean + half]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    report = json.loads(args.validation.read_text(encoding="utf-8"))
    if not report["all_development_gates_pass"]:
        raise ValueError("development gate did not pass")
    if not report["confirmation_data_generation_authorized"]:
        raise ValueError("confirmation generation is not authorized")
    if report["confirmation_test_accessed"]:
        raise ValueError("development report already accessed confirmation data")
    primary = report["primary"]
    reactive = report["reactive_positive_control"]
    primary_values = list(primary["delta_nll_by_collection_seed"].values())
    permutation_values = list(
        primary["permutation_gain_by_collection_seed"].values())
    reactive_values = list(
        reactive["delta_nll_by_collection_seed"].values())
    wrong_delay_values = list(
        reactive["correct_minus_wrong_delay_by_collection_seed"].values())
    data = []
    for path in sorted(args.data_dir.glob("*.pt")):
        data.append({
            "path": path.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    if len(data) != 10:
        raise ValueError(f"expected ten development data files, found {len(data)}")
    primary_mean, primary_ci = t_interval(primary_values)
    permutation_mean, permutation_ci = t_interval(permutation_values)
    reactive_mean, reactive_ci = t_interval(reactive_values)
    wrong_mean, wrong_ci = t_interval(wrong_delay_values)
    audit = {
        "suite": "DRVOI-RHPPO-V7.1-W2-development-freeze",
        "protocol_version": report["protocol_version"],
        "scientific_result": False,
        "confirmation_test_accessed": False,
        "collection_seeds": report["collection_seeds"],
        "data_manifest": data,
        "data_total_bytes": sum(item["bytes"] for item in data),
        "validation_report": {
            "path": args.validation.as_posix(),
            "sha256": sha256(args.validation),
        },
        "development_effects": {
            "primary_delta_nll_per_dimension": {
                "by_seed": primary["delta_nll_by_collection_seed"],
                "mean": primary_mean,
                "small_sample_t_ci95": primary_ci,
                "positive_seeds": sum(value > 0 for value in primary_values),
            },
            "primary_permutation_gain_per_dimension": {
                "by_seed": primary["permutation_gain_by_collection_seed"],
                "mean": permutation_mean,
                "small_sample_t_ci95": permutation_ci,
                "positive_seeds": sum(value > 0 for value in permutation_values),
            },
            "reactive_delta_nll_per_dimension": {
                "by_seed": reactive["delta_nll_by_collection_seed"],
                "mean": reactive_mean,
                "small_sample_t_ci95": reactive_ci,
                "positive_seeds": sum(value > 0 for value in reactive_values),
            },
            "reactive_correct_vs_wrong_delay": {
                "by_seed": reactive[
                    "correct_minus_wrong_delay_by_collection_seed"],
                "mean": wrong_mean,
                "small_sample_t_ci95": wrong_ci,
                "positive_seeds": sum(value > 0 for value in wrong_delay_values),
            },
            "single_head_ablation": reactive[
                "single_head_ablation_gain_per_dimension"],
        },
        "gates": report["gates"],
        "all_development_gates_pass": True,
        "confirmation_data_generation_authorized": True,
        "p1q_authorized": False,
        "o0_authorized": False,
        "long_training_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (args.output.parent / "development_validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
