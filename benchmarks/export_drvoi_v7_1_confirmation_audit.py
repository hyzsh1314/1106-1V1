#!/usr/bin/env python3
"""Freeze the once-only V7.1 confirmation result and data hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    report = json.loads(args.confirmation.read_text(encoding="utf-8"))
    if report.get("test_access_count") != 1:
        raise ValueError("confirmation test was not accessed exactly once")
    if not report.get("all_confirmation_gates_pass"):
        raise ValueError("confirmation gates did not pass")
    data = [{
        "path": path.as_posix(), "bytes": path.stat().st_size,
        "sha256": sha256(path),
    } for path in sorted(args.data_dir.glob("*.pt"))]
    if len(data) != 10:
        raise ValueError(f"expected ten confirmation files, found {len(data)}")
    audit = {
        "suite": "DRVOI-RHPPO-V7.1-W3-confirmation-freeze",
        "protocol_version": report["protocol_version"],
        "scientific_result": True,
        "test_access_count": 1,
        "confirmation_report": {
            "path": args.confirmation.as_posix(),
            "sha256": sha256(args.confirmation),
        },
        "data_manifest": data,
        "data_total_bytes": sum(item["bytes"] for item in data),
        "source_hashes": {
            name: sha256(Path(path)) for name, path in {
                "collector": "benchmarks/collect_drvoi_v7_1_confirmation_data.py",
                "validator": "benchmarks/validate_drvoi_wvr_history_response_v7_1_confirmation.py",
                "config": "configs/drvoi_rhppo_v7_1_wvr.json",
                "development_authorization":
                    "results/drvoi_rhppo_v7_1/w2/development_audit.json",
            }.items()},
        "primary": {
            "mean_delta_nll_per_dimension": report["primary"][
                "mean_delta_nll_per_dimension"],
            "delta_nll_small_sample_t_ci95": report["primary"][
                "delta_nll_small_sample_t_ci95"],
            "positive_folds": report["primary"]["positive_folds"],
            "mean_permutation_gain_per_dimension": report["primary"][
                "mean_permutation_gain_per_dimension"],
            "permutation_small_sample_t_ci95": report["primary"][
                "permutation_small_sample_t_ci95"],
        },
        "reactive_positive_control": {
            key: report["reactive_positive_control"][key] for key in (
                "mean_delta_nll_per_dimension", "delta_nll_small_sample_t_ci95",
                "mean_correct_vs_wrong_delay_per_dimension",
                "wrong_delay_small_sample_t_ci95")},
        "calibration": report["calibration"],
        "gates": report["gates"],
        "all_confirmation_gates_pass": True,
        "g3_b0_robustness_required": True,
        "p1q_authorized": False,
        "o0_authorized": False,
        "long_training_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (args.output.parent / "confirmation_validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
