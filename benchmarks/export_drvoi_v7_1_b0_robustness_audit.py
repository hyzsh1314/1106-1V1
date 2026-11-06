#!/usr/bin/env python3
"""Freeze V7.1 G3 B0 robustness results and source/data hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robustness", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.robustness.read_text(encoding="utf-8"))
    if not report.get("all_g3_gates_pass") or not report.get("p1q_authorized"):
        raise ValueError("G3 did not authorize P1Q")
    files = [{
        "path": path.as_posix(), "bytes": path.stat().st_size,
        "sha256": sha256(path),
    } for path in sorted(args.data_dir.glob("*.pt"))]
    if len(files) != 10:
        raise ValueError("G3 requires ten robustness data files")
    audit = {
        "suite": "DRVOI-RHPPO-V7.1-W4-G3-freeze",
        "protocol_version": report["protocol_version"],
        "scientific_result": False,
        "confirmation_test_accessed": False,
        "robustness_report_sha256": sha256(args.robustness),
        "data_manifest": files,
        "data_total_bytes": sum(item["bytes"] for item in files),
        "B0_seed185": {
            key: report["B0_seed185"][key] for key in (
                "mean_delta_nll_per_dimension", "delta_nll_by_collection_seed",
                "permutation_gain_per_dimension",
                "permutation_gain_by_collection_seed")},
        "B0_seed285": {
            key: report["B0_seed285"][key] for key in (
                "mean_delta_nll_per_dimension", "delta_nll_by_collection_seed",
                "permutation_gain_per_dimension",
                "permutation_gain_by_collection_seed")},
        "numerical_diagnostics": report["numerical_diagnostics"],
        "gates": report["gates"],
        "all_g3_gates_pass": True,
        "p1q_authorized": True,
        "o0_authorized": False,
        "long_training_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (args.output.parent / "b0_robustness_validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
