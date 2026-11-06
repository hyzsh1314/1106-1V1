#!/usr/bin/env python3
"""Freeze V7.3 return-horizon development without opening formal test."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path("results/drvoi_rhppo_v7_3/return_horizon")
    validation_path = root / "development_validation.json"
    config_path = Path("configs/drvoi_rhppo_v7_3_return_horizon.json")
    report = json.loads(validation_path.read_text(encoding="utf-8"))
    if report.get("formal_test_accessed", True):
        raise RuntimeError("V7.3 formal test was accessed")
    if report.get("decision") != "RETURN_TARGET_NO_GO":
        raise RuntimeError("return no-go exporter requires target failure")
    matrix = {}
    for horizon, representations in report["results"].items():
        matrix[horizon] = {}
        for name, value in representations.items():
            metrics = value["best_validation_metrics"]
            matrix[horizon][name] = {
                "pairwise_accuracy": metrics["pairwise_accuracy"],
                "pairwise_accuracy_ci95": metrics["pairwise_accuracy_ci95"],
                "normalized_argmax_regret": metrics[
                    "normalized_argmax_regret"],
                "random_normalized_argmax_regret": metrics[
                    "random_normalized_argmax_regret"],
                "all_gates_pass": value["all_gates_pass"],
            }
    summary = {
        "suite": "DRVOI-RHPPO-V7.3-return-target-no-go",
        "protocol_version": report["protocol_version"],
        "scientific_result": False,
        "formal_test_accessed": False,
        "formal_test_generated": False,
        "train_contexts": report["train_contexts"],
        "validation_contexts": report["validation_contexts"],
        "matrix": matrix,
        "decision": "RETURN_TARGET_NO_GO",
        "selected_horizon": None,
        "diagnosis": (
            "The 12-decision target improves pairwise ordering but fails "
            "argmax regret; 30/120/full targets progressively lose action "
            "identifiability. No horizon satisfies both the oracle and "
            "response-predictable representation gates."),
        "response_target_development_authorized": False,
        "formal_test_generation_authorized": False,
        "p1q_authorized": False,
        "o0_authorized": False,
        "long_training_authorized": False,
        "artifact_sha256": {
            "development_validation": sha256(validation_path),
            "config": sha256(config_path),
            "train_seed63185": sha256(root / "train_seed63185_collection.json"),
            "train_seed64185": sha256(root / "train_seed64185_collection.json"),
            "validation_seed65185": sha256(
                root / "validation_seed65185_collection.json"),
        },
    }
    result_path = root / "return_target_no_go.json"
    result_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    freeze = {
        "suite": "DRVOI-RHPPO-V7.3-development-freeze",
        "decision": summary["decision"],
        "selected_horizon": None,
        "formal_test_accessed": False,
        "formal_test_generation_authorized": False,
        "response_target_development_authorized": False,
        "p1q_authorized": False,
        "o0_authorized": False,
        "long_training_authorized": False,
        "source": str(result_path).replace("\\", "/"),
        "source_sha256": sha256(result_path),
    }
    (root / "development_freeze.json").write_text(
        json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
