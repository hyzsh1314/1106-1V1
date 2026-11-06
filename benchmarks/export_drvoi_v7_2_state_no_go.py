#!/usr/bin/env python3
"""Freeze the V7.2 state/target no-go without authorizing formal data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path("results/drvoi_rhppo_v7_2/state_sufficiency")
    validation_path = root / "development_validation.json"
    config_path = Path("configs/drvoi_rhppo_v7_2_state_sufficiency.json")
    report = json.loads(validation_path.read_text(encoding="utf-8"))
    if report.get("formal_test_accessed", True):
        raise RuntimeError("V7.2 formal test was accessed")
    if report.get("decision") != "STATE_TARGET_NO_GO":
        raise RuntimeError("V7.2 no-go exporter requires oracle failure")
    representations = {}
    for name, value in report["representations"].items():
        metrics = value["best_validation_metrics"]
        representations[name] = {
            "dimension": value["dimension"],
            "best_epoch": value["best_epoch"],
            "pairwise_accuracy": metrics["pairwise_accuracy"],
            "pairwise_accuracy_ci95": metrics["pairwise_accuracy_ci95"],
            "normalized_argmax_regret": metrics[
                "normalized_argmax_regret"],
            "random_normalized_argmax_regret": metrics[
                "random_normalized_argmax_regret"],
            "all_gates_pass": value["all_gates_pass"],
            "final_train_pairwise_accuracy": value[
                "final_train_metrics"]["pairwise_accuracy"],
            "final_train_normalized_argmax_regret": value[
                "final_train_metrics"]["normalized_argmax_regret"],
        }
    summary = {
        "suite": "DRVOI-RHPPO-V7.2-state-target-no-go",
        "protocol_version": report["protocol_version"],
        "scientific_result": False,
        "formal_test_accessed": False,
        "formal_test_generated": False,
        "train_contexts": report["train_contexts"],
        "validation_contexts": report["validation_contexts"],
        "representations": representations,
        "decision": "STATE_TARGET_NO_GO",
        "diagnosis": (
            "Even the future-belief/full-observation/low-level-hidden oracle "
            "cannot generalize a one-action deviation to full-episode return. "
            "The next revision must decompose the value horizon before any "
            "new response model or P1Q confirmation test."),
        "return_horizon_decomposition_required": True,
        "response_development_authorized": False,
        "formal_test_generation_authorized": False,
        "p1q_authorized": False,
        "o0_authorized": False,
        "long_training_authorized": False,
        "artifact_sha256": {
            "development_validation": sha256(validation_path),
            "config": sha256(config_path),
            "train_seed53185": sha256(root / "train_seed53185_collection.json"),
            "train_seed54185": sha256(root / "train_seed54185_collection.json"),
            "validation_seed55185": sha256(
                root / "validation_seed55185_collection.json"),
        },
    }
    result_path = root / "state_target_no_go.json"
    result_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    freeze = {
        "suite": "DRVOI-RHPPO-V7.2-development-freeze",
        "decision": summary["decision"],
        "formal_test_accessed": False,
        "formal_test_generation_authorized": False,
        "response_development_authorized": False,
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
