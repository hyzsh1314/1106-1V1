#!/usr/bin/env python3
"""Freeze the failed V7.1 P1Q development gate without opening test data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compact(report, metrics_key="validation_metrics"):
    metrics = report[metrics_key]
    return {
        "pairwise_accuracy": metrics["pairwise_accuracy"],
        "pairwise_accuracy_ci95": metrics["pairwise_accuracy_ci95"],
        "normalized_argmax_regret": metrics["normalized_argmax_regret"],
        "random_normalized_argmax_regret": metrics[
            "random_normalized_argmax_regret"],
        "all_gates_pass": bool(report.get(
            "all_development_gates_pass", report.get("all_mc_gates_pass", False))),
    }


def main():
    root = Path("results/drvoi_rhppo_v7_1/p1q")
    root.mkdir(parents=True, exist_ok=True)
    source = {
        "hot": Path("results/drvoi_rhppo_v7_1/p1q/hot_development_validation.json"),
        "w2_td": Path("results/drvoi_rhppo_v7_1/p1q/joint_td_development.json"),
        "full_episode_probe": Path(
            "results/drvoi_rhppo_v7_1/p1q/factorized_full_episode_mc_probe.json"),
        "factorized_192": Path(
            "results/drvoi_rhppo_v7_1/p1q/factorized_mc_calibration_failure.json"),
        "joint_192": Path(
            "results/drvoi_rhppo_v7_1/p1q/joint_mc_calibration_192ctx_failure.json"),
        "joint_768": Path(
            "results/drvoi_rhppo_v7_1/p1q/joint_mc_calibration_768ctx_failure.json"),
        "config": Path("configs/drvoi_rhppo_v7_1_p1q.json"),
    }
    missing = [str(path) for path in source.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing P1Q audit inputs: {missing}")
    reports = {key: load(path) for key, path in source.items()
               if path.suffix == ".json"}
    if reports["joint_768"].get("formal_q_test_accessed", True):
        raise RuntimeError("formal Q test was accessed during development")
    if reports["joint_768"].get("all_development_gates_pass", True):
        raise RuntimeError("cannot export no-go from a passing development run")
    summary = {
        "suite": "DRVOI-RHPPO-V7.1-P1Q-development-no-go",
        "protocol_version": reports["joint_768"]["protocol_version"],
        "scientific_result": False,
        "formal_q_test_accessed": False,
        "formal_test_generated": False,
        "hot_passed": bool(reports["hot"].get(
            "wvr_q_test_generation_authorized", False)),
        "td_development": {
            "architecture": reports["w2_td"]["architecture"],
            "selected_target_update_epochs": reports["w2_td"][
                "selected_target_update_epochs"],
            "development_semantics_pass": reports["w2_td"][
                "development_semantics_pass"],
            "episode_spearman": reports["w2_td"]["selected_validation"][
                "episode_spearman"],
        },
        "full_episode_probe": compact(reports["full_episode_probe"], "metrics"),
        "repair_sequence": {
            "factorized_dueling_192_contexts": compact(
                reports["factorized_192"]),
            "joint_dueling_192_contexts": compact(reports["joint_192"]),
            "joint_dueling_768_contexts": compact(reports["joint_768"]),
        },
        "failure_diagnosis": (
            "The registered belief/action/12-D response context does not "
            "generalize same-state counterfactual action value. Increasing "
            "MC training contexts fourfold did not improve held-out ranking."),
        "decision": "P1Q_DEVELOPMENT_NO_GO",
        "formal_test_generation_authorized": False,
        "p1q_authorized": False,
        "o0_authorized": False,
        "short_training_authorized": False,
        "long_training_authorized": False,
        "required_next_revision": (
            "A new pre-registered response/state-sufficiency revision is "
            "required before any new P1Q confirmation test."),
        "artifact_sha256": {key: sha256(path) for key, path in source.items()},
    }
    (root / "p1q_development_no_go.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    freeze = {
        "suite": "DRVOI-RHPPO-V7.1-P1Q-development-freeze",
        "decision": summary["decision"],
        "formal_q_test_accessed": False,
        "formal_test_generation_authorized": False,
        "o0_authorized": False,
        "long_training_authorized": False,
        "source": "results/drvoi_rhppo_v7_1/p1q/p1q_development_no_go.json",
        "source_sha256": sha256(root / "p1q_development_no_go.json"),
    }
    (root / "development_freeze.json").write_text(
        json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
