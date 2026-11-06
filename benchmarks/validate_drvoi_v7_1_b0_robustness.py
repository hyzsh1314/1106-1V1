#!/usr/bin/env python3
"""Validate V7.1 response-direction robustness under B0 seeds 185/285."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from benchmarks.validate_drvoi_wvr_history_response_v7_1_dev import (
    PROTOCOL,
    evaluate_dataset,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b0-seed185", type=Path, nargs="+", required=True)
    parser.add_argument("--b0-seed285", type=Path, nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--seed", type=int, default=39185)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_group(paths, expected_b0_seed):
    expected_collector = sha256(Path(
        "benchmarks/collect_drvoi_v7_1_b0_robustness_data.py"))
    expected_config = sha256(Path("configs/drvoi_rhppo_v7_1_wvr.json"))
    expected_authorization = sha256(Path(
        "results/drvoi_rhppo_v7_1/w3/confirmation_audit.json"))
    episodes = []
    collection_seeds = set()
    ids = set()
    for path_index, path in enumerate(paths):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        report = payload["report"]
        if report.get("protocol_version") != PROTOCOL:
            raise ValueError(f"wrong protocol in {path}")
        if int(report.get("b0_seed", -1)) != expected_b0_seed:
            raise ValueError(f"wrong B0 seed in {path}")
        if report.get("collector_sha256") != expected_collector:
            raise ValueError(f"wrong robustness collector in {path}")
        if report.get("protocol_config_sha256") != expected_config:
            raise ValueError(f"wrong config in {path}")
        if report.get("confirmation_authorization_sha256") != expected_authorization:
            raise ValueError(f"wrong confirmation authorization in {path}")
        if report.get("contains_confirmation_test"):
            raise ValueError(f"robustness data contains confirmation test in {path}")
        if set(report.get("available_splits", ())) != {"train", "validation"}:
            raise ValueError(f"wrong robustness splits in {path}")
        collection_seeds.add(int(report["collection_seed"]))
        for episode in payload["episodes"]:
            key = (int(episode["collection_seed"]), int(episode["episode_id"]))
            if key in ids:
                raise ValueError(f"duplicate robustness episode {key}")
            ids.add(key)
            copied = dict(episode)
            copied["episode_id"] = path_index * 10**9 + int(episode["episode_id"])
            episodes.append(copied)
    if len(collection_seeds) != 5:
        raise ValueError("robustness group requires all five collection seeds")
    return episodes, sorted(collection_seeds)


def group_pass(result):
    delta = list(result["delta_nll_by_collection_seed"].values())
    permutation = list(result["permutation_gain_by_collection_seed"].values())
    return (result["mean_delta_nll_per_dimension"] > 0
            and sum(value > 0 for value in delta) >= 4
            and result["permutation_gain_per_dimension"] > 0
            and sum(value > 0 for value in permutation) >= 4
            and result["training"]["initial_function_max_difference"] <= 1e-7)


def numerical_diagnostic(result):
    events = result["training"].get("gradient_fault_events", [])
    return {
        "raw_gradient_norm_max": result["training"]["raw_gradient_norm_max"],
        "gradient_fault_over_1000": result["training"]["gradient_fault_over_1000"],
        "gradient_fault_events": events,
        "sigma_boundary_fraction": result["sigma_boundary_fraction"],
        "hard_gate": False,
        "reason": (
            "G3 is pre-registered as a direction-robustness gate; G0 was "
            "already passed by the once-only confirmation models"),
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    episodes185, seeds185 = load_group(args.b0_seed185, 185)
    episodes285, seeds285 = load_group(args.b0_seed285, 285)
    if seeds185 != seeds285:
        raise ValueError("collection seeds differ between B0 robustness groups")
    result185, artifact185 = evaluate_dataset(episodes185, "rel", args)
    args.seed += 1009
    result285, artifact285 = evaluate_dataset(episodes285, "rel", args)
    gates = {
        "G3_B0_seed185_direction": {"pass": group_pass(result185)},
        "G3_B0_seed285_direction": {"pass": group_pass(result285)},
    }
    all_pass = all(value["pass"] for value in gates.values())
    report = {
        "suite": "DRVOI-RHPPO-V7.1-G3-B0-robustness",
        "protocol_version": PROTOCOL,
        "scientific_result": False,
        "confirmation_test_accessed": False,
        "collection_seeds": seeds185,
        "B0_seed185": result185,
        "B0_seed285": result285,
        "gates": gates,
        "numerical_diagnostics": {
            "B0_seed185": numerical_diagnostic(result185),
            "B0_seed285": numerical_diagnostic(result285),
        },
        "all_g3_gates_pass": all_pass,
        "p1q_authorized": all_pass,
        "o0_authorized": False,
        "long_training_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    torch.save({"B0_seed185": artifact185, "B0_seed285": artifact285},
               args.output.with_suffix(".pt"))
    print(json.dumps(report, indent=2))
    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
