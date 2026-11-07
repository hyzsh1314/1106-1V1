#!/usr/bin/env python3
"""Validate V12 paired reachability chunks and branch weapon-chain metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from benchmarks.drvoi_v12 import EXECUTION_FEATURE_DIM, write_json_atomic


SUPPORTED_PROTOCOLS = {
    "MRAT-RHPPO-V12-REACH-DATA-2",
    "MRAT-RHPPO-V12-REACH-DATA-3",
    "MRAT-RHPPO-V12-REACH-DATA-4",
}
TARGET_FIELDS = (
    "target_score", "target_legal", "target_first_legal", "target_crash",
    "target_terminal_step", "target_legal_steps", "target_fire_requests",
    "target_accepted_launches", "target_rejected_fire_requests", "target_hits",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(
        (args.data_dir / "dataset-manifest.json").read_text(encoding="utf-8"))
    completed = json.loads(
        (args.data_dir / "COMPLETED.json").read_text(encoding="utf-8"))
    paths = sorted((args.data_dir / "chunks").glob("reachability-*.pt"))
    if not paths:
        raise FileNotFoundError("reachability dataset contains no chunks")
    totals = {key: torch.zeros(2, 5, dtype=torch.long) for key in (
        "legal", "legal_steps", "requests", "accepted", "rejected", "hits")}
    score_sum = torch.zeros(2, 5)
    score_count = better_intercept = rows = 0
    group_keys: set[tuple[int, int]] = set()
    dataset_protocol = manifest["protocol_version"]
    if dataset_protocol not in SUPPORTED_PROTOCOLS:
        raise ValueError(f"unsupported dataset protocol: {dataset_protocol}")
    return_sum = torch.zeros(2, 5)
    timeout_sum = torch.zeros(2, 5, dtype=torch.long)
    return_better_intercept = 0
    has_return_targets = dataset_protocol in {
        "MRAT-RHPPO-V12-REACH-DATA-3",
        "MRAT-RHPPO-V12-REACH-DATA-4",
    }
    for index, path in enumerate(paths):
        if path.name != f"reachability-{index:05d}.pt":
            raise ValueError(f"non-contiguous chunk sequence at {path.name}")
        chunk = torch.load(path, map_location="cpu", weights_only=False)
        if chunk.get("protocol_version") != dataset_protocol:
            raise ValueError(f"protocol mismatch in {path}")
        features = chunk["features"]
        count = features.shape[0]
        if features.shape != (count, EXECUTION_FEATURE_DIM):
            raise ValueError(f"feature shape mismatch in {path}")
        if not bool(torch.isfinite(features).all()):
            raise ValueError(f"non-finite features in {path}")
        for field in TARGET_FIELDS:
            if chunk[field].shape != (count, 2, 5):
                raise ValueError(f"{field} shape mismatch in {path}")
        if not bool(torch.isfinite(chunk["target_score"]).all()):
            raise ValueError(f"non-finite target score in {path}")
        if has_return_targets:
            for field in (
                    "target_discounted_return", "target_terminal_outcome",
                    "target_timeout"):
                if chunk[field].shape != (count, 2, 5):
                    raise ValueError(f"{field} shape mismatch in {path}")
            if not bool(torch.isfinite(
                    chunk["target_discounted_return"]).all()):
                raise ValueError(f"non-finite return target in {path}")
            if bool((chunk["target_terminal_outcome"].abs() > 1).any()):
                raise ValueError(f"invalid terminal outcome in {path}")
            return_sum += chunk["target_discounted_return"].sum(0)
            timeout_sum += chunk["target_timeout"].long().sum(0)
            return_better_intercept += int((
                chunk["target_discounted_return"][:, 1].mean(-1)
                > chunk["target_discounted_return"][:, 0].mean(-1)).sum())
        for field in TARGET_FIELDS[2:]:
            value = chunk[field]
            if value.dtype != torch.bool and bool((value < 0).any()):
                raise ValueError(f"negative branch counter in {field}")
        requests = chunk["target_fire_requests"]
        accepted = chunk["target_accepted_launches"]
        rejected = chunk["target_rejected_fire_requests"]
        if not torch.equal(requests, accepted + rejected):
            raise ValueError(f"weapon request conservation failed in {path}")
        if not torch.equal(
                chunk["target_legal"], chunk["target_legal_steps"] > 0):
            raise ValueError(f"legal opportunity label mismatch in {path}")
        horizon = int(manifest["horizon"])
        if not torch.equal(
                chunk["target_legal"],
                chunk["target_first_legal"] <= horizon):
            raise ValueError(f"first legal censoring mismatch in {path}")
        if int(rejected.sum()) != 0:
            raise ValueError(f"branch produced rejected fire requests in {path}")
        totals["legal"] += chunk["target_legal"].long().sum(0)
        totals["legal_steps"] += chunk["target_legal_steps"].sum(0)
        totals["requests"] += requests.sum(0)
        totals["accepted"] += accepted.sum(0)
        totals["rejected"] += rejected.sum(0)
        totals["hits"] += chunk["target_hits"].sum(0)
        score_sum += chunk["target_score"].sum(0)
        score_count += count
        better_intercept += int((
            chunk["target_score"][:, 1].mean(-1)
            > chunk["target_score"][:, 0].mean(-1)).sum())
        group_keys.update(zip(
            chunk["env_index"].tolist(), chunk["episode_id"].tolist()))
        rows += count
    if rows != int(completed["collected_states"]):
        raise ValueError("chunk row total does not match COMPLETED.json")
    if rows < int(manifest["target_states"]):
        raise ValueError("dataset stopped before target_states")
    if int(totals["requests"][1].sum()) == 0:
        raise ValueError("LOS expert never exercised the weapon chain")
    report = {
        "protocol_version": "MRAT-RHPPO-V12-REACH-DATA-VALIDATION-1",
        "valid": True,
        "chunks": len(paths),
        "states": rows,
        "episode_groups": len(group_keys),
        "teacher_model_decisions": completed["teacher_model_decisions"],
        "online_environment_decisions": completed[
            "online_environment_decisions"],
        "states_per_second": completed["states_per_second"],
        "intercept_better_fraction": better_intercept / rows,
        "discounted_return_intercept_better_fraction": (
            return_better_intercept / rows if has_return_targets else None),
        "score_mean_expert_by_hypothesis": (
            score_sum / score_count).tolist(),
        "discounted_return_mean_expert_by_hypothesis": (
            (return_sum / score_count).tolist()
            if has_return_targets else None),
        "timeout_rate_expert_by_hypothesis": (
            (timeout_sum.float() / rows).tolist()
            if has_return_targets else None),
        "legal_event_rate_expert_by_hypothesis": (
            totals["legal"].float() / rows).tolist(),
        "weapon_chain_expert_by_hypothesis": {
            key: value.tolist() for key, value in totals.items()
            if key != "legal"
        },
    }
    output = args.output or args.data_dir / "validation.json"
    write_json_atomic(output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
