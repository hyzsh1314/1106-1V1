#!/usr/bin/env python3
"""Freeze the V7--V12 evidence and executable identity for MRAT-RHPPO."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EVIDENCE_PATHS = (
    "docs/DRVOI_RHPPO_V7执行结果_20261106.md",
    "docs/analysis/drvoi-rhppo-v8/analysis-report.md",
    "docs/analysis/drvoi-rhppo-v8/summary.json",
    "docs/DRVOI-RHPPO_V9_COACH-RHPPO算法方案.md",
    "docs/analysis/drvoi-rhppo-v11/analysis-report.md",
    "docs/analysis/drvoi-rhppo-v11/exact-summary.csv",
    "docs/DRVOI-RHPPO_V12_MRAT-RHPPO执行方案.md",
    "docs/DRVOI-RHPPO_V12_MRAT执行状态_20261107.md",
    "docs/analysis/drvoi-rhppo-v12-b1-frozen-seed85/summary.json",
    "docs/analysis/drvoi-rhppo-v12-b2-cost-regularized-seed85/summary.json",
    "docs/analysis/drvoi-rhppo-v12-b3-shield-only-diagnostic/summary.json",
    "docs/analysis/drvoi-rhppo-v12-b4-return-gated-smoke/summary.json",
    "docs/analysis/drvoi-rhppo-v12-b5-behavior-gated-smoke/summary.json",
    "runs/drvoi_rhppo_v12/return-teacher-a3-h60-20k-seed85/dataset-manifest.json",
    "runs/drvoi_rhppo_v12/return-model-a4-h60-20k-seed85/report.json",
    "runs/drvoi_rhppo_v12/return-model-a5-behavior-h60-20k-seed85/report.json",
    "runs/drvoi_rhppo_v12/mrat-b4-return-gated-smoke-seed85/latest.pt",
    "runs/drvoi_rhppo_v12/mrat-b5-behavior-gated-smoke-seed85/latest.pt",
    "runs/drvoi_rhppo_v12/return-teacher-a6-onpolicy-h60-20k-seed85/dataset-manifest.json",
    "runs/drvoi_rhppo_v12/return-teacher-a6-onpolicy-h60-20k-seed85/validation.json",
    "runs/drvoi_rhppo_v12/return-model-a7-onpolicy-behavior-h60-20k-seed85/report.json",
    "runs/drvoi_rhppo_v12/mrat-b6-onpolicy-gated-smoke-seed85/latest.pt",
    "docs/analysis/drvoi-rhppo-v12-b6-onpolicy-gated-smoke/summary.json",
    "docs/analysis/drvoi-rhppo-v12-b7-gate-latched-diagnostic/summary.json",
    "docs/analysis/drvoi-rhppo-v12-b8-sprt-latched-diagnostic/summary.json",
    "docs/analysis/drvoi-rhppo-v12-mechanism/analysis-report.md",
    "docs/analysis/drvoi-rhppo-v12-mechanism/stats-appendix.md",
    "runs/drvoi_rhppo_v9/coach-online/development-1m-seed85/latest.pt",
    "runs/drvoi_rhppo_v9/coach-online/development-1m-seed85/eval-hold.json",
    "runs/drvoi_rhppo_v9/coach-online/development-1m-seed85/eval-pursuit.json",
    "runs/drvoi_rhppo_v9/coach-online/development-1m-seed85/eval-evasive.json",
    "runs/drvoi_rhppo_v11_2/race-episode-latched/frozen-v9-seed85/eval-hold.json",
    "runs/drvoi_rhppo_v11_2/race-episode-latched/frozen-v9-seed85/eval-pursuit.json",
    "runs/drvoi_rhppo_v11_2/race-episode-latched/frozen-v9-seed85/eval-evasive.json",
    "runs/drvoi_rhppo_v11_3/race-sustained/frozen-v9-seed85/eval-hold.json",
    "runs/drvoi_rhppo_v11_3/race-sustained/frozen-v9-seed85/eval-pursuit.json",
    "runs/drvoi_rhppo_v11_3/race-sustained/frozen-v9-seed85/eval-evasive.json",
)

EXECUTABLE_PATHS = (
    "configs/drvoi_rhppo_v12_mrat.json",
    "envs/torch6dof/wvr_env.py",
    "envs/torch6dof/f16_dynamics.py",
    "envs/torch6dof/missile.py",
    "benchmarks/drvoi_v9/model.py",
    "benchmarks/drvoi_v10/reachability.py",
    "benchmarks/evaluate_torch_wvr_coach_rhppo.py",
    "benchmarks/evaluate_torch_wvr_race_rhppo_sustained.py",
    "benchmarks/drvoi_v12/snapshot.py",
    "benchmarks/drvoi_v12/reachability_teacher.py",
    "benchmarks/drvoi_v12/execution.py",
    "benchmarks/drvoi_v12/features.py",
    "benchmarks/drvoi_v12/reachability_model.py",
    "benchmarks/drvoi_v12/router.py",
    "benchmarks/drvoi_v12/losses.py",
    "benchmarks/drvoi_v12/metrics.py",
    "benchmarks/collect_drvoi_v12_reachability_data.py",
    "benchmarks/train_drvoi_v12_reachability.py",
    "benchmarks/train_torch_wvr_mrat_rhppo.py",
    "benchmarks/evaluate_torch_wvr_mrat_rhppo.py",
    "benchmarks/analyze_drvoi_v12_b0.py",
    "benchmarks/analyze_drvoi_v12_mechanism.py",
    "benchmarks/validate_drvoi_v12_reachability_data.py",
    "scripts/run_drvoi_v12_reachability_a1_a2_wsl.sh",
    "scripts/supervise_drvoi_v12_reachability_wsl.sh",
    "scripts/start_drvoi_v12_reachability_wsl.sh",
    "scripts/start_drvoi_v12_reachability_windows.ps1",
    "scripts/status_drvoi_v12_reachability_wsl.sh",
    "scripts/run_drvoi_v12_b1_frozen_seed85_wsl.sh",
    "scripts/supervise_drvoi_v12_b1_frozen_seed85_wsl.sh",
    "scripts/start_drvoi_v12_b1_frozen_seed85_windows.ps1",
    "scripts/run_drvoi_v12_b2_cost_regularized_seed85_wsl.sh",
    "scripts/supervise_drvoi_v12_b2_cost_regularized_seed85_wsl.sh",
    "scripts/start_drvoi_v12_b2_cost_regularized_seed85_windows.ps1",
    "scripts/run_drvoi_v12_b3_shield_only_diagnostic_wsl.sh",
    "scripts/supervise_drvoi_v12_b3_shield_only_diagnostic_wsl.sh",
    "scripts/start_drvoi_v12_b3_shield_only_diagnostic_windows.ps1",
    "scripts/run_drvoi_v12_return_teacher_a3_a4_wsl.sh",
    "scripts/supervise_drvoi_v12_return_teacher_a3_a4_wsl.sh",
    "scripts/start_drvoi_v12_return_teacher_a3_a4_windows.ps1",
    "scripts/run_drvoi_v12_b4_return_gated_smoke_wsl.sh",
    "scripts/supervise_drvoi_v12_b4_return_gated_smoke_wsl.sh",
    "scripts/start_drvoi_v12_b4_return_gated_smoke_windows.ps1",
    "scripts/run_drvoi_v12_b5_behavior_gated_smoke_wsl.sh",
    "scripts/supervise_drvoi_v12_b5_behavior_gated_smoke_wsl.sh",
    "scripts/start_drvoi_v12_b5_behavior_gated_smoke_windows.ps1",
    "scripts/run_drvoi_v12_onpolicy_return_a6_a7_wsl.sh",
    "scripts/supervise_drvoi_v12_onpolicy_return_a6_a7_wsl.sh",
    "scripts/start_drvoi_v12_onpolicy_return_a6_a7_windows.ps1",
    "scripts/run_drvoi_v12_b6_onpolicy_gated_smoke_wsl.sh",
    "scripts/supervise_drvoi_v12_b6_onpolicy_gated_smoke_wsl.sh",
    "scripts/start_drvoi_v12_b6_onpolicy_gated_smoke_windows.ps1",
    "scripts/run_drvoi_v12_b7_gate_latched_diagnostic_wsl.sh",
    "scripts/supervise_drvoi_v12_b7_gate_latched_diagnostic_wsl.sh",
    "scripts/start_drvoi_v12_b7_gate_latched_diagnostic_windows.ps1",
    "scripts/run_drvoi_v12_b8_sprt_latched_diagnostic_wsl.sh",
    "scripts/supervise_drvoi_v12_b8_sprt_latched_diagnostic_wsl.sh",
    "scripts/start_drvoi_v12_b8_sprt_latched_diagnostic_windows.ps1",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("development", "formal"),
        default="development")
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/drvoi_rhppo_v12/protocol/development-manifest.json"))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    completed = subprocess.run(
        ("git", *args), cwd=ROOT, check=True, capture_output=True,
        text=True, encoding="utf-8")
    return completed.stdout.strip()


def file_record(relative: str) -> dict[str, object]:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": relative.replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    args = parse_args()
    status = git("status", "--porcelain").splitlines()
    if args.mode == "formal" and status:
        raise RuntimeError(
            "formal manifest requires a clean worktree; use development mode "
            "while implementation is in progress")
    config_path = ROOT / "configs/drvoi_rhppo_v12_mrat.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    payload = {
        "manifest_version": "MRAT-RHPPO-V12-MANIFEST-1",
        "mode": args.mode,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_result": False,
        "protocol_version": config["protocol_version"],
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "git_status": status,
        "evidence": [file_record(path) for path in EVIDENCE_PATHS],
        "executable": [file_record(path) for path in EXECUTABLE_PATHS],
        "evaluation_seed_ledger": {
            "development_reused": [10085],
            "confirmation_reserved_unread": config["evaluation"][
                "confirmation_seeds"],
        },
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({
        "output": str(output),
        "mode": args.mode,
        "git_dirty": bool(status),
        "evidence_files": len(payload["evidence"]),
        "executable_files": len(payload["executable"]),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
