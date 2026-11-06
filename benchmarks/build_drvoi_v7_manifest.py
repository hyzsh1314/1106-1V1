#!/usr/bin/env python3
"""Freeze source/configuration hashes before DRVOI-RHPPO V7 execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import torch

from envs.JSBSim.utils.utils import get_root_dir
from envs.torch6dof import TorchWVRConfig


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    config_path = root / "configs" / "drvoi_rhppo_v7_wvr.json"
    protocol_path = root / "docs" / "DRVOI_RHPPO_V7_WVR响应重建协议.md"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    environment = TorchWVRConfig(num_envs=1)
    lowlevel = Path(get_root_dir()) / "model" / "baseline_model.pt"
    sources = {
        "protocol_config": config_path,
        "execution_protocol": protocol_path,
        "environment": root / "envs" / "torch6dof" / "wvr_env.py",
        "dynamics": root / "envs" / "torch6dof" / "f16_dynamics.py",
        "missile": root / "envs" / "torch6dof" / "missile.py",
        "b0_trainer": root / "benchmarks" / "train_torch_wvr_b0_rhppo_v7.py",
        "response_collector": root / "benchmarks" / "collect_drvoi_v7_response_data.py",
        "response_validator": root / "benchmarks" / "validate_drvoi_wvr_response_v7.py",
        "q_validator": root / "benchmarks" / "validate_drvoi_wvr_q_v7.py",
        "o0_validator": root / "benchmarks" / "validate_drvoi_wvr_o0_v7.py",
        "gate_summary": root / "benchmarks" / "summarize_drvoi_v7_gates.py",
        "response_targets": root / "benchmarks" / "drvoi" / "targets.py",
        "response_model": root / "benchmarks" / "drvoi" / "response.py",
        "calibration": root / "benchmarks" / "drvoi" / "calibration.py",
        "q_assembler": root / "benchmarks" / "drvoi" / "core.py",
        "lowlevel_controller": lowlevel,
    }
    manifest = {
        "protocol_version": protocol["protocol_version"],
        "scientific_result": False,
        "created_before_v7_execution": True,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "environment_contract": {
            "observation_version": environment.observation_version,
            "observation_dim": 39,
            "sim_freq_hz": environment.sim_freq_hz,
            "physics_substeps_per_decision": environment.substeps_per_decision,
            "reward_mode": environment.reward_mode,
            "action_dims": [3, 5, 3, 2],
            "joint_actions": 90,
        },
        "hashes": {name: digest(path) for name, path in sources.items()},
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
