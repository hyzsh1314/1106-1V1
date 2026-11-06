#!/usr/bin/env python3
"""Freeze the platform contract and hashes before DRVOI experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import torch

from envs.JSBSim.utils.utils import get_root_dir
from envs.torch6dof import TorchWVRConfig, TorchWVRVecEnv


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    config_path = root / "configs" / "drvoi_rhppo_v6_wvr.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    lowlevel = Path(get_root_dir()) / "model" / "baseline_model.pt"
    env_config = TorchWVRConfig(
        num_envs=1, reward_mode="competitive", observation_version="v4")
    manifest = {
        "protocol": protocol["protocol_version"],
        "scientific_result": False,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "environment_contract": {
            "observation_dim": TorchWVRVecEnv.observation_size,
            "observation_version": env_config.observation_version,
            "sim_freq_hz": env_config.sim_freq_hz,
            "physics_substeps_per_decision": env_config.substeps_per_decision,
            "reward_mode": env_config.reward_mode,
            "action_dims": [3, 5, 3, 2],
            "joint_actions": 90,
        },
        "hashes": {
            "protocol_config": digest(config_path),
            "environment": digest(root / "envs" / "torch6dof" / "wvr_env.py"),
            "dynamics": digest(root / "envs" / "torch6dof" / "f16_dynamics.py"),
            "missile": digest(root / "envs" / "torch6dof" / "missile.py"),
            "lowlevel_controller": digest(lowlevel),
        },
        "git_commit": None,
        "note": "Workspace is not a Git checkout; SHA-256 source hashes are authoritative.",
    }
    try:
        manifest["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
