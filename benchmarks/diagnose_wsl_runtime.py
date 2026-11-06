"""Diagnose WSL, CUDA, PyTorch and the WVR environment in one command."""

from __future__ import annotations

import argparse
import os
import platform
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--environment-smoke", action="store_true")
    return parser.parse_args()


def is_wsl():
    values = " ".join((platform.release(), platform.version())).lower()
    return "microsoft" in values or "wsl" in values


def environment_smoke():
    from envs.JSBSim.envs.singlecombat_env import SingleCombatEnv

    env = SingleCombatEnv(
        "1v1/ShootMissile/HierarchySelfplay", reuse_fdm=True)
    try:
        env.seed(85)
        obs = env.reset()
        actions = np.asarray([[1, 2, 1, 0], [1, 2, 1, 0]], dtype=np.int64)
        next_obs, reward, done, _ = env.step(actions)
        if not np.isfinite(next_obs).all() or not np.isfinite(reward).all():
            raise RuntimeError("environment produced a non-finite value")
        print(
            "environment=PASS obs_shape={0} reward_shape={1} done_shape={2}".format(
                tuple(obs.shape), tuple(reward.shape), tuple(done.shape)))
    finally:
        env.close()


def main():
    args = parse_args()
    cuda = bool(torch.cuda.is_available())
    print("platform={0}".format(platform.platform()))
    print("wsl={0}".format(is_wsl()))
    print("python={0}".format(platform.python_version()))
    print("cpu_count={0}".format(os.cpu_count()))
    print("project={0}".format(Path.cwd().resolve()))
    print("torch={0}".format(torch.__version__))
    print("torch_cuda_runtime={0}".format(torch.version.cuda))
    print("cuda_available={0}".format(cuda))
    if args.require_cuda and not cuda:
        raise RuntimeError("CUDA is required but unavailable to PyTorch")
    if cuda:
        properties = torch.cuda.get_device_properties(0)
        print("gpu={0}".format(properties.name))
        print("gpu_memory_gib={0:.2f}".format(
            properties.total_memory / (1024 ** 3)))
        print("compute_capability={0}.{1}".format(
            properties.major, properties.minor))
        left = torch.randn((2048, 2048), device="cuda")
        right = torch.randn((2048, 2048), device="cuda")
        result = left @ right
        torch.cuda.synchronize()
        if not torch.isfinite(result).all():
            raise RuntimeError("CUDA matrix multiplication was non-finite")
        print("cuda_compute=PASS")
    if args.environment_smoke:
        environment_smoke()


if __name__ == "__main__":
    main()
