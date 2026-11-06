"""Complete TorchWVR snapshot/restore used only for paired diagnostics."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch


ENV_TENSORS = (
    "aircraft_state", "missile_state", "alive", "remaining_missiles",
    "last_launch_step", "episode_step", "episode_id", "previous_potential",
)


@dataclass
class WVRDiagnosticSnapshot:
    environment: dict[str, torch.Tensor]
    environment_rng: torch.Tensor
    lowlevel_hidden: torch.Tensor
    extra: dict[str, object]


def capture_wvr_snapshot(env, lowlevel_hidden: torch.Tensor,
                         **extra) -> WVRDiagnosticSnapshot:
    missing = [name for name in ENV_TENSORS if not hasattr(env, name)]
    if missing:
        raise AttributeError(f"environment snapshot missing fields: {missing}")
    return WVRDiagnosticSnapshot(
        environment={
            name: getattr(env, name).detach().clone() for name in ENV_TENSORS},
        environment_rng=env.generator.get_state().clone(),
        lowlevel_hidden=lowlevel_hidden.detach().clone(),
        extra=copy.deepcopy(extra),
    )


def restore_wvr_snapshot(env, snapshot: WVRDiagnosticSnapshot):
    for name, value in snapshot.environment.items():
        target = getattr(env, name)
        if target.shape != value.shape or target.dtype != value.dtype:
            raise ValueError(f"snapshot field {name} does not match environment")
        target.copy_(value.to(target.device))
    # Generator state is represented by a CPU ByteTensor even when the
    # generator itself is CUDA-backed.  Moving it to the environment device
    # makes CUDA restore version-dependent and has failed on older PyTorch.
    env.generator.set_state(snapshot.environment_rng.cpu())
    return snapshot.lowlevel_hidden.to(env.device).clone(), copy.deepcopy(
        snapshot.extra)
