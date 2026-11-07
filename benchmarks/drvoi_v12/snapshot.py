"""Exact tensor snapshots for paired Torch WVR counterfactual branches."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Mapping

import torch

from envs.torch6dof import TorchWVRVecEnv


_ENV_TENSOR_NAMES = (
    "aircraft_state",
    "missile_state",
    "alive",
    "remaining_missiles",
    "last_launch_step",
    "episode_step",
    "episode_id",
    "previous_potential",
)


def _clone_tensors(values: Mapping[str, torch.Tensor] | None) -> dict[str, torch.Tensor]:
    if values is None:
        return {}
    cloned: dict[str, torch.Tensor] = {}
    for name, value in values.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise TypeError("execution tensors must be a str->Tensor mapping")
        cloned[name] = value.detach().clone()
    return cloned


@dataclass(frozen=True)
class TorchWVRExecutionSnapshot:
    """Environment and caller-owned recurrent execution state.

    The environment snapshot is sufficient for exact physics replay before a
    terminal reset.  ``execution_tensors`` carries policy/controller state such
    as GRU hidden tensors, previous actions, active options and option ages.
    """

    env_tensors: dict[str, torch.Tensor]
    generator_state: torch.Tensor
    torch_rng_state: torch.Tensor
    cuda_rng_states: tuple[torch.Tensor, ...]
    execution_tensors: dict[str, torch.Tensor]
    num_envs: int

    def clone(self) -> "TorchWVRExecutionSnapshot":
        return TorchWVRExecutionSnapshot(
            env_tensors=_clone_tensors(self.env_tensors),
            generator_state=self.generator_state.clone(),
            torch_rng_state=self.torch_rng_state.clone(),
            cuda_rng_states=tuple(
                state.clone() for state in self.cuda_rng_states),
            execution_tensors=_clone_tensors(self.execution_tensors),
            num_envs=self.num_envs,
        )


def capture_execution_snapshot(
        env: TorchWVRVecEnv,
        execution_tensors: Mapping[str, torch.Tensor] | None = None,
) -> TorchWVRExecutionSnapshot:
    """Capture all mutable tensors needed by a paired branch rollout."""
    env_tensors = {
        name: getattr(env, name).detach().clone()
        for name in _ENV_TENSOR_NAMES
    }
    return TorchWVRExecutionSnapshot(
        env_tensors=env_tensors,
        generator_state=env.generator.get_state().clone(),
        torch_rng_state=torch.random.get_rng_state().clone(),
        cuda_rng_states=tuple(
            state.clone() for state in torch.cuda.get_rng_state_all())
        if torch.cuda.is_available() else (),
        execution_tensors=_clone_tensors(execution_tensors),
        num_envs=env.config.num_envs,
    )


def restore_execution_snapshot(
        env: TorchWVRVecEnv,
        snapshot: TorchWVRExecutionSnapshot,
) -> dict[str, torch.Tensor]:
    """Restore an exact snapshot and return cloned caller-owned state."""
    if snapshot.num_envs != env.config.num_envs:
        raise ValueError(
            "snapshot num_envs does not match the destination environment")
    for name in _ENV_TENSOR_NAMES:
        source = snapshot.env_tensors.get(name)
        if source is None:
            raise KeyError(f"snapshot is missing environment tensor {name!r}")
        current = getattr(env, name)
        if source.shape != current.shape or source.dtype != current.dtype:
            raise ValueError(f"snapshot tensor {name!r} is incompatible")
        setattr(env, name, source.to(device=env.device).clone())
    # RNG restoration APIs require CPU uint8 state tensors even when the
    # generator itself is a CUDA generator. ``map_location='cuda'`` during a
    # resume load otherwise moves these tensors and makes set_state fail.
    env.generator.set_state(snapshot.generator_state.detach().to(
        device="cpu", dtype=torch.uint8))
    torch.random.set_rng_state(snapshot.torch_rng_state.detach().to(
        device="cpu", dtype=torch.uint8))
    if snapshot.cuda_rng_states:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "snapshot contains CUDA RNG state but CUDA is unavailable")
        if len(snapshot.cuda_rng_states) != torch.cuda.device_count():
            raise ValueError("snapshot CUDA RNG device count is incompatible")
        torch.cuda.set_rng_state_all([
            state.detach().to(device="cpu", dtype=torch.uint8)
            for state in snapshot.cuda_rng_states])
    return {
        name: value.to(device=env.device).clone()
        for name, value in snapshot.execution_tensors.items()
    }


def repeat_execution_snapshot(
        snapshot: TorchWVRExecutionSnapshot,
        repeats: int,
) -> TorchWVRExecutionSnapshot:
    """Repeat a branch-major snapshot for one batched counterfactual call.

    Ordinary environment and policy tensors are repeated along their leading
    environment dimension. The low-level controller tensor is first reshaped
    from ``[2*N,1,128]`` to ``[N,2,1,128]`` so aircraft pairs stay adjacent.
    """
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    num_envs = snapshot.num_envs

    def repeat_leading(name: str, value: torch.Tensor) -> torch.Tensor:
        if name in {"lowlevel_hidden", "pending_lowlevel_hidden"}:
            if value.shape[0] != 2 * num_envs:
                raise ValueError(f"{name!r} has incompatible leading size")
            paired = value.reshape(num_envs, 2, *value.shape[1:])
            return paired.repeat(repeats, *([1] * (paired.ndim - 1))).reshape(
                repeats * 2 * num_envs, *value.shape[1:])
        if value.ndim == 0:
            return value.clone()
        if value.shape[0] != num_envs:
            raise ValueError(
                f"execution tensor {name!r} has unsupported leading size")
        return value.repeat(repeats, *([1] * (value.ndim - 1)))

    env_tensors = {}
    for name, value in snapshot.env_tensors.items():
        if value.shape[0] != num_envs:
            raise ValueError(f"environment tensor {name!r} has wrong leading size")
        env_tensors[name] = value.repeat(
            repeats, *([1] * (value.ndim - 1)))
    execution_tensors = {
        name: repeat_leading(name, value)
        for name, value in snapshot.execution_tensors.items()
    }
    return TorchWVRExecutionSnapshot(
        env_tensors=env_tensors,
        generator_state=snapshot.generator_state.clone(),
        torch_rng_state=snapshot.torch_rng_state.clone(),
        cuda_rng_states=tuple(
            state.clone() for state in snapshot.cuda_rng_states),
        execution_tensors=execution_tensors,
        num_envs=num_envs * repeats,
    )


@contextmanager
def snapshot_scope(
        env: TorchWVRVecEnv,
        execution_tensors: Mapping[str, torch.Tensor] | None = None,
) -> Iterator[TorchWVRExecutionSnapshot]:
    """Restore the entry state after temporary branch mutations."""
    entry = capture_execution_snapshot(env, execution_tensors)
    try:
        yield entry
    finally:
        restore_execution_snapshot(env, entry)
