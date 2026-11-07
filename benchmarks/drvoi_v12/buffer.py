"""Flat option-segment storage and SMDP advantage calculation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FlatOptionBatch:
    features: torch.Tensor
    expert: torch.Tensor
    old_log_prob: torch.Tensor
    value: torch.Tensor
    segment_return: torch.Tensor
    next_value: torch.Tensor
    duration: torch.Tensor
    done: torch.Tensor
    policy_mask: torch.Tensor
    env_id: torch.Tensor

    def validate(self) -> None:
        count = self.expert.shape[0]
        if self.features.ndim != 2 or self.features.shape[0] != count:
            raise ValueError("segment features have the wrong shape")
        for name in (
                "old_log_prob", "value", "segment_return", "next_value",
                "duration", "done", "policy_mask", "env_id"):
            if getattr(self, name).shape != (count,):
                raise ValueError(f"segment field {name!r} has the wrong shape")
        if bool((self.duration <= 0).any()):
            raise ValueError("segment duration must be positive")


def compute_flat_smdp_gae(
        batch: FlatOptionBatch,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute reverse per-environment GAE for interleaved option segments."""
    batch.validate()
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must lie in [0,1]")
    dtype = batch.segment_return.dtype
    duration = batch.duration.to(dtype)
    not_done = (~batch.done.bool()).to(dtype)
    gamma_duration = torch.pow(torch.as_tensor(
        gamma, device=duration.device, dtype=dtype), duration)
    trace_duration = torch.pow(torch.as_tensor(
        gamma * gae_lambda, device=duration.device, dtype=dtype), duration)
    delta = (
        batch.segment_return + gamma_duration * not_done * batch.next_value
        - batch.value)
    advantage = torch.zeros_like(delta)
    max_env = int(batch.env_id.max()) + 1 if len(batch.env_id) else 0
    running = torch.zeros(max_env, device=delta.device, dtype=dtype)
    for index in range(len(delta) - 1, -1, -1):
        env = int(batch.env_id[index])
        running[env] = delta[index] + (
            trace_duration[index] * not_done[index] * running[env])
        advantage[index] = running[env]
    return advantage, advantage + batch.value
