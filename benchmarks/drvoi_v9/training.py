"""Training helpers for high-level option segments."""

from __future__ import annotations

import torch


def option_segment_targets(
        rewards: torch.Tensor, done: torch.Tensor,
        state_bootstrap: torch.Tensor, horizon: int,
        gamma: float) -> torch.Tensor:
    """K-step targets with terminal-safe bootstrap.

    ``state_bootstrap`` has shape ``[T+1,B]`` and contains the target option
    value for states inside the rollout plus the final rollout bootstrap.
    """
    if rewards.ndim != 2 or done.shape != rewards.shape:
        raise ValueError("rewards and done must share [T,B] shape")
    if state_bootstrap.shape != (
            rewards.shape[0] + 1, rewards.shape[1]):
        raise ValueError("state_bootstrap must have [T+1,B] shape")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0,1]")
    steps, _ = rewards.shape
    targets = torch.zeros_like(rewards)
    for start in range(steps):
        value = torch.zeros_like(rewards[start])
        alive = torch.ones_like(done[start], dtype=torch.bool)
        discount = 1.0
        end = min(start + horizon, steps)
        for index in range(start, end):
            value = value + discount * rewards[index] * alive.float()
            alive = alive & (~done[index].bool())
            discount *= gamma
        value = value + discount * state_bootstrap[end] * alive.float()
        targets[start] = value
    return targets


@torch.no_grad()
def polyak_update(source: torch.nn.Module, target: torch.nn.Module,
                  tau: float) -> None:
    if not 0.0 < tau <= 1.0:
        raise ValueError("tau must lie in (0,1]")
    source_parameters = dict(source.named_parameters())
    target_parameters = dict(target.named_parameters())
    if source_parameters.keys() != target_parameters.keys():
        raise ValueError("source and target modules do not match")
    for name, target_parameter in target_parameters.items():
        target_parameter.lerp_(source_parameters[name], tau)
    source_buffers = dict(source.named_buffers())
    target_buffers = dict(target.named_buffers())
    if source_buffers.keys() != target_buffers.keys():
        raise ValueError("source and target buffers do not match")
    for name, target_buffer in target_buffers.items():
        if torch.is_floating_point(target_buffer):
            target_buffer.lerp_(source_buffers[name], tau)
        else:
            target_buffer.copy_(source_buffers[name])


def linear_anneal(initial: float, final: float, update: int,
                  updates: int) -> float:
    if updates <= 1:
        return float(final)
    fraction = min(max((update - 1) / (updates - 1), 0.0), 1.0)
    return float(initial + fraction * (final - initial))


def progress_violation_events(stagnation_steps: torch.Tensor,
                              window: int) -> torch.Tensor:
    """Emit one cost at each completed no-progress window."""
    if window <= 0:
        raise ValueError("window must be positive")
    return (
        (stagnation_steps >= window)
        & (stagnation_steps.remainder(window) == 0)
    ).float()

