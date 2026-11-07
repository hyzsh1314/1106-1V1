"""Semi-Markov returns and policy losses for MRAT-RHPPO."""

from __future__ import annotations

import torch


def regularized_step_reward(
        extrinsic_reward: torch.Tensor,
        expert: torch.Tensor,
        switched: torch.Tensor,
        intervention_weight: float = 0.0,
        switch_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the preregistered LOS-intervention and switch regularizers."""
    if (expert.shape != extrinsic_reward.shape
            or switched.shape != extrinsic_reward.shape):
        raise ValueError("reward regularizer inputs must have identical shapes")
    if intervention_weight < 0.0 or switch_weight < 0.0:
        raise ValueError("reward regularizer weights must be non-negative")
    intervention_cost = (
        expert.eq(1).to(extrinsic_reward.dtype) * float(intervention_weight))
    transition_cost = (
        switched.bool().to(extrinsic_reward.dtype) * float(switch_weight))
    return (
        extrinsic_reward - intervention_cost - transition_cost,
        intervention_cost,
        transition_cost,
    )


def variable_duration_gae(
        segment_return: torch.Tensor,
        duration: torch.Tensor,
        done: torch.Tensor,
        values: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE over option segments with duration-dependent discounting.

    ``segment_return`` already contains within-segment discounted rewards and
    has shape ``[K,B]``. ``values`` has shape ``[K+1,B]``.
    """
    if segment_return.ndim != 2:
        raise ValueError("segment_return must have [K,B] shape")
    if duration.shape != segment_return.shape or done.shape != segment_return.shape:
        raise ValueError("duration and done must match segment_return")
    if values.shape != (segment_return.shape[0] + 1, segment_return.shape[1]):
        raise ValueError("values must have [K+1,B] shape")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must lie in [0,1]")
    if bool((duration <= 0).any()):
        raise ValueError("segment durations must be positive")
    duration_float = duration.to(segment_return.dtype)
    gamma_duration = torch.pow(
        torch.as_tensor(gamma, device=segment_return.device,
                        dtype=segment_return.dtype), duration_float)
    trace_duration = torch.pow(
        torch.as_tensor(gamma * gae_lambda, device=segment_return.device,
                        dtype=segment_return.dtype), duration_float)
    not_done = (~done.bool()).to(segment_return.dtype)
    delta = (
        segment_return + gamma_duration * not_done * values[1:]
        - values[:-1]
    )
    advantage = torch.zeros_like(segment_return)
    running = torch.zeros_like(segment_return[0])
    for index in range(segment_return.shape[0] - 1, -1, -1):
        running = delta[index] + (
            trace_duration[index] * not_done[index] * running)
        advantage[index] = running
    return advantage, advantage + values[:-1]


def clipped_policy_loss(
        new_log_prob: torch.Tensor,
        old_log_prob: torch.Tensor,
        advantage: torch.Tensor,
        clip_ratio: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not (new_log_prob.shape == old_log_prob.shape == advantage.shape):
        raise ValueError("policy loss inputs must have identical shapes")
    if clip_ratio < 0.0:
        raise ValueError("clip_ratio must be non-negative")
    ratio = torch.exp(new_log_prob - old_log_prob)
    surrogate = torch.minimum(
        ratio * advantage,
        ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantage)
    approx_kl = ((ratio - 1.0) - (new_log_prob - old_log_prob)).mean()
    clip_fraction = (
        (ratio - 1.0).abs() > clip_ratio).float().mean()
    return -surrogate.mean(), approx_kl, clip_fraction


def option_termination_loss(
        termination_probability: torch.Tensor,
        active_q: torch.Tensor,
        state_value: torch.Tensor,
        eligible: torch.Tensor,
        deliberation_cost: float = 0.0,
) -> torch.Tensor:
    """Option-Critic termination objective in minimization form."""
    shape = termination_probability.shape
    if (active_q.shape != shape or state_value.shape != shape
            or eligible.shape != shape):
        raise ValueError("termination loss inputs must have identical shapes")
    mask = eligible.bool()
    if not bool(mask.any()):
        return termination_probability.sum() * 0.0
    continuation_advantage = active_q - state_value
    return (
        termination_probability[mask]
        * (continuation_advantage[mask] + float(deliberation_cost)).detach()
    ).mean()
