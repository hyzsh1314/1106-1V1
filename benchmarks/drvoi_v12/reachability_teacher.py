"""Physics-grounded reachability targets for MRAT-RHPPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, MutableMapping

import torch

from envs.torch6dof import TorchWVRVecEnv


PolicyStep = Callable[
    [TorchWVRVecEnv, MutableMapping[str, torch.Tensor], torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, MutableMapping[str, torch.Tensor]],
]
PolicyPostStep = Callable[
    [
        TorchWVRVecEnv,
        MutableMapping[str, torch.Tensor],
        torch.Tensor,
        dict[str, torch.Tensor],
    ],
    MutableMapping[str, torch.Tensor],
]


def instantaneous_reachability_components(
        observation: torch.Tensor,
        max_attack_angle_deg: float = 45.0,
) -> dict[str, torch.Tensor]:
    """Reconstruct observable launch-envelope margins from V4 observations."""
    if observation.ndim != 2 or observation.shape[-1] != 39:
        raise ValueError("observation must have [batch,39] shape")
    distance = 10000.0 * observation[:, 13]
    closure = 340.0 * observation[:, 28]
    target_aspect = observation[:, 12]
    angle_off = observation[:, 11]
    los_rate = torch.linalg.vector_norm(observation[:, 26:28], dim=-1)
    dynamic_range = (
        7500.0
        + 8.0 * closure.clamp(-150.0, 550.0)
        - 2200.0 * (1.0 - torch.cos(target_aspect)) * 0.5
        - 5000.0 * (los_rate / 0.20).clamp(0.0, 1.0)
    ).clamp(2500.0, 14000.0)
    max_angle = torch.deg2rad(torch.as_tensor(
        max_attack_angle_deg, device=observation.device,
        dtype=observation.dtype))
    return {
        "distance_m": distance,
        "closure_m_s": closure,
        "los_rate_rad_s": los_rate,
        "dynamic_range_m": dynamic_range,
        "range_margin": (dynamic_range - distance) / 14000.0,
        "angle_margin": (max_angle - angle_off) / max_angle,
        "closure_margin": torch.tanh(closure / 250.0),
        "los_stability_margin": 1.0 - (los_rate / 0.20).clamp(0.0, 1.0),
    }


def instantaneous_reachability_quality(
        observation: torch.Tensor,
        kappa: float = 10.0,
) -> torch.Tensor:
    """Normalized smooth minimum of the four observable margins.

    The normalization makes a vector of all-zero margins map to zero.  This is
    a diagnostic reachability score, not a kill probability or safety proof.
    """
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    components = instantaneous_reachability_components(observation)
    margins = torch.stack((
        components["range_margin"],
        components["angle_margin"],
        components["closure_margin"],
        components["los_stability_margin"],
    ), dim=-1)
    count = torch.as_tensor(
        margins.shape[-1], device=margins.device, dtype=margins.dtype)
    return (
        -torch.logsumexp(-float(kappa) * margins, dim=-1)
        + torch.log(count)
    ) / float(kappa)


@dataclass(frozen=True)
class BranchRolloutResult:
    score: torch.Tensor
    legal_within_horizon: torch.Tensor
    first_legal_step: torch.Tensor
    crashed: torch.Tensor
    terminal_step: torch.Tensor
    legal_opportunity_steps: torch.Tensor
    fire_requests: torch.Tensor
    accepted_launches: torch.Tensor
    rejected_fire_requests: torch.Tensor
    hits: torch.Tensor
    discounted_return: torch.Tensor
    terminal_outcome: torch.Tensor
    timeout: torch.Tensor
    model_decisions: int


def aggregate_hypothesis_targets(
        score: torch.Tensor,
        legal: torch.Tensor,
        first_legal_step: torch.Tensor,
        temperature: float = 0.25,
        weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Conservatively aggregate the final hypothesis dimension.

    Score uses a normalized weighted soft minimum. Legal opportunity is a
    weighted soft probability, while time is conditioned only on hypotheses
    that actually reach a legal opportunity; fully unsuccessful rows remain
    explicitly right-censored through ``event_observed``.
    """
    if score.shape != legal.shape or score.shape != first_legal_step.shape:
        raise ValueError("hypothesis target tensors must share one shape")
    if score.ndim < 1 or score.shape[-1] < 1:
        raise ValueError("targets require a non-empty hypothesis dimension")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    hypotheses = score.shape[-1]
    if weights is None:
        weights = torch.full(
            (hypotheses,), 1.0 / hypotheses,
            device=score.device, dtype=score.dtype)
    else:
        if weights.shape != (hypotheses,):
            raise ValueError("weights must match the hypothesis dimension")
        weights = weights.to(device=score.device, dtype=score.dtype)
        if bool((weights < 0.0).any()) or float(weights.sum()) <= 0.0:
            raise ValueError("weights must be non-negative with positive sum")
        weights = weights / weights.sum()
    log_weights = weights.clamp_min(torch.finfo(score.dtype).tiny).log()
    robust_score = -float(temperature) * torch.logsumexp(
        log_weights - score / float(temperature), dim=-1)
    legal_float = legal.to(score.dtype)
    probability = (legal_float * weights).sum(-1)
    time_numerator = (
        first_legal_step.to(score.dtype) * legal_float * weights).sum(-1)
    conditional_time = time_numerator / probability.clamp_min(1.0e-8)
    event_observed = legal.any(-1)
    conditional_time = torch.where(
        event_observed, conditional_time, torch.zeros_like(conditional_time))
    return {
        "score": robust_score,
        "probability": probability,
        "time": conditional_time,
        "event_observed": event_observed,
    }


@torch.no_grad()
def rollout_reachability_branch(
        env: TorchWVRVecEnv,
        learner_side: torch.Tensor,
        execution_state: MutableMapping[str, torch.Tensor],
        policy_step: PolicyStep,
        policy_post_step: PolicyPostStep | None = None,
        horizon: int = 30,
        gamma: float = 0.98,
        legal_bonus: float = 1.0,
        crash_penalty: float = 1.0,
        kappa: float = 10.0,
) -> tuple[BranchRolloutResult, MutableMapping[str, torch.Tensor]]:
    """Roll one branch without using states after terminal auto-reset.

    ``policy_step`` returns complete aircraft controls, fire requests and the
    updated caller-owned execution state.  ``policy_post_step`` runs after the
    environment transition so recurrent states can apply terminal masks using
    the actual ``done`` tensor. Rows become inactive at their first terminal
    transition even though ``TorchWVRVecEnv.step`` internally resets them.
    Post-reset observations can therefore never improve the target.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must lie in (0,1]")
    if learner_side.shape != (env.config.num_envs,):
        raise ValueError("learner_side must have one index per environment")
    learner_side = learner_side.to(device=env.device, dtype=torch.long)
    if bool(((learner_side < 0) | (learner_side > 1)).any()):
        raise ValueError("learner_side entries must be 0 or 1")
    env_index = torch.arange(env.config.num_envs, device=env.device)
    active = torch.ones(env.config.num_envs, dtype=torch.bool, device=env.device)
    score = torch.full(
        (env.config.num_envs,), -torch.inf, device=env.device)
    first_legal = torch.full(
        (env.config.num_envs,), horizon + 1, dtype=torch.long,
        device=env.device)
    crashed = torch.zeros_like(active)
    terminal_step = torch.full_like(first_legal, horizon + 1)
    legal_steps = torch.zeros_like(first_legal)
    fire_requests = torch.zeros_like(first_legal)
    accepted_launches = torch.zeros_like(first_legal)
    rejected_requests = torch.zeros_like(first_legal)
    hits = torch.zeros_like(first_legal)
    discounted_return = torch.zeros(
        env.config.num_envs, device=env.device)
    terminal_outcome = torch.zeros_like(discounted_return)
    timed_out = torch.zeros_like(active)
    state = execution_state
    model_decisions = 0

    for step in range(horizon):
        observation = env.observations()
        learner_observation = observation[env_index, learner_side]
        quality = instantaneous_reachability_quality(
            learner_observation, kappa=kappa)
        legal = env.legal_fire_mask()[env_index, learner_side]
        first_legal = torch.where(
            active & legal & (first_legal > horizon),
            torch.full_like(first_legal, step), first_legal)
        candidate = (
            (float(gamma) ** step) * quality
            + float(legal_bonus) * legal.float()
        )
        score = torch.where(active, torch.maximum(score, candidate), score)

        controls, fire_request, state = policy_step(env, state, active)
        if controls.shape != (env.config.num_envs, 2, 4):
            raise ValueError("policy_step controls have the wrong shape")
        if fire_request.shape != (env.config.num_envs, 2):
            raise ValueError("policy_step fire_request has the wrong shape")
        _, reward, done, info = env.step(controls, fire_request)
        if policy_post_step is not None:
            state = policy_post_step(env, state, done, info)
        learner_crash = info["crashes"][env_index, learner_side]
        legal_steps += (
            active & info["legal_fire_opportunities"][
                env_index, learner_side]).long()
        fire_requests += (
            active & info["fire_requests"][env_index, learner_side]).long()
        accepted_launches += (
            active & info["accepted_launches"][env_index, learner_side]).long()
        rejected_requests += (
            active & info["rejected_fire_requests"][
                env_index, learner_side]).long()
        hits += (active & info["hits"][env_index, learner_side]).long()
        learner_reward = reward[env_index, learner_side]
        discounted_return += (
            active.to(learner_reward.dtype)
            * (float(gamma) ** step) * learner_reward)
        learner_outcome = torch.where(
            learner_side == 0, info["outcome_red"], -info["outcome_red"])
        terminal_outcome = torch.where(
            active & done, learner_outcome, terminal_outcome)
        timed_out = timed_out | (active & info["timeout"])
        crashed = crashed | (active & learner_crash)
        score = torch.where(
            active & learner_crash, score - float(crash_penalty), score)
        first_terminal = active & done
        terminal_step = torch.where(
            first_terminal, torch.full_like(terminal_step, step + 1),
            terminal_step)
        model_decisions += int(active.sum())
        active = active & ~done
        if not bool(active.any()):
            break

    score = torch.where(torch.isfinite(score), score, torch.zeros_like(score))
    return BranchRolloutResult(
        score=score,
        legal_within_horizon=first_legal <= horizon,
        first_legal_step=first_legal,
        crashed=crashed,
        terminal_step=terminal_step,
        legal_opportunity_steps=legal_steps,
        fire_requests=fire_requests,
        accepted_launches=accepted_launches,
        rejected_fire_requests=rejected_requests,
        hits=hits,
        discounted_return=discounted_return,
        terminal_outcome=terminal_outcome,
        timeout=timed_out,
        model_decisions=model_decisions,
    ), state
