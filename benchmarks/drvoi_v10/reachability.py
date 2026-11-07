"""Reachability guard and behaviour anchors for BARC-RHPPO."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from benchmarks.drvoi_v7_1.macro_options import (
    NUM_MACRO_OPTIONS,
    macro_option_actions,
    macro_option_legal_mask,
)


LEAD_INTERCEPT_OPTION = 1


def nonclosing_intercept_guard(
        observation: torch.Tensor, min_range_normalized: float = 0.60,
        max_closure_normalized: float = 0.05) -> torch.Tensor:
    """Identify distant states in which tail pursuit lacks useful closure.

    Observation 13 is range/10 km and observation 28 is inertial closure/
    340 m/s.  The defaults therefore activate beyond 6 km whenever closure
    is at most 17 m/s.  This is a kinematic admissibility test, not a reward.
    """
    if observation.ndim != 2 or observation.shape[-1] != 39:
        raise ValueError("observation must have [batch,39] shape")
    return (
        (observation[:, 13] >= float(min_range_normalized))
        & (observation[:, 28] <= float(max_closure_normalized))
    )


def reachability_option_legal_mask(
        observation: torch.Tensor,
        min_range_normalized: float = 0.60,
        max_closure_normalized: float = 0.05) -> torch.Tensor:
    """Apply a hard option shield only where pursuit is non-closing."""
    legal = macro_option_legal_mask(observation)
    guard = nonclosing_intercept_guard(
        observation, min_range_normalized=min_range_normalized,
        max_closure_normalized=max_closure_normalized)
    guarded = torch.zeros_like(legal)
    guarded[:, LEAD_INTERCEPT_OPTION] = True
    return torch.where(guard[:, None], guarded, legal)


def anchored_action_logits(
        logits: list[torch.Tensor], observation: torch.Tensor,
        option: torch.Tensor, strength: torch.Tensor | float) -> list[torch.Tensor]:
    """Add a finite macro-action prior to learned residual action logits."""
    if len(logits) != 4:
        raise ValueError("expected four action heads")
    anchor = macro_option_actions(observation, option, elapsed=1)
    # The reachability audit controller fires on every legal intercept shot.
    # Preserve that complete behaviour rather than anchoring manoeuvre only.
    intercept_shot = (
        (option == LEAD_INTERCEPT_OPTION) & (observation[:, 23] >= 0.5))
    anchor[:, 3] = torch.where(
        intercept_shot, torch.ones_like(anchor[:, 3]), anchor[:, 3])
    result = []
    for index, values in enumerate(logits):
        bias = F.one_hot(
            anchor[:, index], num_classes=values.shape[-1]
        ).to(dtype=values.dtype)
        result.append(values + torch.as_tensor(
            strength, device=values.device, dtype=values.dtype) * bias)
    return result


def guarded_fraction(observation: torch.Tensor) -> torch.Tensor:
    """Return a scalar diagnostic without synchronising the device."""
    return nonclosing_intercept_guard(observation).float().mean()


def reachability_shielded_actions(
        observation: torch.Tensor,
        learned_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Arbitrate between a learned expert and the LOS-intercept expert."""
    if learned_action.shape != (observation.shape[0], 4):
        raise ValueError("learned_action must have [batch,4] shape")
    guard = nonclosing_intercept_guard(observation)
    option = torch.full(
        (observation.shape[0],), LEAD_INTERCEPT_OPTION,
        dtype=torch.long, device=observation.device)
    intercept = macro_option_actions(observation, option, elapsed=1)
    intercept[:, 3] = (observation[:, 23] >= 0.5).long()
    return torch.where(guard[:, None], intercept, learned_action), guard


class HystereticReachabilityArbitrator:
    """Latch the intercept expert until a clear recovery certificate exists."""

    def __init__(self, exit_range_normalized: float = 0.55,
                 exit_closure_normalized: float = 0.10) -> None:
        self.exit_range_normalized = float(exit_range_normalized)
        self.exit_closure_normalized = float(exit_closure_normalized)
        self.active: torch.Tensor | None = None

    def __call__(self, observation: torch.Tensor, learned_action: torch.Tensor,
                 reset: torch.Tensor) -> torch.Tensor:
        if reset.shape != observation.shape[:1]:
            raise ValueError("reset must have one flag per observation")
        if self.active is None or self.active.shape != reset.shape:
            self.active = torch.zeros_like(reset, dtype=torch.bool)
        self.active = self.active.to(observation.device) & ~reset.bool()
        enter = nonclosing_intercept_guard(observation)
        use_intercept = self.active | enter
        option = torch.full(
            (observation.shape[0],), LEAD_INTERCEPT_OPTION,
            dtype=torch.long, device=observation.device)
        intercept = macro_option_actions(observation, option, elapsed=1)
        intercept[:, 3] = (observation[:, 23] >= 0.5).long()
        action = torch.where(
            use_intercept[:, None], intercept, learned_action)
        recovered = (
            (observation[:, 13] <= self.exit_range_normalized)
            & (observation[:, 28] >= self.exit_closure_normalized))
        launched = use_intercept & (observation[:, 23] >= 0.5)
        self.active = use_intercept & ~recovered & ~launched
        return action


class EpisodeLatchedReachabilityArbitrator:
    """Keep the intercept expert active until the environment resets."""

    def __init__(self) -> None:
        self.active: torch.Tensor | None = None

    def __call__(self, observation: torch.Tensor, learned_action: torch.Tensor,
                 reset: torch.Tensor) -> torch.Tensor:
        if reset.shape != observation.shape[:1]:
            raise ValueError("reset must have one flag per observation")
        if self.active is None or self.active.shape != reset.shape:
            self.active = torch.zeros_like(reset, dtype=torch.bool)
        self.active = self.active.to(observation.device) & ~reset.bool()
        self.active = self.active | nonclosing_intercept_guard(observation)
        option = torch.full(
            (observation.shape[0],), LEAD_INTERCEPT_OPTION,
            dtype=torch.long, device=observation.device)
        intercept = macro_option_actions(observation, option, elapsed=1)
        intercept[:, 3] = (observation[:, 23] >= 0.5).long()
        return torch.where(self.active[:, None], intercept, learned_action)


class SustainedReachabilityArbitrator:
    """Latch only after a sustained, rather than instantaneous, violation."""

    def __init__(self, enter_steps: int = 12) -> None:
        if enter_steps <= 0:
            raise ValueError("enter_steps must be positive")
        self.enter_steps = int(enter_steps)
        self.active: torch.Tensor | None = None
        self.violation_steps: torch.Tensor | None = None

    def __call__(self, observation: torch.Tensor, learned_action: torch.Tensor,
                 reset: torch.Tensor) -> torch.Tensor:
        if reset.shape != observation.shape[:1]:
            raise ValueError("reset must have one flag per observation")
        if self.active is None or self.active.shape != reset.shape:
            self.active = torch.zeros_like(reset, dtype=torch.bool)
            self.violation_steps = torch.zeros_like(reset, dtype=torch.long)
        reset = reset.to(observation.device).bool()
        self.active = self.active.to(observation.device) & ~reset
        self.violation_steps = self.violation_steps.to(observation.device)
        self.violation_steps = torch.where(
            reset, torch.zeros_like(self.violation_steps),
            self.violation_steps)
        violation = nonclosing_intercept_guard(observation)
        self.violation_steps = torch.where(
            violation, self.violation_steps + 1,
            torch.zeros_like(self.violation_steps))
        self.active = self.active | (self.violation_steps >= self.enter_steps)
        option = torch.full(
            (observation.shape[0],), LEAD_INTERCEPT_OPTION,
            dtype=torch.long, device=observation.device)
        intercept = macro_option_actions(observation, option, elapsed=1)
        intercept[:, 3] = (observation[:, 23] >= 0.5).long()
        return torch.where(self.active[:, None], intercept, learned_action)


__all__ = [
    "LEAD_INTERCEPT_OPTION",
    "HystereticReachabilityArbitrator",
    "EpisodeLatchedReachabilityArbitrator",
    "SustainedReachabilityArbitrator",
    "NUM_MACRO_OPTIONS",
    "anchored_action_logits",
    "guarded_fraction",
    "nonclosing_intercept_guard",
    "reachability_shielded_actions",
    "reachability_option_legal_mask",
]
