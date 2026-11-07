"""Behaviour-anchored option model for BARC-RHPPO."""

from __future__ import annotations

import torch
from torch.distributions import Categorical

from benchmarks.drvoi_v9.model import AdaptiveOptionRecurrentActorCritic

from .reachability import anchored_action_logits


class ReachabilityAnchoredOptionActorCritic(
        AdaptiveOptionRecurrentActorCritic):
    """COACH actor-critic whose option primitives remain behaviourally real.

    The neural heads learn residual logits.  A finite prior is added to the
    action prescribed by the active closed-loop macro option.  Unlike hard
    action replacement, PPO can override the prior when evidence supports it.
    """

    def __init__(self, *args, anchor_strength: float = 8.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if anchor_strength < 0.0:
            raise ValueError("anchor_strength must be non-negative")
        self.register_buffer(
            "anchor_strength",
            torch.tensor(float(anchor_strength), dtype=torch.float32))

    def conditioned_distributions(
            self, belief: torch.Tensor, option: torch.Tensor,
            observation: torch.Tensor | None = None) -> list[Categorical]:
        distributions = super().conditioned_distributions(
            belief, option, observation)
        if observation is None:
            return distributions
        logits = anchored_action_logits(
            [distribution.logits for distribution in distributions],
            observation, option, self.anchor_strength)
        return [Categorical(logits=value) for value in logits]


__all__ = ["ReachabilityAnchoredOptionActorCritic"]
