"""COACH-RHPPO V9 components."""

from .model import AdaptiveOptionRecurrentActorCritic
from .option_prior import confidence_gated_option_prior
from .training import (
    linear_anneal,
    option_segment_targets,
    polyak_update,
    progress_violation_events,
)

__all__ = [
    "AdaptiveOptionRecurrentActorCritic",
    "confidence_gated_option_prior",
    "linear_anneal",
    "option_segment_targets",
    "polyak_update",
    "progress_violation_events",
]



