"""V10 behaviour-anchored reachability-constrained RHPPO."""

from .model import ReachabilityAnchoredOptionActorCritic
from .reachability import (
    LEAD_INTERCEPT_OPTION,
    HystereticReachabilityArbitrator,
    EpisodeLatchedReachabilityArbitrator,
    SustainedReachabilityArbitrator,
    anchored_action_logits,
    nonclosing_intercept_guard,
    reachability_option_legal_mask,
    reachability_shielded_actions,
)

__all__ = [
    "LEAD_INTERCEPT_OPTION",
    "HystereticReachabilityArbitrator",
    "EpisodeLatchedReachabilityArbitrator",
    "SustainedReachabilityArbitrator",
    "ReachabilityAnchoredOptionActorCritic",
    "anchored_action_logits",
    "nonclosing_intercept_guard",
    "reachability_option_legal_mask",
    "reachability_shielded_actions",
]
