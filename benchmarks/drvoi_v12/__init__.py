"""MRAT-RHPPO V12 components."""

from .buffer import FlatOptionBatch, compute_flat_smdp_gae

from .reachability_teacher import (
    BranchRolloutResult,
    aggregate_hypothesis_targets,
    instantaneous_reachability_components,
    instantaneous_reachability_quality,
    rollout_reachability_branch,
)
from .metrics import (
    METRIC_SCHEMA_VERSION,
    REQUIRED_UPDATE_METRICS,
    append_update_metrics,
    validate_update_metrics,
    write_json_atomic,
)
from .losses import (
    clipped_policy_loss,
    option_termination_loss,
    regularized_step_reward,
    variable_duration_gae,
)
from .execution import (
    OPPONENT_HYPOTHESES,
    V9ExecutionAdapter,
    V9ExecutionBundle,
    load_frozen_v9_execution,
    lead_intercept_expert_actions,
    opponent_hypothesis_actions,
)
from .features import (
    EXECUTION_FEATURE_DIM,
    ROUTER_FEATURE_DIM,
    build_execution_features,
    build_router_features,
    summarize_lowlevel_hidden,
)
from .reachability_model import (
    FrozenReachabilityPredictor,
    ReachabilityEnsemble,
    ReachabilityPrediction,
    load_frozen_reachability_predictor,
    reachability_distillation_loss,
)
from .router import (
    INTERCEPT_EXPERT,
    LEARNED_EXPERT,
    MRATArbitrator,
    MRATRouter,
    PersistentReachabilityShield,
    conservative_router_logits,
)
from .snapshot import (
    TorchWVRExecutionSnapshot,
    capture_execution_snapshot,
    repeat_execution_snapshot,
    restore_execution_snapshot,
    snapshot_scope,
)

__all__ = [
    "BranchRolloutResult",
    "EXECUTION_FEATURE_DIM",
    "FlatOptionBatch",
    "FrozenReachabilityPredictor",
    "INTERCEPT_EXPERT",
    "LEARNED_EXPERT",
    "METRIC_SCHEMA_VERSION",
    "MRATArbitrator",
    "MRATRouter",
    "OPPONENT_HYPOTHESES",
    "PersistentReachabilityShield",
    "REQUIRED_UPDATE_METRICS",
    "ReachabilityEnsemble",
    "ReachabilityPrediction",
    "ROUTER_FEATURE_DIM",
    "TorchWVRExecutionSnapshot",
    "V9ExecutionAdapter",
    "V9ExecutionBundle",
    "aggregate_hypothesis_targets",
    "append_update_metrics",
    "build_execution_features",
    "build_router_features",
    "capture_execution_snapshot",
    "clipped_policy_loss",
    "compute_flat_smdp_gae",
    "conservative_router_logits",
    "instantaneous_reachability_components",
    "instantaneous_reachability_quality",
    "load_frozen_v9_execution",
    "lead_intercept_expert_actions",
    "load_frozen_reachability_predictor",
    "option_termination_loss",
    "opponent_hypothesis_actions",
    "reachability_distillation_loss",
    "regularized_step_reward",
    "repeat_execution_snapshot",
    "restore_execution_snapshot",
    "rollout_reachability_branch",
    "snapshot_scope",
    "summarize_lowlevel_hidden",
    "validate_update_metrics",
    "variable_duration_gae",
    "write_json_atomic",
]
