"""Versioned training-time metric contract for MRAT-RHPPO."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping


METRIC_SCHEMA_VERSION = "MRAT-RHPPO-V12-METRICS-1"

REQUIRED_UPDATE_METRICS = (
    # Task outcomes.
    "update",
    "environment_decisions",
    "completed_episodes",
    "learner_wins",
    "learner_losses",
    "draws",
    "timeouts",
    "crashes",
    "mean_reward_ext",
    # Weapon chain.
    "fire_requests",
    "accepted_launches",
    "rejected_fire_requests",
    "legal_fire_opportunities",
    "hits",
    # Arbitration and option semantics.
    "learned_expert_steps",
    "intercept_expert_steps",
    "router_intervention_steps",
    "forced_intervention_steps",
    "expert_switches",
    "completed_option_segments",
    "mean_option_dwell",
    # Reachability diagnostics.
    "reachability_lcb_learned_mean",
    "reachability_lcb_intercept_mean",
    "reachability_regret_mean",
    "force_condition_fraction",
    # PPO health.
    "router_policy_loss",
    "router_value_loss",
    "router_approx_kl",
    "router_entropy",
    "termination_loss",
    "low_policy_loss",
    "low_approx_kl",
    # Constraints.
    "bad_reachability_cost",
    "crash_cost",
    "bad_reachability_multiplier",
    "crash_multiplier",
    # Runtime and compute accounting.
    "rollout_s",
    "teacher_s",
    "update_s",
    "environment_decisions_per_s",
    "online_environment_decisions",
    "model_branch_decisions",
    "wall_time_s",
)


def validate_update_metrics(metrics: Mapping[str, object]) -> None:
    """Reject incomplete/non-finite update records before they reach disk."""
    missing = [key for key in REQUIRED_UPDATE_METRICS if key not in metrics]
    if missing:
        raise ValueError(f"missing required V12 metrics: {missing}")
    for key in REQUIRED_UPDATE_METRICS:
        value = metrics[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"metric {key!r} must be a real scalar")
        if not math.isfinite(float(value)):
            raise ValueError(f"metric {key!r} is not finite")
    if int(metrics["environment_decisions"]) < 0:
        raise ValueError("environment_decisions must be non-negative")
    if int(metrics["model_branch_decisions"]) < 0:
        raise ValueError("model_branch_decisions must be non-negative")


def append_update_metrics(path: Path, metrics: Mapping[str, object]) -> None:
    """Append one validated JSONL record with an explicit schema version."""
    validate_update_metrics(metrics)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metric_schema": METRIC_SCHEMA_VERSION, **dict(metrics)}
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    temporary.replace(path)
