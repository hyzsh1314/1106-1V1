"""Auditable online feature construction for reachability and routing."""

from __future__ import annotations

from typing import Mapping

import torch
from torch.nn import functional as F

from benchmarks.drvoi_v7_1.macro_options import NUM_MACRO_OPTIONS


OBSERVATION_DIM = 39
V9_HIDDEN_DIM = 256
PREVIOUS_ACTION_DIM = 13
LOWLEVEL_SUMMARY_DIM = 8
EXPERT_COUNT = 2
EXECUTION_FEATURE_DIM = (
    OBSERVATION_DIM
    + V9_HIDDEN_DIM
    + PREVIOUS_ACTION_DIM
    + NUM_MACRO_OPTIONS
    + 1
    + LOWLEVEL_SUMMARY_DIM
    + EXPERT_COUNT
    + 1
)
ROUTER_REACHABILITY_DIM = 8
ROUTER_FEATURE_DIM = EXECUTION_FEATURE_DIM + ROUTER_REACHABILITY_DIM


def summarize_lowlevel_hidden(
        lowlevel_hidden: torch.Tensor,
        num_envs: int,
) -> torch.Tensor:
    """Compress each aircraft controller hidden state to four moments."""
    if lowlevel_hidden.shape != (num_envs * 2, 1, 128):
        raise ValueError("lowlevel_hidden must have [2*num_envs,1,128] shape")
    values = lowlevel_hidden.reshape(num_envs, 2, 128)
    mean = values.mean(-1)
    std = values.std(-1, unbiased=False)
    rms = values.square().mean(-1).sqrt()
    max_abs = values.abs().amax(-1)
    return torch.stack((mean, std, rms, max_abs), -1).flatten(1)


def build_execution_features(
        learner_observation: torch.Tensor,
        execution_state: Mapping[str, torch.Tensor],
        max_internal_option_age: int = 150,
        max_expert_age: int = 150,
) -> torch.Tensor:
    """Build features available online without opponent labels/future state."""
    if learner_observation.ndim != 2 or learner_observation.shape[-1] != 39:
        raise ValueError("learner_observation must have [batch,39] shape")
    batch = learner_observation.shape[0]
    hidden = execution_state["hidden"]
    previous_action = execution_state["previous_action"]
    active_option = execution_state["active_option"].long()
    option_age = execution_state["option_age"]
    selected_expert = execution_state["selected_expert"].long()
    expert_age = execution_state.get(
        "expert_age", torch.zeros_like(option_age))
    if hidden.shape != (batch, V9_HIDDEN_DIM):
        raise ValueError("V9 hidden state has the wrong shape")
    if previous_action.shape != (batch, PREVIOUS_ACTION_DIM):
        raise ValueError("previous action has the wrong shape")
    option_one_hot = F.one_hot(
        active_option, NUM_MACRO_OPTIONS).to(learner_observation.dtype)
    expert_one_hot = F.one_hot(
        selected_expert, EXPERT_COUNT).to(learner_observation.dtype)
    lowlevel = summarize_lowlevel_hidden(
        execution_state["lowlevel_hidden"], batch)
    option_age_feature = (
        option_age.float() / max(float(max_internal_option_age), 1.0)
    ).clamp(0.0, 1.0)[:, None]
    expert_age_feature = (
        expert_age.float() / max(float(max_expert_age), 1.0)
    ).clamp(0.0, 1.0)[:, None]
    features = torch.cat((
        learner_observation,
        hidden,
        previous_action,
        option_one_hot,
        option_age_feature,
        lowlevel,
        expert_one_hot,
        expert_age_feature,
    ), -1)
    if features.shape != (batch, EXECUTION_FEATURE_DIM):
        raise RuntimeError("execution feature dimension drifted")
    return features


def build_router_features(
        standardized_execution_features: torch.Tensor,
        reachability: Mapping[str, torch.Tensor],
        horizon: float = 30.0,
) -> torch.Tensor:
    """Combine online execution state with two-expert reach predictions."""
    if (standardized_execution_features.ndim != 2
            or standardized_execution_features.shape[-1]
            != EXECUTION_FEATURE_DIM):
        raise ValueError("standardized execution features have the wrong shape")
    batch = standardized_execution_features.shape[0]
    values = []
    for key in ("lcb", "score_mean", "probability", "time"):
        value = reachability[key]
        if value.shape != (batch, 2):
            raise ValueError(f"reachability {key!r} has the wrong shape")
        values.append(value / float(horizon) if key == "time" else value)
    features = torch.cat((standardized_execution_features, *values), -1)
    if features.shape != (batch, ROUTER_FEATURE_DIM):
        raise RuntimeError("router feature dimension drifted")
    return features
