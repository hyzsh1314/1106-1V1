"""Strict step/end contract shared by PBGR rollout and vector wrappers."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


class PBGRStepContractError(ValueError):
    """Raised when legacy step data is ambiguous or internally inconsistent."""


@dataclass(frozen=True)
class EpisodeEnd:
    terminated: bool
    truncated: bool
    reason: Optional[str] = None

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    @property
    def bootstrap_mask(self) -> float:
        return 0.0 if self.terminated else 1.0

    @property
    def continuation_mask(self) -> float:
        return 0.0 if self.done else 1.0


def episode_done(done: Any) -> bool:
    """Collapse legacy scalar/vector/dict done forms without changing meaning."""

    if isinstance(done, (bool, np.bool_)):
        return bool(done)
    if isinstance(done, dict):
        if not done:
            raise PBGRStepContractError("done dictionary must not be empty")
        values = list(done.values())
    elif isinstance(done, (list, tuple, np.ndarray)):
        values = np.asarray(done)
        if values.size == 0:
            raise PBGRStepContractError("done array must not be empty")
        return bool(np.all(values))
    else:
        raise PBGRStepContractError(
            "unsupported done type: {}".format(type(done).__name__)
        )
    return bool(np.all(np.asarray(values)))


def _strict_bool(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    array = np.asarray(value)
    if array.shape == () and array.dtype.kind == "b":
        return bool(array.item())
    raise PBGRStepContractError("{} must be an explicit boolean".format(name))


def classify_episode_end(
    done: Any, info: Dict[str, Any], timeout_is_terminal: bool = True
) -> EpisodeEnd:
    """Validate or explicitly adapt legacy end-of-episode semantics.

    The frozen COSR task defines its in-task ``max_steps`` outcome as a draw
    terminal, so ``timeout_is_terminal`` defaults to ``True``.  A collector or
    infrastructure cutoff is always a truncation and therefore bootstraps.
    All other terminal legacy steps still fail closed unless both Gymnasium
    flags are explicit.
    """

    if not isinstance(info, dict):
        raise PBGRStepContractError("info must be a dictionary")
    if not isinstance(timeout_is_terminal, bool):
        raise PBGRStepContractError("timeout_is_terminal must be a boolean")
    collapsed_done = episode_done(done)
    reason = info.get("termination_reason")
    if reason is not None and not isinstance(reason, str):
        raise PBGRStepContractError("termination_reason must be a string or None")
    reason_key = reason.lower() if reason is not None else None
    cutoff_scope = info.get("cutoff_scope")
    task_timeout_reasons = {"timeout", "time_limit", "max_steps", "episode_limit"}
    infrastructure_reasons = {
        "infrastructure_cutoff", "collector_cutoff", "rollout_cutoff"
    }
    if "terminated" not in info or "truncated" not in info:
        if not collapsed_done:
            terminated = False
            truncated = False
        elif cutoff_scope == "infrastructure" or reason_key in infrastructure_reasons:
            terminated = False
            truncated = True
        elif reason_key in task_timeout_reasons:
            terminated = timeout_is_terminal
            truncated = not timeout_is_terminal
        else:
            raise PBGRStepContractError(
                "terminal legacy step must explicitly provide terminated and truncated"
            )
    else:
        terminated = _strict_bool(info["terminated"], "terminated")
        truncated = _strict_bool(info["truncated"], "truncated")
    if terminated and truncated:
        raise PBGRStepContractError(
            "a step cannot be both terminated and truncated"
        )
    if collapsed_done != (terminated or truncated):
        raise PBGRStepContractError("done disagrees with terminated/truncated")
    return EpisodeEnd(terminated=terminated, truncated=truncated, reason=reason)


_MISSING = object()


def preserve_final_observation(
    info: Any, observation: Any, share_observation: Any = _MISSING
):
    """Return copied info carrying the pre-reset terminal observation(s)."""

    if isinstance(info, dict):
        result = dict(info)
    else:
        result = {"legacy_info": copy.deepcopy(info)}
    if "final_observation" not in result:
        result["final_observation"] = copy.deepcopy(observation)
    if share_observation is not _MISSING and "final_share_observation" not in result:
        result["final_share_observation"] = copy.deepcopy(share_observation)
    return result


def require_final_observation(info: Dict[str, Any], shared: bool = False):
    """Read a terminal observation, failing closed if the vector env lost it."""

    if not isinstance(info, dict):
        raise PBGRStepContractError("info must be a dictionary")
    key = "final_share_observation" if shared else "final_observation"
    if key not in info:
        raise PBGRStepContractError("missing {}".format(key))
    return info[key]
