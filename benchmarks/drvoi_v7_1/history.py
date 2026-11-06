"""Leak-free explicit observation/action histories for DRVOI V7.1."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from benchmarks.drvoi.core import ACTION_DIMS


NEUTRAL_ACTION = (1, 2, 1, 0)


def multi_discrete_history_to_one_hot(actions: torch.Tensor) -> torch.Tensor:
    """Encode `[...,4]` actions as `[...,13]` without flattening history."""
    if actions.shape[-1] != len(ACTION_DIMS):
        raise ValueError("actions must end with four multi-discrete heads")
    encoded = [
        torch.nn.functional.one_hot(
            actions[..., index].long(), classes).float()
        for index, classes in enumerate(ACTION_DIMS)
    ]
    return torch.cat(encoded, -1)


@dataclass(frozen=True)
class HistoryBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    valid: torch.Tensor
    wrong_delay_actions: torch.Tensor

    @property
    def context(self) -> torch.Tensor:
        return torch.cat((
            self.observations.flatten(-2), self.valid.float()), -1)

    @property
    def action_input(self) -> torch.Tensor:
        return self.actions.flatten(-2)

    @property
    def wrong_delay_action_input(self) -> torch.Tensor:
        return self.wrong_delay_actions.flatten(-2)


def assemble_episode_history(
        observations: torch.Tensor,
        actions: torch.Tensor,
        length: int = 3,
        neutral_action: tuple[int, int, int, int] = NEUTRAL_ACTION,
        ) -> HistoryBatch:
    """Build histories for one complete episode with no pre-episode leakage.

    Correct action history at `t` ends in `a_t`.  The wrong-delay control is
    the non-overlapping older window `[a_(t-5),a_(t-4),a_(t-3)]`; critically,
    it cannot expose the reactive controller's causal `a_(t-2)`. Missing
    entries use the neutral action while the observation validity mask is 0.
    """
    if length <= 0:
        raise ValueError("history length must be positive")
    if observations.ndim != 2 or observations.shape[-1] != 39:
        raise ValueError("observations must have [T,39] shape")
    if actions.shape != (observations.shape[0], 4):
        raise ValueError("actions must have [T,4] shape")
    steps = observations.shape[0]
    obs_history = torch.zeros(
        steps, length, observations.shape[-1],
        dtype=observations.dtype, device=observations.device)
    neutral = torch.tensor(
        neutral_action, dtype=actions.dtype, device=actions.device)
    action_history = neutral.expand(steps, length, -1).clone()
    wrong_history = neutral.expand(steps, length, -1).clone()
    valid = torch.zeros(steps, length, dtype=torch.bool, device=observations.device)
    for slot in range(length):
        lag = length - 1 - slot
        if lag < steps:
            obs_history[lag:, slot] = observations[:steps - lag]
            action_history[lag:, slot] = actions[:steps - lag]
            valid[lag:, slot] = True
        wrong_lag = lag + length
        if wrong_lag < steps:
            wrong_history[wrong_lag:, slot] = actions[:steps - wrong_lag]
    return HistoryBatch(
        observations=obs_history,
        actions=multi_discrete_history_to_one_hot(action_history),
        valid=valid,
        wrong_delay_actions=multi_discrete_history_to_one_hot(wrong_history),
    )


class VectorHistoryBuffer:
    """Static-shape history buffer for parallel environments."""

    def __init__(self, num_envs: int, observation_dim: int = 39,
                 length: int = 3, device: str | torch.device = "cpu") -> None:
        if num_envs <= 0 or length <= 0:
            raise ValueError("num_envs and length must be positive")
        self.num_envs = int(num_envs)
        self.observation_dim = int(observation_dim)
        self.length = int(length)
        self.device = torch.device(device)
        self.observations = torch.zeros(
            num_envs, length, observation_dim, device=self.device)
        neutral = torch.tensor(
            NEUTRAL_ACTION, dtype=torch.long, device=self.device)
        self.actions = neutral.expand(num_envs, length, -1).clone()
        self.valid = torch.zeros(
            num_envs, length, dtype=torch.bool, device=self.device)
        # A second, non-overlapping window is retained solely for the
        # wrong-delay control.  It excludes every action in the current L=3
        # history, including the known causal a[t-2].
        self._action_lag = neutral.expand(num_envs, length * 2, -1).clone()

    def clear(self, reset: torch.Tensor | None = None) -> None:
        if reset is None:
            selected = torch.ones(
                self.num_envs, dtype=torch.bool, device=self.device)
        else:
            selected = reset.to(device=self.device, dtype=torch.bool)
            if selected.shape != (self.num_envs,):
                raise ValueError("reset must have [num_envs] shape")
        self.observations[selected] = 0.0
        neutral = torch.tensor(
            NEUTRAL_ACTION, dtype=torch.long, device=self.device)
        self.actions[selected] = neutral
        self._action_lag[selected] = neutral
        self.valid[selected] = False

    def append(self, observation: torch.Tensor, action: torch.Tensor,
               reset: torch.Tensor | None = None) -> HistoryBatch:
        if observation.shape != (self.num_envs, self.observation_dim):
            raise ValueError("observation shape does not match buffer")
        if action.shape != (self.num_envs, 4):
            raise ValueError("action must have [num_envs,4] shape")
        if reset is not None:
            self.clear(reset)
        self.observations = torch.roll(self.observations, -1, dims=1)
        self.actions = torch.roll(self.actions, -1, dims=1)
        self.valid = torch.roll(self.valid, -1, dims=1)
        self._action_lag = torch.roll(self._action_lag, -1, dims=1)
        self.observations[:, -1] = observation
        self.actions[:, -1] = action
        self.valid[:, -1] = True
        self._action_lag[:, -1] = action
        wrong = self._action_lag[:, :self.length]
        return HistoryBatch(
            observations=self.observations.clone(),
            actions=multi_discrete_history_to_one_hot(self.actions),
            valid=self.valid.clone(),
            wrong_delay_actions=multi_discrete_history_to_one_hot(wrong),
        )
