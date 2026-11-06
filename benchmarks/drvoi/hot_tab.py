"""Exact hidden-opponent-type diagnostic benchmark from DRVOI-RHPPO V6."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HOTContext:
    prior: np.ndarray
    reliabilities: np.ndarray
    reward_matrix: np.ndarray


class HOTTab:
    """Two-step enumerable information acquisition task with four controls."""

    MODES = ("N0", "N1", "N2", "N3")

    def __init__(self, mode: str = "N2") -> None:
        self.set_control_mode(mode)
        self._rng = np.random.default_rng(0)
        self._latent = 0
        self._phase = 0
        self._last_response = -1

    def set_control_mode(self, mode: str) -> None:
        if mode not in self.MODES:
            raise ValueError(f"unknown HOT-Tab control mode: {mode}")
        self.mode = mode

    def context(self) -> HOTContext:
        prior = np.full(3, 1.0 / 3.0)
        if self.mode == "N0":
            reliabilities = np.full(3, 1.0 / 3.0)
        else:
            reliabilities = np.array([0.90, 0.60, 1.0 / 3.0])
        if self.mode == "N1":
            reward = np.ones((3, 3))
        else:
            reward = np.eye(3)
        if self.mode == "N3":
            prior = np.array([0.70, 0.20, 0.10])
        return HOTContext(prior, reliabilities, reward)

    @staticmethod
    def response_likelihood(reliability: float) -> np.ndarray:
        likelihood = np.full((3, 3), (1.0 - reliability) / 2.0)
        np.fill_diagonal(likelihood, reliability)
        return likelihood

    def oracle_q_voi(self, context: HOTContext | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return terminal Q[z,g] and information-action VOI for all three actions."""
        ctx = context or self.context()
        q_by_action = np.empty((3, 3, 3), dtype=np.float64)
        voi = np.empty(3, dtype=np.float64)
        for action, reliability in enumerate(ctx.reliabilities):
            likelihood = self.response_likelihood(float(reliability))
            predictive = ctx.prior @ likelihood
            posterior = (
                ctx.prior[:, None] * likelihood
                / np.clip(predictive[None, :], 1.0e-12, None))
            q = posterior.T @ ctx.reward_matrix
            q_by_action[action] = q
            informed = np.sum(predictive * q.max(axis=1))
            common = np.max(np.sum(predictive[:, None] * q, axis=0))
            voi[action] = informed - common
        return q_by_action, voi

    def oracle_posterior(self, action: int, response: int,
                         context: HOTContext | None = None) -> np.ndarray:
        ctx = context or self.context()
        likelihood = self.response_likelihood(float(ctx.reliabilities[action]))
        unnormalized = ctx.prior * likelihood[:, response]
        return unnormalized / unnormalized.sum()

    def sample_contexts(self, n: int) -> list[HOTContext]:
        if n <= 0:
            raise ValueError("n must be positive")
        return [self.context() for _ in range(n)]

    def reset(self, seed: int | None = None, opponent_type: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        ctx = self.context()
        self._latent = int(
            opponent_type if opponent_type is not None
            else self._rng.choice(3, p=ctx.prior))
        self._phase = 0
        self._last_response = -1
        return np.concatenate((ctx.prior, np.zeros(3))), {"phase": 0}

    def step(self, action: int):
        ctx = self.context()
        if self._phase == 0:
            likelihood = self.response_likelihood(float(ctx.reliabilities[action]))
            self._last_response = int(self._rng.choice(3, p=likelihood[self._latent]))
            self._phase = 1
            obs = np.concatenate((
                self.oracle_posterior(action, self._last_response),
                np.eye(3)[self._last_response]))
            return obs, 0.0, False, False, {"response": self._last_response}
        reward = float(ctx.reward_matrix[self._latent, action])
        self._phase = 2
        return np.zeros(6), reward, True, False, {"latent_type": self._latent}

    def get_latent_type(self) -> int:
        return self._latent
