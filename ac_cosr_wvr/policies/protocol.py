"""The only policy output type accepted by COSR evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod


class PhysicalPolicy(ABC):
    """A frozen controller that returns physical F-16 control commands.

    Order is always ``[aileron, elevator, rudder, throttle]`` with bounds
    ``[-1, -1, -1, 0.4]`` to ``[1, 1, 1, 0.9]``.
    """

    policy_id = None

    @abstractmethod
    def reset(self):
        """Reset episode-local controller state without changing parameters."""

    @abstractmethod
    def act(self, observation, deterministic=True):
        """Return one bounded physical action."""
