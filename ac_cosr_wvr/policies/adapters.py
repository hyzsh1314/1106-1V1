"""Adapters that make policy families obey the physical action protocol."""

from __future__ import annotations

import numpy as np

from .action_mapping import LegacyDiscreteActionMapper, validate_physical_action
from .protocol import PhysicalPolicy


class LegacyDiscretePolicyAdapter(PhysicalPolicy):
    def __init__(self, policy_id, actor, mapper=None):
        self.policy_id = policy_id
        self.actor = actor
        self.mapper = mapper or LegacyDiscreteActionMapper()

    def reset(self):
        reset = getattr(self.actor, "reset", None)
        if reset is not None:
            reset()

    def act(self, observation, deterministic=True):
        try:
            output = self.actor.act(observation, deterministic=deterministic)
        except AttributeError:
            output = self.actor(observation)
        return self.mapper(output)


class RecurrentLAGPPOAdapter(PhysicalPolicy):
    """Adapter for the repository's real ``algorithms.ppo.PPOPolicy.act`` API.

    It owns per-episode ``rnn_states_actor`` and ``masks`` exactly as the LAG
    PPO implementation expects: ``act(obs, rnn_states_actor, masks,
    deterministic=False) -> (actions, next_rnn_states_actor)``. No checkpoint
    path is embedded here; loading and hash checks remain registry concerns.
    """

    def __init__(self, policy_id, ppo_policy, mapper=None):
        self.policy_id = policy_id
        self.ppo_policy = ppo_policy
        self.mapper = mapper or LegacyDiscreteActionMapper()
        actor = getattr(ppo_policy, "actor", None)
        if actor is None:
            raise TypeError("ppo_policy must expose the LAG PPO actor")
        try:
            self._layers = int(actor.recurrent_hidden_layers)
            self._hidden = int(actor.recurrent_hidden_size)
        except AttributeError as error:
            raise TypeError("LAG PPO actor lacks recurrent state metadata") from error
        self._rnn_states = None
        self._masks = None
        self.reset()

    def reset(self):
        self._rnn_states = np.zeros((1, self._layers, self._hidden), dtype=np.float32)
        # A zero mask forces reset at the first recurrent transition.
        self._masks = np.zeros((1, 1), dtype=np.float32)
        prep = getattr(self.ppo_policy, "prep_rollout", None)
        if prep is not None:
            prep()

    @staticmethod
    def _array(value):
        detach = getattr(value, "detach", None)
        if detach is not None:
            value = detach().cpu().numpy()
        return np.asarray(value)

    def act(self, observation, deterministic=True):
        obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        actions, next_state = self.ppo_policy.act(obs, self._rnn_states, self._masks,
                                                   deterministic=bool(deterministic))
        self._rnn_states = self._array(next_state).astype(np.float32, copy=False)
        self._masks = np.ones((1, 1), dtype=np.float32)
        return self.mapper(self._array(actions).reshape(-1))


def optional_ppo_checkpoint_forward(policy):
    """Perform a forward-interface check only for an already loaded policy.

    The repository has no independently verified formal PPO checkpoint in this
    branch, so callers without a loaded object receive a blocked status rather
    than a fabricated path or a synthetic loading claim.
    """
    if policy is None:
        return {"status": "BLOCKED_CHECKPOINT_UNAVAILABLE"}
    actor = getattr(policy, "actor", None)
    if actor is None:
        return {"status": "BLOCKED_INVALID_POLICY_INTERFACE"}
    return {"status": "READY_FOR_CALLER_SUPPLIED_FORWARD_CHECK"}


class ContinuousPolicyAdapter(PhysicalPolicy):
    def __init__(self, policy_id, actor):
        self.policy_id = policy_id
        self.actor = actor

    def reset(self):
        reset = getattr(self.actor, "reset", None)
        if reset is not None:
            reset()

    def act(self, observation, deterministic=True):
        try:
            action = self.actor.act(observation, deterministic=deterministic)
        except AttributeError:
            action = self.actor(observation)
        return validate_physical_action(action, clip=True)


class ConstantPhysicalPolicy(PhysicalPolicy):
    """Rule/probe policy useful for deterministic tests and smoke runs."""

    def __init__(self, policy_id, action):
        self.policy_id = policy_id
        self._action = validate_physical_action(action, clip=False)

    def reset(self):
        return None

    def act(self, observation, deterministic=True):
        return self._action.copy()
