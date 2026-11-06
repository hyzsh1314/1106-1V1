"""Recurrent hierarchical PPO backbone used only by DRVOI experiments."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Categorical

from .core import ACTION_DIMS


class RecurrentMultiDiscreteActorCritic(nn.Module):
    """GRU belief encoder with factorized actor and ext/int value critics."""

    def __init__(self, obs_dim: int = 39, hidden_size: int = 256,
                 action_embedding_size: int = 64) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.hidden_size = int(hidden_size)
        self.action_embedding = nn.Linear(sum(ACTION_DIMS), action_embedding_size)
        self.obs_encoder = nn.Linear(obs_dim, hidden_size)
        self.input_projection = nn.Linear(
            hidden_size + action_embedding_size, hidden_size)
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.belief_norm = nn.LayerNorm(hidden_size)
        self.actor_heads = nn.ModuleList(
            nn.Linear(hidden_size, size) for size in ACTION_DIMS)
        self.value_ext = nn.Linear(hidden_size, 1)
        self.value_int = nn.Linear(hidden_size, 1)

    def initial_hidden(self, batch_size: int, *, device=None) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def belief_step(self, observation: torch.Tensor,
                    previous_action_one_hot: torch.Tensor,
                    hidden: torch.Tensor,
                    reset: torch.Tensor | None = None):
        if reset is not None:
            hidden = hidden * (~reset.bool()).float().unsqueeze(-1)
        obs_features = torch.relu(self.obs_encoder(observation))
        action_features = torch.relu(self.action_embedding(previous_action_one_hot))
        recurrent_input = torch.relu(self.input_projection(
            torch.cat((obs_features, action_features), dim=-1)))
        next_hidden = self.gru(recurrent_input, hidden)
        belief = self.belief_norm(next_hidden)
        return belief, next_hidden

    def distributions(self, belief: torch.Tensor,
                      observation: torch.Tensor | None = None) -> list[Categorical]:
        logits = [head(belief) for head in self.actor_heads]
        if observation is not None:
            impossible = torch.finfo(logits[0].dtype).min
            altitude = observation[:, 0]
            altitude_logits = logits[0].clone()
            altitude_logits[:, 2] = altitude_logits[:, 2].masked_fill(
                altitude <= 0.8, impossible)
            altitude_logits[:, 1] = altitude_logits[:, 1].masked_fill(
                altitude <= 0.6, impossible)
            logits[0] = altitude_logits
            fire_logits = logits[-1].clone()
            fire_logits[:, 1] = fire_logits[:, 1].masked_fill(
                observation[:, 23] < 0.5, impossible)
            logits[-1] = fire_logits
        return [Categorical(logits=value) for value in logits]

    def act(self, observation: torch.Tensor, previous_action_one_hot: torch.Tensor,
            hidden: torch.Tensor, reset: torch.Tensor | None = None,
            deterministic: bool = False):
        belief, next_hidden = self.belief_step(
            observation, previous_action_one_hot, hidden, reset)
        distributions = self.distributions(belief, observation)
        actions = torch.stack([
            distribution.probs.argmax(-1) if deterministic else distribution.sample()
            for distribution in distributions
        ], dim=-1)
        log_prob = sum(
            distribution.log_prob(actions[:, index])
            for index, distribution in enumerate(distributions))
        normalized_entropy = sum(
            distribution.entropy() / math.log(size)
            for distribution, size in zip(distributions, ACTION_DIMS)) / len(ACTION_DIMS)
        return {
            "action": actions,
            "log_prob": log_prob,
            "normalized_entropy": normalized_entropy,
            "value_ext": self.value_ext(belief).squeeze(-1),
            "value_int": self.value_int(belief).squeeze(-1),
            "belief": belief,
            "hidden": next_hidden,
        }

    def evaluate_actions(self, belief: torch.Tensor, actions: torch.Tensor,
                         observation: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        distributions = self.distributions(belief, observation)
        log_prob = sum(
            distribution.log_prob(actions[:, index])
            for index, distribution in enumerate(distributions))
        entropy = sum(
            distribution.entropy() / math.log(size)
            for distribution, size in zip(distributions, ACTION_DIMS)) / len(ACTION_DIMS)
        return {
            "log_prob": log_prob,
            "normalized_entropy": entropy,
            "value_ext": self.value_ext(belief).squeeze(-1),
            "value_int": self.value_int(belief).squeeze(-1),
        }

    def sequence_beliefs(self, observations: torch.Tensor,
                         previous_actions_one_hot: torch.Tensor,
                         initial_hidden: torch.Tensor,
                         resets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if observations.ndim != 3:
            raise ValueError("sequence observations must have [T,B,D] shape")
        hidden = initial_hidden
        beliefs = []
        for step in range(observations.shape[0]):
            belief, hidden = self.belief_step(
                observations[step], previous_actions_one_hot[step], hidden,
                resets[step])
            beliefs.append(belief)
        return torch.stack(beliefs), hidden


def dual_channel_gae(reward_ext: torch.Tensor, reward_int: torch.Tensor,
                     value_ext: torch.Tensor, value_int: torch.Tensor,
                     next_value_ext: torch.Tensor, next_value_int: torch.Tensor,
                     done: torch.Tensor, gamma: float = 0.99,
                     gae_lambda: float = 0.95):
    """Compute ext/int GAE separately; terminal transitions never bootstrap."""
    if reward_ext.shape != reward_int.shape or reward_ext.ndim != 2:
        raise ValueError("dual GAE rewards must share [T,B] shape")
    mask = (~done.bool()).float()
    delta_ext = reward_ext + gamma * next_value_ext * mask - value_ext
    delta_int = reward_int + gamma * next_value_int * mask - value_int
    advantage_ext = torch.zeros_like(reward_ext)
    advantage_int = torch.zeros_like(reward_int)
    running_ext = torch.zeros_like(reward_ext[0])
    running_int = torch.zeros_like(reward_int[0])
    for step in reversed(range(reward_ext.shape[0])):
        running_ext = delta_ext[step] + gamma * gae_lambda * mask[step] * running_ext
        running_int = delta_int[step] + gamma * gae_lambda * mask[step] * running_int
        advantage_ext[step] = running_ext
        advantage_int[step] = running_int
    return {
        "advantage_ext": advantage_ext,
        "advantage_int": advantage_int,
        "return_ext": advantage_ext + value_ext,
        "return_int": advantage_int + value_int,
    }
