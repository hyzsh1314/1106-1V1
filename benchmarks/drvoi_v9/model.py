"""Option-conditioned recurrent actor-critic for COACH-RHPPO."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Categorical

from benchmarks.drvoi.core import ACTION_DIMS
from benchmarks.drvoi.rhppo import RecurrentMultiDiscreteActorCritic


class AdaptiveOptionRecurrentActorCritic(RecurrentMultiDiscreteActorCritic):
    """RHPPO backbone with explicit high-level and conditioned low-level heads.

    ``load_from_rhppo`` initializes the option channels to zero, making every
    option reproduce the source B0 policy before any V9 optimization.
    """

    def __init__(self, obs_dim: int = 39, hidden_size: int = 256,
                 response_feature_dim: int = 24, num_options: int = 7,
                 option_embedding_size: int = 16) -> None:
        super().__init__(obs_dim=obs_dim, hidden_size=hidden_size)
        self.response_feature_dim = int(response_feature_dim)
        self.num_options = int(num_options)
        self.option_embedding_size = int(option_embedding_size)
        self.option_embedding = nn.Embedding(
            self.num_options, self.option_embedding_size)
        conditioned_size = self.hidden_size + self.option_embedding_size
        self.actor_heads = nn.ModuleList(
            nn.Linear(conditioned_size, size) for size in ACTION_DIMS)
        option_input = self.hidden_size + self.response_feature_dim
        self.option_policy = nn.Sequential(
            nn.LayerNorm(option_input),
            nn.Linear(option_input, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.num_options),
        )
        nn.init.zeros_(self.option_policy[-1].weight)
        nn.init.zeros_(self.option_policy[-1].bias)
        self.option_q_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(option_input),
                nn.Linear(option_input, self.hidden_size),
                nn.SiLU(),
                nn.Linear(self.hidden_size, self.num_options),
            ) for _ in range(2)
        ])

    def load_from_rhppo(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load a B0 checkpoint while keeping initial behavior option-free."""
        own = self.state_dict()
        backbone_prefixes = (
            "action_embedding.", "obs_encoder.", "input_projection.",
            "gru.", "belief_norm.", "value_ext.", "value_int.",
        )
        for key, value in state_dict.items():
            if key.startswith(backbone_prefixes):
                own[key].copy_(value)
        for index, head in enumerate(self.actor_heads):
            source_weight = state_dict[f"actor_heads.{index}.weight"]
            source_bias = state_dict[f"actor_heads.{index}.bias"]
            with torch.no_grad():
                head.weight.zero_()
                head.weight[:, :self.hidden_size].copy_(source_weight)
                head.bias.copy_(source_bias)

    def option_logits(self, belief: torch.Tensor,
                      response_features: torch.Tensor,
                      legal_mask: torch.Tensor,
                      prior_logits: torch.Tensor | None = None,
                      prior_coefficient: float = 0.0) -> torch.Tensor:
        if response_features.shape != (
                belief.shape[0], self.response_feature_dim):
            raise ValueError("response features have the wrong shape")
        if legal_mask.shape != (belief.shape[0], self.num_options):
            raise ValueError("legal mask has the wrong shape")
        if bool((~legal_mask.bool().any(-1)).any()):
            raise ValueError("each row must contain a legal option")
        logits = self.option_policy(torch.cat((belief, response_features), -1))
        if prior_logits is not None and prior_coefficient != 0.0:
            if prior_logits.shape != logits.shape:
                raise ValueError("prior logits have the wrong shape")
            finite_prior = torch.where(
                legal_mask.bool(), prior_logits, torch.zeros_like(prior_logits))
            logits = logits + float(prior_coefficient) * finite_prior
        impossible = torch.finfo(logits.dtype).min
        return logits.masked_fill(~legal_mask.bool(), impossible)

    def option_distribution(self, *args, **kwargs) -> Categorical:
        return Categorical(logits=self.option_logits(*args, **kwargs))

    def option_q_values(self, belief: torch.Tensor,
                        response_features: torch.Tensor,
                        legal_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat((belief, response_features), -1)
        impossible = torch.finfo(inputs.dtype).min
        return tuple(
            head(inputs).masked_fill(~legal_mask.bool(), impossible)
            for head in self.option_q_heads
        )

    def conditioned_distributions(
            self, belief: torch.Tensor, option: torch.Tensor,
            observation: torch.Tensor | None = None) -> list[Categorical]:
        option_features = self.option_embedding(option.long())
        features = torch.cat((belief, option_features), -1)
        logits = [head(features) for head in self.actor_heads]
        if observation is not None:
            impossible = torch.finfo(logits[0].dtype).min
            altitude_logits = logits[0].clone()
            altitude_logits[:, 2] = altitude_logits[:, 2].masked_fill(
                observation[:, 0] <= 0.8, impossible)
            altitude_logits[:, 1] = altitude_logits[:, 1].masked_fill(
                observation[:, 0] <= 0.6, impossible)
            logits[0] = altitude_logits
            fire_logits = logits[-1].clone()
            fire_logits[:, 1] = fire_logits[:, 1].masked_fill(
                observation[:, 23] < 0.5, impossible)
            logits[-1] = fire_logits
        return [Categorical(logits=value) for value in logits]

    def act_conditioned(self, observation: torch.Tensor,
                        previous_action_one_hot: torch.Tensor,
                        hidden: torch.Tensor, option: torch.Tensor,
                        reset: torch.Tensor | None = None,
                        deterministic: bool = False):
        belief, next_hidden = self.belief_step(
            observation, previous_action_one_hot, hidden, reset)
        distributions = self.conditioned_distributions(
            belief, option, observation)
        actions = torch.stack([
            distribution.probs.argmax(-1) if deterministic
            else distribution.sample()
            for distribution in distributions
        ], -1)
        log_prob = sum(
            distribution.log_prob(actions[:, index])
            for index, distribution in enumerate(distributions))
        entropy = sum(
            distribution.entropy() / math.log(size)
            for distribution, size in zip(distributions, ACTION_DIMS)
        ) / len(ACTION_DIMS)
        return {
            "action": actions,
            "log_prob": log_prob,
            "normalized_entropy": entropy,
            "value_ext": self.value_ext(belief).squeeze(-1),
            "value_int": self.value_int(belief).squeeze(-1),
            "belief": belief,
            "hidden": next_hidden,
        }

    def evaluate_conditioned(self, belief: torch.Tensor,
                             option: torch.Tensor,
                             actions: torch.Tensor,
                             observation: torch.Tensor | None = None):
        distributions = self.conditioned_distributions(
            belief, option, observation)
        log_prob = sum(
            distribution.log_prob(actions[:, index])
            for index, distribution in enumerate(distributions))
        entropy = sum(
            distribution.entropy() / math.log(size)
            for distribution, size in zip(distributions, ACTION_DIMS)
        ) / len(ACTION_DIMS)
        return {
            "log_prob": log_prob,
            "normalized_entropy": entropy,
            "value_ext": self.value_ext(belief).squeeze(-1),
            "value_int": self.value_int(belief).squeeze(-1),
        }

