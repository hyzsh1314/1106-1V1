"""Cross-fitted extrinsic Q proxy for DRVOI candidate evaluation."""

from __future__ import annotations

import torch
from torch import nn

from .core import ACTION_DIMS, JOINT_ACTIONS, action_ids_to_multi_one_hot


class ActionEmbeddingQ(nn.Module):
    """Shared scalar Q(x,g) with an explicit 13-D joint-action encoding."""

    def __init__(self, context_dim: int, hidden_size: int = 256,
                 action_embedding_size: int = 32) -> None:
        super().__init__()
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, hidden_size), nn.SiLU())
        self.action_encoder = nn.Sequential(
            nn.Linear(13, action_embedding_size), nn.SiLU())
        self.output = nn.Sequential(
            nn.Linear(hidden_size + action_embedding_size, hidden_size),
            nn.SiLU(), nn.Linear(hidden_size, 1))
        self.register_buffer(
            "all_action_encodings",
            action_ids_to_multi_one_hot(torch.arange(JOINT_ACTIONS)))

    def forward(self, context: torch.Tensor,
                action_one_hot: torch.Tensor) -> torch.Tensor:
        context_features = self.context_encoder(context)
        action_features = self.action_encoder(action_one_hot)
        return self.output(torch.cat((context_features, action_features), -1)).squeeze(-1)

    def all_actions(self, context: torch.Tensor) -> torch.Tensor:
        batch = context.shape[0]
        expanded_context = context[:, None, :].expand(-1, JOINT_ACTIONS, -1)
        actions = self.all_action_encodings[None].expand(batch, -1, -1)
        return self(
            expanded_context.reshape(batch * JOINT_ACTIONS, -1),
            actions.reshape(batch * JOINT_ACTIONS, -1)).reshape(
                batch, JOINT_ACTIONS)


class FactorizedDuelingQ(nn.Module):
    """Context-conditioned value plus identifiable action advantages.

    The earlier action-embedding critic could explain nearly all return
    variation with its state trunk and consequently learned an almost
    context-independent preferred action.  This critic explicitly separates
    the state value from zero-mean per-head and pairwise action advantages.
    It therefore uses the factorization of the four WVR action heads while
    retaining interactions such as altitude-command x throttle.
    """

    def __init__(self, context_dim: int, hidden_size: int = 256) -> None:
        super().__init__()
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, hidden_size), nn.SiLU())
        self.value = nn.Linear(hidden_size, 1)
        self.advantage_heads = nn.ModuleList(
            nn.Linear(hidden_size, size) for size in ACTION_DIMS)
        self.pairs = tuple(
            (left, right) for left in range(len(ACTION_DIMS))
            for right in range(left + 1, len(ACTION_DIMS)))
        self.pair_heads = nn.ModuleList(
            nn.Linear(hidden_size, ACTION_DIMS[left] * ACTION_DIMS[right])
            for left, right in self.pairs)
        self.register_buffer(
            "all_action_encodings",
            action_ids_to_multi_one_hot(torch.arange(JOINT_ACTIONS)))

    @staticmethod
    def _segments(action_one_hot: torch.Tensor) -> list[torch.Tensor]:
        return list(torch.split(action_one_hot, ACTION_DIMS, dim=-1))

    def forward(self, context: torch.Tensor,
                action_one_hot: torch.Tensor) -> torch.Tensor:
        features = self.context_encoder(context)
        segments = self._segments(action_one_hot)
        result = self.value(features).squeeze(-1)
        for segment, head in zip(segments, self.advantage_heads):
            advantage = head(features)
            advantage = advantage - advantage.mean(-1, keepdim=True)
            result = result + (advantage * segment).sum(-1)
        for (left, right), head in zip(self.pairs, self.pair_heads):
            left_size, right_size = ACTION_DIMS[left], ACTION_DIMS[right]
            interaction = head(features).reshape(
                *features.shape[:-1], left_size, right_size)
            # Double centering makes the interaction orthogonal to the
            # value and first-order action terms.
            interaction = (
                interaction - interaction.mean(-1, keepdim=True)
                - interaction.mean(-2, keepdim=True)
                + interaction.mean((-2, -1), keepdim=True))
            selected = segments[left].unsqueeze(-1) * segments[right].unsqueeze(-2)
            result = result + (interaction * selected).sum((-2, -1))
        return result

    def all_actions(self, context: torch.Tensor) -> torch.Tensor:
        features = self.context_encoder(context)
        action_segments = self._segments(self.all_action_encodings)
        result = self.value(features)
        for action_segment, head in zip(
                action_segments, self.advantage_heads):
            advantage = head(features)
            advantage = advantage - advantage.mean(-1, keepdim=True)
            result = result + advantage @ action_segment.transpose(0, 1)
        for (left, right), head in zip(self.pairs, self.pair_heads):
            left_size, right_size = ACTION_DIMS[left], ACTION_DIMS[right]
            interaction = head(features).reshape(
                *features.shape[:-1], left_size, right_size)
            interaction = (
                interaction - interaction.mean(-1, keepdim=True)
                - interaction.mean(-2, keepdim=True)
                + interaction.mean((-2, -1), keepdim=True))
            action_pairs = (
                action_segments[left].unsqueeze(-1)
                * action_segments[right].unsqueeze(-2))
            result = result + torch.einsum(
                "bij,aij->ba", interaction, action_pairs)
        return result


class JointDuelingQ(nn.Module):
    """Dueling critic with an explicit context-conditioned 90-action head."""

    def __init__(self, context_dim: int, hidden_size: int = 256) -> None:
        super().__init__()
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, hidden_size), nn.SiLU())
        self.value = nn.Linear(hidden_size, 1)
        self.joint_advantage = nn.Linear(hidden_size, JOINT_ACTIONS)
        self.register_buffer(
            "action_strides", torch.tensor((30, 6, 2, 1), dtype=torch.long))

    def all_actions(self, context: torch.Tensor) -> torch.Tensor:
        features = self.context_encoder(context)
        advantage = self.joint_advantage(features)
        advantage = advantage - advantage.mean(-1, keepdim=True)
        return self.value(features) + advantage

    def forward(self, context: torch.Tensor,
                action_one_hot: torch.Tensor) -> torch.Tensor:
        components = [
            segment.argmax(-1) for segment in torch.split(
                action_one_hot, ACTION_DIMS, dim=-1)]
        action_id = sum(
            component * self.action_strides[index]
            for index, component in enumerate(components))
        return self.all_actions(context).gather(
            -1, action_id[:, None]).squeeze(-1)


class _ResidualSiLUBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + torch.nn.functional.silu(self.linear(self.norm(value)))


class MacroOptionDuelingQ(nn.Module):
    """Dueling critic for a small legal-mask-aware tactical option set."""

    def __init__(self, context_dim: int, num_options: int = 7,
                 hidden_size: int = 256) -> None:
        super().__init__()
        if num_options < 2:
            raise ValueError("num_options must be at least two")
        self.num_options = int(num_options)
        self.input_norm = nn.LayerNorm(context_dim)
        self.input_projection = nn.Linear(context_dim, hidden_size)
        self.blocks = nn.Sequential(
            _ResidualSiLUBlock(hidden_size),
            _ResidualSiLUBlock(hidden_size))
        self.value = nn.Linear(hidden_size, 1)
        self.option_advantage = nn.Linear(hidden_size, self.num_options)

    def _features(self, context: torch.Tensor) -> torch.Tensor:
        value = torch.nn.functional.silu(
            self.input_projection(self.input_norm(context)))
        return self.blocks(value)

    def all_options(self, context: torch.Tensor,
                    legal_mask: torch.Tensor | None = None) -> torch.Tensor:
        features = self._features(context)
        advantage = self.option_advantage(features)
        if legal_mask is None:
            centered = advantage - advantage.mean(-1, keepdim=True)
        else:
            if legal_mask.shape != advantage.shape:
                raise ValueError("legal_mask must share [batch,options] shape")
            legal = legal_mask.to(device=advantage.device, dtype=torch.bool)
            count = legal.sum(-1, keepdim=True)
            if bool((count == 0).any()):
                raise ValueError("every context needs at least one legal option")
            mean = (advantage * legal).sum(-1, keepdim=True) / count
            centered = advantage - mean
        return self.value(features) + centered

    def all_actions(self, context: torch.Tensor,
                    legal_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.all_options(context, legal_mask)

    def forward(self, context: torch.Tensor, option: torch.Tensor,
                legal_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.all_options(context, legal_mask).gather(
            -1, option.long().unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def double_q_target(reward_ext: torch.Tensor, done: torch.Tensor,
                    next_context: torch.Tensor, online: ActionEmbeddingQ,
                    target: ActionEmbeddingQ, gamma: float = 0.99,
                    discount_steps: int = 1):
    """Extrinsic-only Double-DQN target; no intrinsic input is accepted."""
    if discount_steps <= 0:
        raise ValueError("discount_steps must be positive")
    online_values = online.all_actions(next_context)
    next_action = online_values.argmax(-1)
    target_values = target.all_actions(next_context)
    bootstrap = target_values.gather(-1, next_action[:, None]).squeeze(-1)
    return reward_ext + (gamma ** discount_steps) * (~done.bool()).float() * bootstrap


def q_huber_loss(model: ActionEmbeddingQ, context: torch.Tensor,
                 action_one_hot: torch.Tensor,
                 target_value: torch.Tensor) -> torch.Tensor:
    prediction = model(context, action_one_hot)
    return torch.nn.functional.smooth_l1_loss(prediction, target_value)
