"""Cross-fitted extrinsic Q proxy for DRVOI candidate evaluation."""

from __future__ import annotations

import torch
from torch import nn

from .core import JOINT_ACTIONS, action_ids_to_multi_one_hot


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


@torch.no_grad()
def double_q_target(reward_ext: torch.Tensor, done: torch.Tensor,
                    next_context: torch.Tensor, online: ActionEmbeddingQ,
                    target: ActionEmbeddingQ, gamma: float = 0.99):
    """Extrinsic-only Double-DQN target; no intrinsic input is accepted."""
    online_values = online.all_actions(next_context)
    next_action = online_values.argmax(-1)
    target_values = target.all_actions(next_context)
    bootstrap = target_values.gather(-1, next_action[:, None]).squeeze(-1)
    return reward_ext + gamma * (~done.bool()).float() * bootstrap


def q_huber_loss(model: ActionEmbeddingQ, context: torch.Tensor,
                 action_one_hot: torch.Tensor,
                 target_value: torch.Tensor) -> torch.Tensor:
    prediction = model(context, action_one_hot)
    return torch.nn.functional.smooth_l1_loss(prediction, target_value)
