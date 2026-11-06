import inspect

import torch

from benchmarks.drvoi.core import action_ids_to_multi_one_hot
from benchmarks.drvoi.q_model import ActionEmbeddingQ, double_q_target


def test_action_embedding_q_scores_all_90_joint_actions():
    model = ActionEmbeddingQ(context_dim=32, hidden_size=16, action_embedding_size=8)
    values = model.all_actions(torch.randn(5, 32))
    assert values.shape == (5, 90)
    assert torch.isfinite(values).all()


def test_double_q_target_masks_terminal_bootstrap():
    online = ActionEmbeddingQ(context_dim=8, hidden_size=16, action_embedding_size=8)
    target = ActionEmbeddingQ(context_dim=8, hidden_size=16, action_embedding_size=8)
    result = double_q_target(
        torch.tensor([1.0, 2.0]), torch.tensor([True, False]),
        torch.randn(2, 8), online, target, gamma=0.9)
    assert result[0] == 1.0
    assert torch.isfinite(result[1])


def test_q_target_interface_cannot_receive_intrinsic_reward():
    parameters = inspect.signature(double_q_target).parameters
    assert "reward_int" not in parameters
    assert "intrinsic" not in parameters


def test_scalar_q_accepts_selected_action_encoding():
    model = ActionEmbeddingQ(context_dim=8, hidden_size=16, action_embedding_size=8)
    context = torch.randn(4, 8)
    action = action_ids_to_multi_one_hot(torch.tensor([0, 1, 44, 89]))
    assert model(context, action).shape == (4,)
