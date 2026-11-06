import torch

from benchmarks.drvoi.core import action_ids_to_multi_one_hot
from benchmarks.drvoi.q_model import JointDuelingQ


def test_joint_q_explicit_forward_matches_all_actions():
    torch.manual_seed(19)
    model = JointDuelingQ(context_dim=7, hidden_size=16)
    context = torch.randn(5, 7)
    values = model.all_actions(context)
    for action_id in (0, 7, 46, 89):
        encoded = action_ids_to_multi_one_hot(
            torch.full((5,), action_id))
        assert torch.allclose(
            model(context, encoded), values[:, action_id], atol=1e-6)


def test_joint_q_advantages_are_identifiable_zero_mean():
    torch.manual_seed(23)
    model = JointDuelingQ(context_dim=4, hidden_size=12)
    context = torch.randn(3, 4)
    features = model.context_encoder(context)
    assert torch.allclose(
        model.all_actions(context).mean(-1),
        model.value(features).squeeze(-1), atol=1e-6)
