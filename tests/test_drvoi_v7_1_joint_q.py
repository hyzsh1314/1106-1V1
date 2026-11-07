import torch

from benchmarks.drvoi.core import action_ids_to_multi_one_hot
from benchmarks.drvoi.q_model import JointDuelingQ, MacroOptionDuelingQ


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


def test_macro_option_q_centers_only_legal_advantages():
    torch.manual_seed(29)
    model = MacroOptionDuelingQ(context_dim=5, num_options=7, hidden_size=16)
    context = torch.randn(4, 5)
    legal = torch.ones(4, 7, dtype=torch.bool)
    legal[:, -1] = False
    features = model._features(context)
    values = model.all_options(context, legal)
    legal_mean = (values * legal).sum(-1) / legal.sum(-1)
    assert torch.allclose(
        legal_mean, model.value(features).squeeze(-1), atol=1e-6)
    selected = torch.tensor([0, 1, 2, 3])
    assert torch.allclose(
        model(context, selected, legal),
        values.gather(-1, selected[:, None]).squeeze(-1))
