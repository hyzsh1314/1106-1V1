import torch

from benchmarks.drvoi.core import action_ids_to_multi_one_hot
from benchmarks.drvoi.q_model import FactorizedDuelingQ


def test_factorized_q_all_actions_matches_explicit_forward():
    torch.manual_seed(4)
    model = FactorizedDuelingQ(context_dim=7, hidden_size=16)
    context = torch.randn(3, 7)
    all_values = model.all_actions(context)
    for action_id in (0, 17, 52, 89):
        action = action_ids_to_multi_one_hot(
            torch.full((3,), action_id))
        assert torch.allclose(
            all_values[:, action_id], model(context, action), atol=1e-6)


def test_factorized_q_has_zero_mean_action_advantage():
    torch.manual_seed(8)
    model = FactorizedDuelingQ(context_dim=5, hidden_size=12)
    context = torch.randn(4, 5)
    values = model.all_actions(context)
    features = model.context_encoder(context)
    baseline = model.value(features).squeeze(-1)
    assert torch.allclose(values.mean(-1), baseline, atol=1e-6)
