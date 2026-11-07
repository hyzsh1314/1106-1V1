import torch

from benchmarks.collect_drvoi_v8_macro_option_data import (
    aggregate_rollouts,
    bootstrapped_return,
)
from benchmarks.validate_drvoi_v8_macro_option_value import (
    macro_option_loss,
    option_metrics,
)


def test_bootstrap_masks_terminal_and_uses_gamma_power_horizon():
    result = bootstrapped_return(
        torch.tensor([1.0, 2.0]), torch.tensor([4.0, 9.0]),
        torch.tensor([True, False]), elapsed=2, gamma=0.5)
    assert torch.allclose(result, torch.tensor([2.0, 2.0]))


def test_rollout_aggregation_averages_only_continuation_replicates():
    value = torch.arange(24, dtype=torch.float32)
    result = aggregate_rollouts(value, contexts=2, options=3, samples=4)
    assert result.shape == (2, 3)
    assert torch.allclose(result[0], torch.tensor([1.5, 5.5, 9.5]))


def test_macro_loss_and_metrics_ignore_illegal_launch_option():
    truth = torch.tensor([[3.0, 2.0, 1.0, 0.0, -1.0, -2.0, 100.0]])
    prediction = truth.clone()
    prediction[:, -1] = 999.0
    legal = torch.tensor([[True, True, True, True, True, True, False]])
    cfg = {
        "raw_value_loss_weight": 1.0,
        "centered_advantage_loss_weight": 2.0,
        "soft_rank_loss_weight": 0.5,
        "soft_rank_temperature": 0.1,
    }
    loss, _ = macro_option_loss(prediction, truth, legal, cfg)
    assert torch.isfinite(loss)
    metrics = option_metrics(prediction, truth, legal, {
        "minimum_value_range": 0.01,
        "minimum_pair_gap_fraction": 0.05,
        "bootstrap_samples": 50,
    }, seed=3)
    assert metrics["pairwise_accuracy"] == 1.0
    assert metrics["normalized_argmax_regret"] == 0.0
    assert metrics["weighted_normalized_argmax_regret"] == 0.0
