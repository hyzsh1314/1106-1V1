"""Pre-registered structured controls for the V7.1 Q-ready gate."""

from __future__ import annotations

import numpy as np
import torch


def pairwise_ranking_accuracy(predicted: torch.Tensor, truth: torch.Tensor,
                              minimum_gap: float = 1.0e-6) -> torch.Tensor:
    """Return one pairwise ranking accuracy per probe context."""
    if predicted.shape != truth.shape or predicted.ndim != 2:
        raise ValueError("predicted and truth must share [contexts,actions] shape")
    left, right = torch.triu_indices(truth.shape[1], truth.shape[1], offset=1)
    true_gap = truth[:, left] - truth[:, right]
    predicted_gap = predicted[:, left] - predicted[:, right]
    valid = true_gap.abs() >= minimum_gap
    correct = (true_gap * predicted_gap) > 0
    return (correct & valid).sum(-1).float() / valid.sum(-1).clamp_min(1)


def normalized_argmax_regret(predicted: torch.Tensor,
                             truth: torch.Tensor) -> torch.Tensor:
    """Regret of predicted argmax, normalized by each context's value range."""
    chosen = predicted.argmax(-1)
    chosen_truth = truth.gather(-1, chosen[:, None]).squeeze(-1)
    regret = truth.max(-1).values - chosen_truth
    value_range = truth.max(-1).values - truth.min(-1).values
    return regret / value_range.clamp_min(1.0e-8)


def bootstrap_context_mean(values: torch.Tensor, seed: int,
                           samples: int = 2000):
    values = values.detach().cpu().numpy()
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        estimates[index] = rng.choice(values, len(values), replace=True).mean()
    return float(values.mean()), np.quantile(estimates, [0.025, 0.975]).tolist()


def structured_flip_report(before: torch.Tensor, after: torch.Tensor,
                           expected_before: torch.Tensor,
                           expected_after: torch.Tensor):
    predicted_before = before.argmax(-1)
    predicted_after = after.argmax(-1)
    expected_flip = expected_before != expected_after
    correct = ((predicted_before == expected_before)
               & (predicted_after == expected_after) & expected_flip)
    return {
        "cases": int(correct.numel()),
        "correct": int(correct.sum()),
        "all_pass": bool(correct.all()),
    }


def irrelevant_perturbation_report(before: torch.Tensor, after: torch.Tensor,
                                   expected_action: torch.Tensor):
    stable = before.argmax(-1) == after.argmax(-1)
    correct = before.argmax(-1) == expected_action
    after_correct = after.argmax(-1) == expected_action
    return {
        "cases": int(stable.numel()),
        "stable": int(stable.sum()),
        "correct_before": int(correct.sum()),
        "correct_after": int(after_correct.sum()),
        "all_pass": bool((stable & correct & after_correct).all()),
        "mean_absolute_q_shift": float((before - after).abs().mean()),
    }
