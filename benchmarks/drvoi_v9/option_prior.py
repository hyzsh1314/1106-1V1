"""Confidence-gated conservative option prior for COACH-RHPPO."""

from __future__ import annotations

import torch


def confidence_gated_option_prior(
        q_a: torch.Tensor, q_b: torch.Tensor, weights: torch.Tensor,
        legal_mask: torch.Tensor, gap_temperature: float = 0.25,
        eps: float = 1.0e-6) -> dict[str, torch.Tensor]:
    """Build a soft prior without converting uncertainty into a reward.

    Args:
        q_a/q_b: ``[B,S,K]`` cross-fitted option values.
        weights: ``[B,S]`` response-candidate weights.
        legal_mask: ``[B,K]`` legal options.
    """
    if q_a.shape != q_b.shape or q_a.ndim != 3:
        raise ValueError("q_a and q_b must share [B,S,K] shape")
    if weights.shape != q_a.shape[:2]:
        raise ValueError("weights must have [B,S] shape")
    if legal_mask.shape != (q_a.shape[0], q_a.shape[2]):
        raise ValueError("legal_mask must have [B,K] shape")
    if gap_temperature <= 0.0:
        raise ValueError("gap_temperature must be positive")
    legal = legal_mask.bool()
    if bool((~legal.any(-1)).any()):
        raise ValueError("each row must contain a legal option")
    normalized_weights = weights / weights.sum(-1, keepdim=True).clamp_min(eps)
    expected_a = (q_a * normalized_weights.unsqueeze(-1)).sum(1)
    expected_b = (q_b * normalized_weights.unsqueeze(-1)).sum(1)
    conservative = torch.minimum(expected_a, expected_b)
    legal_count = legal.sum(-1).clamp_min(1)
    fold_gap = (
        (expected_a - expected_b).abs() * legal.float()
    ).sum(-1) / legal_count
    confidence = torch.exp(-fold_gap / gap_temperature).clamp(0.0, 1.0)

    legal_values = conservative.masked_fill(~legal, 0.0)
    mean = legal_values.sum(-1, keepdim=True) / legal_count.unsqueeze(-1)
    centered = (conservative - mean).masked_fill(~legal, 0.0)
    variance = centered.square().sum(-1, keepdim=True) / legal_count.unsqueeze(-1)
    normalized = centered / variance.sqrt().clamp_min(eps)
    impossible = torch.finfo(q_a.dtype).min
    prior_logits = (confidence.unsqueeze(-1) * normalized).masked_fill(
        ~legal, impossible)
    selected = prior_logits.argmax(-1)
    advantage = conservative.gather(-1, selected.unsqueeze(-1)).squeeze(-1) - (
        legal_values.sum(-1) / legal_count)
    return {
        "prior_logits": prior_logits,
        "selected_option": selected,
        "confidence": confidence,
        "fold_gap": fold_gap,
        "conservative_value": conservative,
        "option_advantage": advantage.clamp_min(0.0),
    }

