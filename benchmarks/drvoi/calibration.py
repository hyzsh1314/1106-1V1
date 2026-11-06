"""Shared V7 calibration used by validation and online candidate sampling."""

from __future__ import annotations

import torch


def mixture_moments(means: torch.Tensor,
                    variances: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return equal-member mixture mean and diagonal total variance."""
    if means.shape != variances.shape or means.ndim < 3:
        raise ValueError("means and variances must share member-major shape")
    mean = means.mean(0)
    variance = (variances + means.square()).mean(0) - mean.square()
    return mean, variance.clamp_min(1.0e-12)


def fit_nll_variance_scale(means: torch.Tensor, variances: torch.Tensor,
                           targets: torch.Tensor, minimum: float = 0.05,
                           maximum: float = 20.0) -> torch.Tensor:
    """Fit per-dimension Gaussian moment scale by calibration NLL.

    For fixed mean and base total variance, the NLL optimum is the average
    standardized squared residual. The result is chosen on calibration only.
    """
    mean, total_variance = mixture_moments(means, variances)
    scale = ((targets - mean).square() / total_variance).mean(0)
    return scale.clamp(minimum, maximum).detach()


def calibrated_member_parameters(
        means: torch.Tensor, variances: torch.Tensor,
        variance_scale: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale both epistemic mean spread and within-member variance.

    This guarantees that the calibrated mixture total variance is the original
    total variance multiplied by `variance_scale` (per response dimension).
    """
    if variance_scale is None:
        return means, variances
    mixture_mean = means.mean(0, keepdim=True)
    scale = variance_scale.to(device=means.device, dtype=means.dtype)
    while scale.ndim < means.ndim:
        scale = scale.unsqueeze(0)
    calibrated_means = mixture_mean + scale.sqrt() * (means - mixture_mean)
    calibrated_variances = variances * scale
    return calibrated_means, calibrated_variances


def interval_coverage(mean: torch.Tensor, variance: torch.Tensor,
                      targets: torch.Tensor, z_score: float) -> torch.Tensor:
    return ((targets - mean).abs() <= z_score * variance.sqrt()).float().mean(0)
