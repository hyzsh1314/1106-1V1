import torch

from benchmarks.drvoi.calibration import (
    calibrated_member_parameters,
    fit_nll_variance_scale,
    mixture_moments,
)


def test_calibrated_member_transform_scales_total_variance_exactly():
    means = torch.tensor([[[0.0], [1.0]], [[2.0], [3.0]]])
    variances = torch.ones_like(means)
    original_mean, original_variance = mixture_moments(means, variances)
    scaled_means, scaled_variances = calibrated_member_parameters(
        means, variances, torch.tensor([4.0]))
    new_mean, new_variance = mixture_moments(scaled_means, scaled_variances)
    assert torch.allclose(new_mean, original_mean)
    assert torch.allclose(new_variance, 4.0 * original_variance)


def test_nll_variance_scale_is_fit_per_dimension_and_clamped():
    means = torch.zeros(2, 4, 2)
    variances = torch.ones_like(means)
    targets = torch.tensor([[2.0, 0.0]]).expand(4, -1)
    scale = fit_nll_variance_scale(
        means, variances, targets, minimum=0.05, maximum=20.0)
    assert torch.allclose(scale, torch.tensor([4.0, 0.05]))
