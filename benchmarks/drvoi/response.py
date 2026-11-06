"""Action-conditioned probabilistic response ensemble for DRVOI."""

from __future__ import annotations

import copy
import math

import torch
from torch import nn

from .core import RESPONSE_CONTINUOUS_DIM
from .calibration import calibrated_member_parameters


class ResponseMember(nn.Module):
    def __init__(self, belief_dim: int, action_dim: int = 13,
                 hidden_size: int = 256, minimum_variance: float = 1.0e-4):
        super().__init__()
        self.minimum_variance = float(minimum_variance)
        self.network = nn.Sequential(
            nn.Linear(belief_dim + action_dim, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, hidden_size), nn.SiLU())
        self.mean = nn.Linear(hidden_size, RESPONSE_CONTINUOUS_DIM)
        self.raw_variance = nn.Linear(hidden_size, RESPONSE_CONTINUOUS_DIM)
        self.side_logits = nn.Linear(hidden_size, 3)

    def forward(self, belief: torch.Tensor, action_one_hot: torch.Tensor):
        features = self.network(torch.cat((belief, action_one_hot), dim=-1))
        return (
            self.mean(features),
            self.minimum_variance + torch.nn.functional.softplus(
                self.raw_variance(features)),
            self.side_logits(features),
        )


class ResponseEnsemble(nn.Module):
    def __init__(self, belief_dim: int, members: int = 5,
                 hidden_size: int = 256, minimum_variance: float = 1.0e-4):
        super().__init__()
        self.members = nn.ModuleList([
            ResponseMember(belief_dim, hidden_size=hidden_size,
                           minimum_variance=minimum_variance)
            for _ in range(members)])

    def forward(self, belief: torch.Tensor, action_one_hot: torch.Tensor):
        outputs = [member(belief, action_one_hot) for member in self.members]
        return tuple(torch.stack(items) for items in zip(*outputs))

    def delayed_snapshot(self):
        snapshot = copy.deepcopy(self).eval()
        snapshot.requires_grad_(False)
        return snapshot

    @torch.no_grad()
    def sample_candidates(self, belief: torch.Tensor,
                          action_one_hot: torch.Tensor,
                          current_response: torch.Tensor,
                          samples_per_member: int = 4,
                          generator: torch.Generator | None = None,
                          variance_scale: torch.Tensor | None = None,
                          side_temperature: float = 1.0):
        if samples_per_member <= 0 or samples_per_member % 2 != 0:
            raise ValueError("antithetic sampling requires positive even K")
        means, variances, side_logits = self(belief, action_one_hot)
        if variance_scale is not None:
            variances = variances * variance_scale[None, None, :]
        member_count, batch, dimension = means.shape
        half = samples_per_member // 2
        epsilon = torch.randn(
            member_count, batch, half, dimension,
            device=means.device, dtype=means.dtype, generator=generator)
        epsilon = torch.cat((epsilon, -epsilon), dim=2)
        delta = means[:, :, None, :] + variances.sqrt()[:, :, None, :] * epsilon
        continuous = current_response[None, :, None, :] + delta
        side_probs = (side_logits / side_temperature).softmax(-1)
        expanded_probs = side_probs[:, :, None, :].expand(
            -1, -1, samples_per_member, -1)
        side = torch.multinomial(
            expanded_probs.reshape(-1, 3), 1, replacement=True,
            generator=generator).reshape(member_count, batch, samples_per_member)
        return {
            "continuous": continuous.permute(1, 0, 2, 3).reshape(
                batch, member_count * samples_per_member, dimension),
            "side": side.permute(1, 0, 2).reshape(
                batch, member_count * samples_per_member),
            "weights": torch.full(
                (batch, member_count * samples_per_member),
                1.0 / (member_count * samples_per_member),
                device=means.device, dtype=means.dtype),
        }


class ContinuousResponseMember(nn.Module):
    """V7 continuous-only member with bounded standard deviation."""

    def __init__(self, input_dim: int, response_dim: int,
                 action_dim: int = 13, hidden_size: int = 256,
                 sigma_min: float = 1.0e-3,
                 sigma_max: float = 10.0) -> None:
        super().__init__()
        if not 0.0 < sigma_min < sigma_max:
            raise ValueError("sigma bounds must satisfy 0 < min < max")
        self.response_dim = int(response_dim)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.network = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, hidden_size), nn.SiLU())
        self.mean = nn.Linear(hidden_size, response_dim)
        self.raw_log_sigma = nn.Linear(hidden_size, response_dim)

    def forward(self, inputs: torch.Tensor, action_one_hot: torch.Tensor):
        features = self.network(torch.cat((inputs, action_one_hot), dim=-1))
        mean = self.mean(features)
        log_min = math.log(self.sigma_min)
        log_max = math.log(self.sigma_max)
        log_sigma = log_min + (log_max - log_min) * torch.sigmoid(
            self.raw_log_sigma(features))
        return mean, torch.exp(2.0 * log_sigma)


class ContinuousResponseEnsemble(nn.Module):
    """V7 continuous response ensemble for Rel or Opp targets."""

    def __init__(self, input_dim: int, response_dim: int,
                 members: int = 5, hidden_size: int = 256,
                 sigma_min: float = 1.0e-3,
                 sigma_max: float = 10.0) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.response_dim = int(response_dim)
        self.members = nn.ModuleList([
            ContinuousResponseMember(
                input_dim, response_dim, hidden_size=hidden_size,
                sigma_min=sigma_min, sigma_max=sigma_max)
            for _ in range(members)])

    def forward(self, inputs: torch.Tensor, action_one_hot: torch.Tensor):
        outputs = [member(inputs, action_one_hot) for member in self.members]
        return tuple(torch.stack(items) for items in zip(*outputs))

    def delayed_snapshot(self):
        snapshot = copy.deepcopy(self).eval()
        snapshot.requires_grad_(False)
        return snapshot

    @torch.no_grad()
    def sample_candidates(
            self, inputs: torch.Tensor, action_one_hot: torch.Tensor,
            samples_per_member: int = 4,
            generator: torch.Generator | None = None,
            variance_scale: torch.Tensor | None = None,
            target_mean: torch.Tensor | None = None,
            target_scale: torch.Tensor | None = None,
            current_response: torch.Tensor | None = None):
        if samples_per_member <= 0 or samples_per_member % 2 != 0:
            raise ValueError("antithetic sampling requires positive even K")
        means, variances = self(inputs, action_one_hot)
        means, variances = calibrated_member_parameters(
            means, variances, variance_scale)
        member_count, batch, dimension = means.shape
        half = samples_per_member // 2
        epsilon = torch.randn(
            member_count, batch, half, dimension,
            device=means.device, dtype=means.dtype, generator=generator)
        epsilon = torch.cat((epsilon, -epsilon), dim=2)
        standardized_delta = (
            means[:, :, None, :]
            + variances.sqrt()[:, :, None, :] * epsilon)
        flat_standardized = standardized_delta.permute(1, 0, 2, 3).reshape(
            batch, member_count * samples_per_member, dimension)
        result = {
            "delta_standardized": flat_standardized,
            "weights": torch.full(
                (batch, member_count * samples_per_member),
                1.0 / (member_count * samples_per_member),
                device=means.device, dtype=means.dtype),
        }
        if (target_mean is None) != (target_scale is None):
            raise ValueError("target_mean and target_scale must be supplied together")
        delta = flat_standardized
        if target_mean is not None:
            delta = delta * target_scale + target_mean
        result["delta"] = delta
        if current_response is not None:
            result["continuous"] = current_response[:, None, :] + delta
        return result


def continuous_member_log_likelihood(means: torch.Tensor,
                                     variances: torch.Tensor,
                                     targets: torch.Tensor) -> torch.Tensor:
    residual = targets.unsqueeze(0) - means
    return -0.5 * (
        residual.square() / variances + variances.log()
        + math.log(2.0 * math.pi)).sum(-1)


def continuous_mixture_nll(ensemble: ContinuousResponseEnsemble,
                           inputs: torch.Tensor,
                           action_one_hot: torch.Tensor,
                           targets: torch.Tensor,
                           per_dimension: bool = True) -> torch.Tensor:
    means, variances = ensemble(inputs, action_one_hot)
    log_likelihood = continuous_member_log_likelihood(
        means, variances, targets)
    nll = (-torch.logsumexp(log_likelihood, dim=0)
           + math.log(log_likelihood.shape[0]))
    if per_dimension:
        nll = nll / targets.shape[-1]
    return nll


def member_log_likelihood(means: torch.Tensor, variances: torch.Tensor,
                          side_logits: torch.Tensor,
                          target_delta: torch.Tensor,
                          target_side: torch.Tensor) -> torch.Tensor:
    """Return per-member joint log likelihood with shape [M,B]."""
    residual = target_delta.unsqueeze(0) - means
    continuous = -0.5 * (
        residual.square() / variances + variances.log() + math.log(2.0 * math.pi)
    ).sum(-1)
    side_log_probs = side_logits.log_softmax(-1)
    side = side_log_probs.gather(
        -1, target_side[None, :, None].expand(side_logits.shape[0], -1, 1)
    ).squeeze(-1)
    return continuous + side


def mixture_nll(ensemble: ResponseEnsemble, belief: torch.Tensor,
                action_one_hot: torch.Tensor, target_delta: torch.Tensor,
                target_side: torch.Tensor) -> torch.Tensor:
    means, variances, side_logits = ensemble(belief, action_one_hot)
    log_likelihood = member_log_likelihood(
        means, variances, side_logits, target_delta, target_side)
    return -torch.logsumexp(log_likelihood, dim=0) + math.log(log_likelihood.shape[0])


@torch.no_grad()
def prequential_learning_progress(current: ResponseEnsemble,
                                  delayed: ResponseEnsemble,
                                  belief: torch.Tensor,
                                  action_one_hot: torch.Tensor,
                                  target_delta: torch.Tensor,
                                  target_side: torch.Tensor):
    """Score both snapshots before callers insert the current sample."""
    current_nll = mixture_nll(
        current, belief, action_one_hot, target_delta, target_side)
    delayed_nll = mixture_nll(
        delayed, belief, action_one_hot, target_delta, target_side)
    return {
        "current_nll": current_nll,
        "delayed_nll": delayed_nll,
        "lp": (delayed_nll - current_nll).clamp_min(0.0),
    }
