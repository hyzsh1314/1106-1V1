"""Lightweight ensemble that distils paired physics reachability rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ReachabilityPrediction:
    member_score: torch.Tensor
    member_probability: torch.Tensor
    member_time: torch.Tensor
    score_mean: torch.Tensor
    score_std: torch.Tensor
    probability_mean: torch.Tensor
    time_mean: torch.Tensor

    def lower_confidence_bound(self, z_value: float) -> torch.Tensor:
        if z_value < 0.0:
            raise ValueError("z_value must be non-negative")
        return self.score_mean - float(z_value) * self.score_std


class ReachabilityEnsembleMember(nn.Module):
    def __init__(self, feature_dim: int, hidden_size: int = 128,
                 expert_embedding_size: int = 8) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_size <= 0:
            raise ValueError("feature_dim and hidden_size must be positive")
        self.feature_dim = int(feature_dim)
        self.expert_embedding = nn.Embedding(2, expert_embedding_size)
        input_dim = self.feature_dim + expert_embedding_size
        self.backbone = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.score_head = nn.Linear(hidden_size, 1)
        self.probability_head = nn.Linear(hidden_size, 1)
        self.time_head = nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor,
                expert: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if features.ndim != 2 or features.shape[-1] != self.feature_dim:
            raise ValueError("features have the wrong shape")
        if expert.shape != features.shape[:1]:
            raise ValueError("expert must have one id per feature row")
        if bool(((expert < 0) | (expert > 1)).any()):
            raise ValueError("expert ids must be 0 or 1")
        encoded = torch.cat((
            features, self.expert_embedding(expert.long())), dim=-1)
        hidden = self.backbone(encoded)
        score = self.score_head(hidden).squeeze(-1)
        probability = torch.sigmoid(
            self.probability_head(hidden).squeeze(-1))
        time = F.softplus(self.time_head(hidden).squeeze(-1))
        return score, probability, time


class ReachabilityEnsemble(nn.Module):
    def __init__(self, feature_dim: int, members: int = 3,
                 hidden_size: int = 128,
                 expert_embedding_size: int = 8) -> None:
        super().__init__()
        if members < 2:
            raise ValueError("at least two members are required")
        self.feature_dim = int(feature_dim)
        self.members = nn.ModuleList([
            ReachabilityEnsembleMember(
                feature_dim, hidden_size=hidden_size,
                expert_embedding_size=expert_embedding_size)
            for _ in range(members)
        ])

    def forward(self, features: torch.Tensor,
                expert: torch.Tensor) -> ReachabilityPrediction:
        outputs = [member(features, expert) for member in self.members]
        scores = torch.stack([item[0] for item in outputs])
        probabilities = torch.stack([item[1] for item in outputs])
        times = torch.stack([item[2] for item in outputs])
        return ReachabilityPrediction(
            member_score=scores,
            member_probability=probabilities,
            member_time=times,
            score_mean=scores.mean(0),
            score_std=scores.std(0, unbiased=True).clamp_min(1.0e-6),
            probability_mean=probabilities.mean(0),
            time_mean=times.mean(0),
        )


def reachability_distillation_loss(
        prediction: ReachabilityPrediction,
        target_score: torch.Tensor,
        target_probability: torch.Tensor,
        target_time: torch.Tensor,
        event_observed: torch.Tensor,
        censor_horizon: float,
        score_coef: float = 1.0,
        probability_coef: float = 1.0,
        time_coef: float = 0.25,
        censored_coef: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Supervise score, soft event probability and censored time-to-event."""
    shape = prediction.score_mean.shape
    for name, value in (
            ("target_score", target_score),
            ("target_probability", target_probability),
            ("target_time", target_time),
            ("event_observed", event_observed)):
        if value.shape != shape:
            raise ValueError(f"{name} has the wrong shape")
    if censor_horizon <= 0.0:
        raise ValueError("censor_horizon must be positive")
    event = event_observed.bool()
    target_probability = target_probability.clamp(0.0, 1.0)
    target_score_members = target_score.unsqueeze(0).expand_as(
        prediction.member_score)
    target_probability_members = target_probability.unsqueeze(0).expand_as(
        prediction.member_probability)
    score_loss = F.smooth_l1_loss(
        prediction.member_score, target_score_members)
    probability_loss = F.binary_cross_entropy(
        prediction.member_probability, target_probability_members)
    if bool(event.any()):
        time_target = target_time.unsqueeze(0).expand_as(
            prediction.member_time)
        time_mask = event.unsqueeze(0).expand_as(prediction.member_time)
        time_loss = F.smooth_l1_loss(
            prediction.member_time[time_mask], time_target[time_mask])
    else:
        time_loss = prediction.member_time.sum() * 0.0
    censored = ~event
    if bool(censored.any()):
        censored_time = prediction.member_time[:, censored]
        censored_loss = F.relu(
            float(censor_horizon) - censored_time).square().mean()
    else:
        censored_loss = prediction.member_time.sum() * 0.0
    total = (
        float(score_coef) * score_loss
        + float(probability_coef) * probability_loss
        + float(time_coef) * time_loss
        + float(censored_coef) * censored_loss
    )
    return {
        "loss": total,
        "score_loss": score_loss,
        "probability_loss": probability_loss,
        "time_loss": time_loss,
        "censored_loss": censored_loss,
    }


class FrozenReachabilityPredictor(nn.Module):
    """Checkpoint-backed standardized inference for both V12 experts."""

    def __init__(self, model: ReachabilityEnsemble,
                 feature_mean: torch.Tensor, feature_scale: torch.Tensor,
                 lcb_z: float) -> None:
        super().__init__()
        if feature_mean.shape != (model.feature_dim,):
            raise ValueError("feature_mean has the wrong shape")
        if feature_scale.shape != feature_mean.shape:
            raise ValueError("feature_scale has the wrong shape")
        self.model = model.eval().requires_grad_(False)
        self.register_buffer("feature_mean", feature_mean.float())
        self.register_buffer("feature_scale", feature_scale.float())
        self.lcb_z = float(lcb_z)

    @torch.no_grad()
    def predict_both(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 2 or features.shape[-1] != self.model.feature_dim:
            raise ValueError("features have the wrong shape")
        batch = features.shape[0]
        paired_features = features.repeat_interleave(2, 0)
        expert = torch.arange(2, device=features.device).repeat(batch)
        standardized = (
            paired_features - self.feature_mean) / self.feature_scale
        prediction = self.model(standardized, expert)
        member_score = prediction.member_score.reshape(
            len(self.model.members), batch, 2)
        score_mean = prediction.score_mean.reshape(batch, 2)
        score_std = prediction.score_std.reshape(batch, 2)
        return {
            "member_score": member_score,
            "score_mean": score_mean,
            "score_std": score_std,
            "lcb": score_mean - self.lcb_z * score_std,
            "probability": prediction.probability_mean.reshape(batch, 2),
            "time": prediction.time_mean.reshape(batch, 2),
            "standardized_features": (
                features - self.feature_mean) / self.feature_scale,
        }


def load_frozen_reachability_predictor(
        checkpoint: str | Path,
        device: torch.device | str,
) -> tuple[FrozenReachabilityPredictor, dict]:
    device = torch.device(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("algorithm") != "mrat_reachability_ensemble":
        raise ValueError("checkpoint is not a V12 reachability ensemble")
    model = ReachabilityEnsemble(
        feature_dim=int(payload["feature_dim"]),
        members=int(payload["members"]),
        hidden_size=int(payload["hidden_size"])).to(device)
    model.load_state_dict(payload["model"])
    predictor = FrozenReachabilityPredictor(
        model, payload["feature_mean"].to(device),
        payload["feature_scale"].to(device),
        float(payload["selected_lcb_z"])).to(device)
    return predictor, payload
