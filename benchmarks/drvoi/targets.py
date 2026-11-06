"""V7 response targets built only from legal WVR v4 observations."""

from __future__ import annotations

from dataclasses import dataclass

import torch


BRANCH_DIMS = {"rel": 12, "opp": 3}


def response_state_v7(observations: torch.Tensor, branch: str = "rel") -> torch.Tensor:
    """Extract the pre-registered continuous response state.

    `rel` is the 12-D relative-dynamics state. `opp` is the opponent NED
    velocity already present in legal observation indices 18:21.
    """
    if observations.shape[-1] != 39:
        raise ValueError("V7 response extraction requires v4 39-D observations")
    if branch == "opp":
        return observations[..., 18:21]
    if branch != "rel":
        raise ValueError(f"unknown V7 response branch: {branch}")
    ao = observations[..., 11]
    ta = observations[..., 12]
    return torch.stack((
        observations[..., 9], observations[..., 10],
        torch.sin(ao), torch.cos(ao), torch.sin(ta), torch.cos(ta),
        observations[..., 13], observations[..., 24], observations[..., 25],
        observations[..., 26], observations[..., 27], observations[..., 28],
    ), dim=-1)


@dataclass(frozen=True)
class DeltaTargets:
    current: torch.Tensor
    future: torch.Tensor
    delta: torch.Tensor
    valid: torch.Tensor
    horizon: int
    branch: str


def build_delta_targets(observations: torch.Tensor, dones: torch.Tensor,
                        delta: int, branch: str = "rel") -> DeltaTargets:
    """Build t→t+delta targets without crossing terminal/reset boundaries.

    Observations have shape `[T+1,...,39]`; `dones[t]` describes transition
    `t→t+1`. The returned leading dimension is `T+1-delta`.
    """
    if delta <= 0:
        raise ValueError("delta must be positive")
    if observations.shape[0] != dones.shape[0] + 1:
        raise ValueError("observations must contain one more time step than dones")
    if observations.shape[0] <= delta:
        raise ValueError("sequence is too short for requested delta")
    states = response_state_v7(observations, branch)
    count = states.shape[0] - delta
    current = states[:count]
    future = states[delta:delta + count]
    valid = torch.ones_like(dones[:count], dtype=torch.bool)
    for offset in range(delta):
        valid &= ~dones[offset:offset + count].bool()
    return DeltaTargets(
        current=current, future=future, delta=future - current,
        valid=valid, horizon=delta, branch=branch)


@dataclass
class TargetStandardizer:
    mean: torch.Tensor
    scale: torch.Tensor
    epsilon: float = 1.0e-6

    @classmethod
    def fit(cls, values: torch.Tensor, valid: torch.Tensor | None = None,
            epsilon: float = 1.0e-6) -> "TargetStandardizer":
        if valid is not None:
            selected = values[valid]
        else:
            selected = values.reshape(-1, values.shape[-1])
        if selected.numel() == 0:
            raise ValueError("cannot fit target standardizer without valid samples")
        mean = selected.mean(0)
        scale = selected.std(0, unbiased=False).clamp_min(epsilon)
        return cls(mean=mean.detach(), scale=scale.detach(), epsilon=epsilon)

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.scale.clamp_min(self.epsilon)

    def inverse(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.scale + self.mean

    def state_dict(self) -> dict[str, object]:
        return {"mean": self.mean, "scale": self.scale,
                "epsilon": float(self.epsilon)}

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "TargetStandardizer":
        return cls(mean=state["mean"], scale=state["scale"],
                   epsilon=float(state["epsilon"]))
