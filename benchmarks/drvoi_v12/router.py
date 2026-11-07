"""Expert router, termination head and persistent reachability shield."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical


LEARNED_EXPERT = 0
INTERCEPT_EXPERT = 1


def conservative_router_logits(
        logits: torch.Tensor,
        learned_score: torch.Tensor,
        intercept_score: torch.Tensor,
        margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask active LOS routing unless its conservative advantage is positive."""
    if logits.ndim != 2 or logits.shape[-1] != 2:
        raise ValueError("router logits must have shape [batch,2]")
    if (learned_score.shape != logits.shape[:1]
            or intercept_score.shape != logits.shape[:1]):
        raise ValueError("router scores must have one value per row")
    allowed = (intercept_score - learned_score) >= float(margin)
    gated = logits.clone()
    gated[:, INTERCEPT_EXPERT] = torch.where(
        allowed, gated[:, INTERCEPT_EXPERT],
        torch.full_like(gated[:, INTERCEPT_EXPERT], -1.0e9))
    return gated, allowed


class MRATRouter(nn.Module):
    """Two-expert semi-Markov router with expert-conditional termination."""

    def __init__(self, input_dim: int, hidden_size: int = 128,
                 expert_embedding_size: int = 8) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_size <= 0:
            raise ValueError("input_dim and hidden_size must be positive")
        self.input_dim = int(input_dim)
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.router_head = nn.Linear(hidden_size, 2)
        self.q_head = nn.Linear(hidden_size, 2)
        self.expert_embedding = nn.Embedding(2, expert_embedding_size)
        self.termination_head = nn.Sequential(
            nn.LayerNorm(hidden_size + expert_embedding_size + 1),
            nn.Linear(hidden_size + expert_embedding_size + 1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError("router features have the wrong shape")
        hidden = self.encoder(features)
        return {
            "hidden": hidden,
            "router_logits": self.router_head(hidden),
            "q_values": self.q_head(hidden),
        }

    def distribution(self, features: torch.Tensor) -> Categorical:
        return Categorical(logits=self(features)["router_logits"])

    def termination_distribution(
            self, features: torch.Tensor, expert: torch.Tensor,
            normalized_dwell: torch.Tensor) -> Bernoulli:
        encoded = self(features)["hidden"]
        if expert.shape != features.shape[:1]:
            raise ValueError("expert has the wrong shape")
        if normalized_dwell.shape != features.shape[:1]:
            raise ValueError("normalized_dwell has the wrong shape")
        termination_input = torch.cat((
            encoded, self.expert_embedding(expert.long()),
            normalized_dwell[:, None]), dim=-1)
        return Bernoulli(logits=self.termination_head(
            termination_input).squeeze(-1))

    def state_value(self, features: torch.Tensor) -> torch.Tensor:
        outputs = self(features)
        probabilities = torch.softmax(outputs["router_logits"], dim=-1)
        return (probabilities * outputs["q_values"]).sum(-1)


@dataclass(frozen=True)
class ShieldStep:
    forced_intercept: torch.Tensor
    force_condition: torch.Tensor
    release_condition: torch.Tensor
    newly_forced: torch.Tensor
    newly_released: torch.Tensor


class PersistentReachabilityShield:
    """Sustained entry and hysteretic release using two expert LCBs."""

    def __init__(self, bad_threshold: float, safe_threshold: float,
                 intercept_advantage: float, force_steps: int = 12,
                 release_steps: int = 6) -> None:
        if safe_threshold <= bad_threshold:
            raise ValueError("safe_threshold must exceed bad_threshold")
        if intercept_advantage < 0.0:
            raise ValueError("intercept_advantage must be non-negative")
        if force_steps <= 0 or release_steps <= 0:
            raise ValueError("persistence steps must be positive")
        self.bad_threshold = float(bad_threshold)
        self.safe_threshold = float(safe_threshold)
        self.intercept_advantage = float(intercept_advantage)
        self.force_steps = int(force_steps)
        self.release_steps = int(release_steps)
        self.forced: torch.Tensor | None = None
        self.force_count: torch.Tensor | None = None
        self.release_count: torch.Tensor | None = None

    def _ensure(self, reference: torch.Tensor) -> None:
        if self.forced is None or self.forced.shape != reference.shape:
            self.forced = torch.zeros_like(reference, dtype=torch.bool)
            self.force_count = torch.zeros_like(reference, dtype=torch.long)
            self.release_count = torch.zeros_like(reference, dtype=torch.long)
        self.forced = self.forced.to(reference.device)
        self.force_count = self.force_count.to(reference.device)
        self.release_count = self.release_count.to(reference.device)

    def step(self, learned_lcb: torch.Tensor, intercept_lcb: torch.Tensor,
             reset: torch.Tensor) -> ShieldStep:
        if learned_lcb.shape != intercept_lcb.shape:
            raise ValueError("expert LCB tensors must have identical shapes")
        if reset.shape != learned_lcb.shape:
            raise ValueError("reset must have one flag per row")
        self._ensure(learned_lcb)
        reset = reset.to(learned_lcb.device).bool()
        self.forced = self.forced & ~reset
        self.force_count = torch.where(
            reset, torch.zeros_like(self.force_count), self.force_count)
        self.release_count = torch.where(
            reset, torch.zeros_like(self.release_count), self.release_count)
        force_condition = (
            (learned_lcb < self.bad_threshold)
            & ((intercept_lcb - learned_lcb) > self.intercept_advantage)
        )
        release_condition = learned_lcb > self.safe_threshold
        self.force_count = torch.where(
            ~self.forced & force_condition,
            self.force_count + 1, torch.zeros_like(self.force_count))
        newly_forced = (~self.forced) & (
            self.force_count >= self.force_steps)
        self.forced = self.forced | newly_forced
        self.release_count = torch.where(
            self.forced & release_condition,
            self.release_count + 1, torch.zeros_like(self.release_count))
        newly_released = self.forced & (
            self.release_count >= self.release_steps)
        self.forced = self.forced & ~newly_released
        self.force_count = torch.where(
            newly_released, torch.zeros_like(self.force_count),
            self.force_count)
        self.release_count = torch.where(
            self.forced, self.release_count, torch.zeros_like(self.release_count))
        return ShieldStep(
            forced_intercept=self.forced.clone(),
            force_condition=force_condition,
            release_condition=release_condition,
            newly_forced=newly_forced,
            newly_released=newly_released,
        )

    def state_dict(self) -> dict[str, torch.Tensor | None]:
        return {
            "forced": None if self.forced is None else self.forced.clone(),
            "force_count": None if self.force_count is None else self.force_count.clone(),
            "release_count": None if self.release_count is None else self.release_count.clone(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor | None]) -> None:
        self.forced = None if state["forced"] is None else state["forced"].clone()
        self.force_count = (
            None if state["force_count"] is None
            else state["force_count"].clone())
        self.release_count = (
            None if state["release_count"] is None
            else state["release_count"].clone())


@dataclass(frozen=True)
class ArbitrationStep:
    expert: torch.Tensor
    decision: torch.Tensor
    switched: torch.Tensor
    forced: torch.Tensor
    dwell: torch.Tensor


class MRATArbitrator:
    """State machine that combines learned option termination and shielding."""

    def __init__(self, minimum_dwell: int = 6,
                 maximum_dwell: int = 150) -> None:
        if minimum_dwell <= 0 or maximum_dwell < minimum_dwell:
            raise ValueError("invalid dwell limits")
        self.minimum_dwell = int(minimum_dwell)
        self.maximum_dwell = int(maximum_dwell)
        self.expert: torch.Tensor | None = None
        self.dwell: torch.Tensor | None = None
        self.was_forced: torch.Tensor | None = None

    def _ensure(self, reference: torch.Tensor) -> None:
        if self.expert is None or self.expert.shape != reference.shape:
            self.expert = torch.zeros_like(reference, dtype=torch.long)
            self.dwell = torch.zeros_like(reference, dtype=torch.long)
            self.was_forced = torch.zeros_like(reference, dtype=torch.bool)
        self.expert = self.expert.to(reference.device)
        self.dwell = self.dwell.to(reference.device)
        self.was_forced = self.was_forced.to(reference.device)

    def step(self, proposed_expert: torch.Tensor,
             termination_request: torch.Tensor,
             forced_intercept: torch.Tensor,
             reset: torch.Tensor) -> ArbitrationStep:
        shape = proposed_expert.shape
        if (termination_request.shape != shape
                or forced_intercept.shape != shape or reset.shape != shape):
            raise ValueError("arbitration inputs must have identical shapes")
        if bool(((proposed_expert < 0) | (proposed_expert > 1)).any()):
            raise ValueError("proposed expert ids must be 0 or 1")
        self._ensure(proposed_expert)
        reset = reset.to(proposed_expert.device).bool()
        forced_intercept = forced_intercept.to(
            proposed_expert.device).bool()
        released = self.was_forced & ~forced_intercept
        eligible = self.dwell >= self.minimum_dwell
        decision = (
            reset | released | (self.dwell >= self.maximum_dwell)
            | (eligible & termination_request.bool())
        )
        candidate = torch.where(
            decision, proposed_expert.long(), self.expert)
        selected = torch.where(
            forced_intercept,
            torch.full_like(candidate, INTERCEPT_EXPERT), candidate)
        switched = selected != self.expert
        # A termination followed by reselecting the same expert still starts
        # a new option instance. Its dwell must reset just like a switch;
        # otherwise an eligible termination can recur every subsequent step.
        self.dwell = torch.where(
            reset | decision | switched,
            torch.ones_like(self.dwell), self.dwell + 1)
        self.expert = selected
        self.was_forced = forced_intercept
        return ArbitrationStep(
            expert=selected.clone(),
            decision=decision,
            switched=switched,
            forced=forced_intercept,
            dwell=self.dwell.clone(),
        )

    def state_dict(self) -> dict[str, torch.Tensor | None]:
        return {
            "expert": None if self.expert is None else self.expert.clone(),
            "dwell": None if self.dwell is None else self.dwell.clone(),
            "was_forced": (
                None if self.was_forced is None else self.was_forced.clone()),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor | None]) -> None:
        self.expert = None if state["expert"] is None else state["expert"].clone()
        self.dwell = None if state["dwell"] is None else state["dwell"].clone()
        self.was_forced = (
            None if state["was_forced"] is None
            else state["was_forced"].clone())
