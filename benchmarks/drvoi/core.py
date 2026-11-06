"""Causally constrained DRVOI estimators for the Torch WVR platform."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch


ACTION_DIMS = (3, 5, 3, 2)
JOINT_ACTIONS = 90
RESPONSE_CONTINUOUS_DIM = 12


def action_ids_to_multi_one_hot(action_ids: torch.Tensor) -> torch.Tensor:
    """Convert flattened joint action IDs to the four-head 13-D encoding."""
    ids = action_ids.long()
    if bool(((ids < 0) | (ids >= JOINT_ACTIONS)).any()):
        raise ValueError("joint action id is outside [0, 90)")
    components = []
    remainder = ids
    for index, size in enumerate(ACTION_DIMS):
        stride = 1
        for later in ACTION_DIMS[index + 1:]:
            stride *= later
        component = torch.div(remainder, stride, rounding_mode="floor")
        remainder = remainder.remainder(stride)
        components.append(torch.nn.functional.one_hot(
            component, num_classes=size).to(torch.float32))
    return torch.cat(components, dim=-1)


def multi_discrete_to_one_hot(actions: torch.Tensor) -> torch.Tensor:
    """Encode explicit `[...,4]` action components as 13-D multi-one-hot."""
    if actions.shape[-1] != len(ACTION_DIMS):
        raise ValueError("actions must have four components")
    encoded = []
    for index, size in enumerate(ACTION_DIMS):
        component = actions[..., index].long()
        if bool(((component < 0) | (component >= size)).any()):
            raise ValueError(f"action component {index} is outside its range")
        encoded.append(torch.nn.functional.one_hot(
            component, num_classes=size).to(torch.float32))
    return torch.cat(encoded, dim=-1)


def extract_response_state(observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the legal 12-D continuous response and three-class side target."""
    if observations.shape[-1] != 39:
        raise ValueError("DRVOI response extraction requires v4 39-D observations")
    ao = observations[..., 11]
    ta = observations[..., 12]
    continuous = torch.stack((
        observations[..., 9], observations[..., 10],
        torch.sin(ao), torch.cos(ao), torch.sin(ta), torch.cos(ta),
        observations[..., 13], observations[..., 24], observations[..., 25],
        observations[..., 26], observations[..., 27], observations[..., 28],
    ), dim=-1)
    side = observations[..., 14]
    side_class = torch.where(side < -0.5, 0, torch.where(side > 0.5, 2, 1))
    return continuous, side_class.long()


def cross_fitted_voi(q_a: torch.Tensor, q_b: torch.Tensor,
                     weights: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute signed A→B/B→A VOI without sharing selection noise.

    Args:
        q_a/q_b: `[batch, candidates, actions]` values from isolated folds.
        weights: `[batch, candidates]` normalized candidate probabilities.
    """
    if q_a.shape != q_b.shape or q_a.ndim != 3:
        raise ValueError("cross-fitted Q tensors must share [B,S,G] shape")
    if weights.shape != q_a.shape[:2]:
        raise ValueError("candidate weights must have [B,S] shape")
    normalized = weights / weights.sum(-1, keepdim=True).clamp_min(1.0e-12)

    def direction(select: torch.Tensor, evaluate: torch.Tensor):
        per_candidate_action = select.argmax(-1, keepdim=True)
        chosen_values = evaluate.gather(-1, per_candidate_action).squeeze(-1)
        t1 = (normalized * chosen_values).sum(-1)
        common_action = (
            normalized.unsqueeze(-1) * select).sum(1).argmax(-1)
        common_index = common_action[:, None, None].expand(
            -1, evaluate.shape[1], 1)
        common_values = evaluate.gather(-1, common_index).squeeze(-1)
        t2 = (normalized * common_values).sum(-1)
        return t1, t2, t1 - t2

    t1_ab, t2_ab, value_ab = direction(q_a, q_b)
    t1_ba, t2_ba, value_ba = direction(q_b, q_a)
    signed = 0.5 * (value_ab + value_ba)
    return {
        "t1_ab": t1_ab, "t2_ab": t2_ab, "value_ab": value_ab,
        "t1_ba": t1_ba, "t2_ba": t2_ba, "value_ba": value_ba,
        "signed_voi": signed, "positive_voi": signed.clamp_min(0.0),
    }


def ensemble_mi(means: torch.Tensor, variances: torch.Tensor,
                side_probabilities: torch.Tensor,
                epsilon: float = 1.0e-8) -> dict[str, torch.Tensor]:
    """V6 ENS-MI: continuous moment upper-bound proxy plus categorical BALD.

    Inputs have member-major shapes `[M,B,D]`, `[M,B,D]`, and `[M,B,3]`.
    """
    if means.shape != variances.shape or means.ndim != 3:
        raise ValueError("ensemble means and variances must share [M,B,D] shape")
    if side_probabilities.ndim != 3 or side_probabilities.shape[:2] != means.shape[:2]:
        raise ValueError("side probabilities must have [M,B,3] shape")
    aleatoric = variances.mean(0).clamp_min(epsilon)
    epistemic = means.var(0, unbiased=False)
    continuous = 0.5 * torch.log1p(epistemic / aleatoric).sum(-1)
    probs = side_probabilities.clamp_min(epsilon)
    mixture = probs.mean(0)
    mixture_entropy = -(mixture * mixture.log()).sum(-1)
    member_entropy = -(probs * probs.log()).sum(-1).mean(0)
    side = (mixture_entropy - member_entropy).clamp_min(0.0)
    return {"continuous": continuous, "side": side, "total": continuous + side}


@dataclass(frozen=True)
class QTransition:
    episode_id: int
    fold: int
    context: torch.Tensor
    next_action: torch.Tensor
    next_reward_ext: torch.Tensor
    next_context: torch.Tensor | None
    done: bool


class CrossFittedQTransitionAssembler:
    """Delay one record so Q(x_t,a_{t+1}) receives r_ext,t+1 and x_t+1."""

    def __init__(self) -> None:
        self._pending: dict[int, tuple[int, torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def fold_for_episode(episode_id: int) -> int:
        return int(episode_id) & 1

    def observe_context(self, env_id: int, episode_id: int,
                        context: torch.Tensor, next_action: torch.Tensor,
                        next_reward_ext: torch.Tensor,
                        done: bool) -> QTransition | None:
        """Finish the preceding tuple, then cache the current incomplete tuple."""
        completed = None
        if env_id in self._pending:
            previous_episode, previous_context, previous_action = self._pending[env_id]
            if previous_episode != episode_id:
                raise ValueError("episode changed before terminal Q tuple was closed")
            completed = QTransition(
                episode_id=episode_id,
                fold=self.fold_for_episode(episode_id),
                context=previous_context,
                next_action=previous_action,
                next_reward_ext=next_reward_ext,
                next_context=None if done else context,
                done=bool(done),
            )
        if done:
            self._pending.pop(env_id, None)
        else:
            self._pending[env_id] = (episode_id, context, next_action)
        return completed

    def reset_env(self, env_id: int) -> None:
        self._pending.pop(env_id, None)


@dataclass(frozen=True)
class CrossDeltaQTransition:
    episode_id: int
    fold: int
    start_step: int
    delta: int
    context: torch.Tensor
    evaluated_action: torch.Tensor
    discounted_return_ext: torch.Tensor
    next_context: torch.Tensor | None
    done: bool


@dataclass
class _DeltaRecord:
    step: int
    belief: torch.Tensor
    action: torch.Tensor
    response: torch.Tensor
    reward_ext: torch.Tensor
    done: bool


class CrossDeltaQTransitionAssembler:
    """Assemble V7 Q tuples with response horizon and Q stride `delta`.

    Each observed record is the pre-action state at decision step `t`, the
    action at `t`, and the extrinsic reward/done returned by that action. A
    non-terminal tuple starting at `t` becomes available when state `t+2Δ`
    is observed. A terminal tuple is retained only when the full Δ-step return
    window has completed exactly at the terminal transition.
    """

    def __init__(self, delta: int, gamma: float = 0.99) -> None:
        if delta <= 0:
            raise ValueError("delta must be positive")
        self.delta = int(delta)
        self.gamma = float(gamma)
        self._records: dict[int, list[_DeltaRecord]] = {}
        self._episodes: dict[int, int] = {}
        self._next_base: dict[int, int] = {}
        self.invalid_counts = {
            "response_incomplete": 0,
            "return_incomplete": 0,
            "episode_changed": 0,
        }

    @staticmethod
    def fold_for_episode(episode_id: int) -> int:
        return int(episode_id) & 1

    @staticmethod
    def _context(record: _DeltaRecord,
                 future_response: torch.Tensor) -> torch.Tensor:
        return torch.cat((record.belief, record.action, future_response), -1)

    def _make_transition(self, records: list[_DeltaRecord], base: int,
                         episode_id: int, terminal: bool) -> CrossDeltaQTransition:
        delta = self.delta
        current = records[base]
        decision = records[base + delta]
        rewards = torch.stack([
            records[base + delta + offset].reward_ext
            * (self.gamma ** offset) for offset in range(delta)])
        discounted_return = rewards.sum(0)
        context = self._context(current, decision.response)
        if terminal:
            next_context = None
        else:
            next_record = records[base + delta]
            next_context = self._context(
                next_record, records[base + 2 * delta].response)
        return CrossDeltaQTransition(
            episode_id=episode_id,
            fold=self.fold_for_episode(episode_id),
            start_step=current.step,
            delta=delta,
            context=context,
            evaluated_action=decision.action,
            discounted_return_ext=discounted_return,
            next_context=next_context,
            done=terminal,
        )

    def observe_step(self, env_id: int, episode_id: int, step: int,
                     belief: torch.Tensor, action: torch.Tensor,
                     response: torch.Tensor, reward_ext: torch.Tensor,
                     done: bool) -> list[CrossDeltaQTransition]:
        """Observe one sequential decision record and emit completed tuples."""
        if env_id in self._episodes and self._episodes[env_id] != episode_id:
            self.invalid_counts["episode_changed"] += max(
                len(self._records.get(env_id, []))
                - self._next_base.get(env_id, 0), 0)
            self.reset_env(env_id)
        if env_id not in self._records:
            self._records[env_id] = []
            self._episodes[env_id] = int(episode_id)
            self._next_base[env_id] = 0
        records = self._records[env_id]
        if records and step != records[-1].step + 1:
            raise ValueError("cross-delta records must have consecutive step indices")
        records.append(_DeltaRecord(
            step=int(step), belief=belief.detach(), action=action.detach(),
            response=response.detach(), reward_ext=reward_ext.detach(),
            done=bool(done)))
        emitted: list[CrossDeltaQTransition] = []
        current_index = len(records) - 1
        # Current pre-action state completes a non-terminal next context for
        # base=current_index-2Δ. Its return ended on the preceding record.
        nonterminal_base = current_index - 2 * self.delta
        if nonterminal_base >= self._next_base[env_id]:
            if nonterminal_base != self._next_base[env_id]:
                raise RuntimeError("cross-delta base sequence became misaligned")
            emitted.append(self._make_transition(
                records, nonterminal_base, episode_id, terminal=False))
            self._next_base[env_id] += self.delta
        if done:
            # A complete terminal return ends at base+2Δ-1.
            terminal_base = current_index - (2 * self.delta - 1)
            if terminal_base >= 0 and terminal_base == self._next_base[env_id]:
                emitted.append(self._make_transition(
                    records, terminal_base, episode_id, terminal=True))
                self._next_base[env_id] += self.delta
            for base in range(
                    self._next_base[env_id], current_index + 1, self.delta):
                if base + self.delta > current_index:
                    self.invalid_counts["response_incomplete"] += 1
                else:
                    self.invalid_counts["return_incomplete"] += 1
            self.reset_env(env_id)
        return emitted

    def reset_env(self, env_id: int) -> None:
        self._records.pop(env_id, None)
        self._episodes.pop(env_id, None)
        self._next_base.pop(env_id, None)

    def state_dict(self) -> dict[str, object]:
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "records": copy.deepcopy(self._records),
            "episodes": copy.deepcopy(self._episodes),
            "next_base": copy.deepcopy(self._next_base),
            "invalid_counts": dict(self.invalid_counts),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if int(state["delta"]) != self.delta:
            raise ValueError("cannot restore assembler with different delta")
        if float(state["gamma"]) != self.gamma:
            raise ValueError("cannot restore assembler with different gamma")
        self._records = copy.deepcopy(state["records"])
        self._episodes = copy.deepcopy(state["episodes"])
        self._next_base = copy.deepcopy(state["next_base"])
        self.invalid_counts = dict(state["invalid_counts"])
