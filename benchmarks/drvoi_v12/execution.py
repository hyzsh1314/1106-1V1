"""Frozen V9 and LOS-expert execution for paired MRAT branches.

The state is deliberately a flat ``str -> Tensor`` mapping so it can be
captured by :mod:`benchmarks.drvoi_v12.snapshot` without hidden Python state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import MutableMapping

import torch

from benchmarks.benchmark_torch6dof_ppo import lowlevel_controls
from benchmarks.drvoi import extract_response_state, multi_discrete_to_one_hot
from benchmarks.drvoi.response import calibrated_member_parameters
from benchmarks.drvoi_v7_1.history import VectorHistoryBuffer
from benchmarks.drvoi_v7_1.macro_options import (
    NUM_MACRO_OPTIONS,
    macro_option_actions,
    macro_option_legal_mask,
)
from benchmarks.drvoi_v9 import (
    AdaptiveOptionRecurrentActorCritic,
    confidence_gated_option_prior,
)
from benchmarks.train_torch_wvr_coach_rhppo import response_summary
from benchmarks.train_torch_wvr_ppo import fixed_mixture_actions
from benchmarks.train_torch_wvr_mods_rhppo import (
    load_history_response,
    load_option_q,
    option_values,
)
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.utils.utils import get_root_dir
from envs.torch6dof import TorchWVRVecEnv


LEARNED_EXPERT = 0
INTERCEPT_EXPERT = 1
OPPONENT_HYPOTHESES = (
    "hold",
    "pursuit",
    "evasive_left",
    "evasive_right",
    "bounded_turn_continuation",
)


def lead_intercept_expert_actions(observation: torch.Tensor) -> torch.Tensor:
    """Complete LOS expert, including fire on every legal opportunity."""
    if observation.ndim != 2 or observation.shape[-1] != 39:
        raise ValueError("observation must have [batch,39] shape")
    option = torch.full(
        (observation.shape[0],), INTERCEPT_EXPERT,
        dtype=torch.long, device=observation.device)
    action = macro_option_actions(observation, option, elapsed=1)
    action[:, 3] = (observation[:, 23] >= 0.5).long()
    return action


def opponent_hypothesis_actions(
        observation: torch.Tensor,
        hypothesis: torch.Tensor,
        episode_step: torch.Tensor | None = None,
) -> torch.Tensor:
    """Observation-only deterministic opponent hypotheses in one GPU batch."""
    if observation.ndim != 3 or observation.shape[-2:] != (2, 39):
        raise ValueError("observation must have [batch,2,39] shape")
    batch = observation.shape[0]
    if hypothesis.shape != (batch,):
        raise ValueError("hypothesis must have [batch] shape")
    hypothesis = hypothesis.to(device=observation.device, dtype=torch.long)
    if bool(((hypothesis < 0) | (hypothesis >= len(OPPONENT_HYPOTHESES))).any()):
        raise ValueError("opponent hypothesis id is out of range")
    action = torch.zeros(batch, 2, 4, dtype=torch.long,
                         device=observation.device)
    action[..., 0] = 1
    action[..., 2] = 1
    side = observation[..., 14]
    pursuit = torch.where(side >= 0.0, 4, 0)
    kind = hypothesis[:, None]
    action[..., 1] = torch.where(
        kind == 0, torch.full_like(pursuit, 2),
        torch.where(
            kind == 1, pursuit,
            torch.where(
                kind == 2, torch.zeros_like(pursuit),
                torch.where(kind == 3, torch.full_like(pursuit, 4), pursuit))))
    action[..., 2] = torch.where(
        kind == 0, torch.ones_like(action[..., 2]),
        torch.zeros_like(action[..., 2]))
    if episode_step is None:
        phase = torch.zeros(batch, dtype=torch.bool, device=observation.device)
    else:
        if episode_step.shape != (batch,):
            raise ValueError("episode_step must have [batch] shape")
        # A bounded continuation keeps its turn for 30 decisions before
        # reversing. This is deterministic and uses no privileged geometry.
        phase = torch.div(
            episode_step.to(torch.long), 30, rounding_mode="floor"
        ).remainder(2).bool()
    bounded_heading = torch.where(
        phase[:, None], torch.full_like(pursuit, 4), torch.zeros_like(pursuit))
    action[..., 1] = torch.where(kind == 4, bounded_heading, action[..., 1])
    action[..., 3] = (observation[..., 23] >= 0.5).long()
    return action


def _history_buffer_from_state(
        state: MutableMapping[str, torch.Tensor],
        num_envs: int,
        device: torch.device,
) -> VectorHistoryBuffer:
    buffer = VectorHistoryBuffer(num_envs, device=device, length=3)
    buffer.observations = state["history_observations"].clone()
    buffer.actions = state["history_actions"].clone()
    buffer.valid = state["history_valid"].clone()
    buffer._action_lag = state["history_action_lag"].clone()
    return buffer


def _store_history(
        state: MutableMapping[str, torch.Tensor],
        buffer: VectorHistoryBuffer,
) -> None:
    state["history_observations"] = buffer.observations
    state["history_actions"] = buffer.actions
    state["history_valid"] = buffer.valid
    state["history_action_lag"] = buffer._action_lag


@dataclass(frozen=True)
class V9ExecutionBundle:
    executor: "V9ExecutionAdapter"
    checkpoint_payload: dict


class V9ExecutionAdapter:
    """Execute frozen COACH-RHPPO while shadowing external expert actions."""

    def __init__(
            self,
            model: AdaptiveOptionRecurrentActorCritic,
            response: dict,
            q_folds,
            lowlevel: BaselineActor,
            option_prior_coefficient: float,
            option_commitment: int,
            samples_per_member: int = 4,
            gap_temperature: float = 0.25,
    ) -> None:
        self.model = model
        self.response = response
        self.q_folds = q_folds
        self.lowlevel = lowlevel
        self.option_prior_coefficient = float(option_prior_coefficient)
        self.option_commitment = int(option_commitment)
        self.samples_per_member = int(samples_per_member)
        self.gap_temperature = float(gap_temperature)
        if self.option_commitment <= 0:
            raise ValueError("option commitment must be positive")
        if self.samples_per_member <= 0 or self.samples_per_member % 2:
            raise ValueError(
                "samples_per_member must be positive and even for antithetic sampling")

    def initial_state(
            self,
            num_envs: int,
            device: torch.device | str,
            learner_side: torch.Tensor | None = None,
            opponent_hypothesis: torch.Tensor | None = None,
            training_opponent_kind: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        device = torch.device(device)
        if learner_side is None:
            learner_side = torch.arange(num_envs, device=device).remainder(2)
        if opponent_hypothesis is None:
            opponent_hypothesis = torch.arange(
                num_envs, device=device).remainder(3)
        history = VectorHistoryBuffer(num_envs, device=device, length=3)
        state = {
            "hidden": self.model.initial_hidden(num_envs, device=device),
            "previous_action": torch.zeros(num_envs, 13, device=device),
            "reset": torch.ones(num_envs, dtype=torch.bool, device=device),
            "active_option": torch.zeros(
                num_envs, dtype=torch.long, device=device),
            "option_age": torch.full(
                (num_envs,), self.option_commitment,
                dtype=torch.long, device=device),
            "lowlevel_hidden": torch.zeros(
                num_envs * 2, 1, 128, device=device),
            "learner_side": learner_side.to(device=device, dtype=torch.long),
            "opponent_hypothesis": opponent_hypothesis.to(
                device=device, dtype=torch.long),
            "selected_expert": torch.zeros(
                num_envs, dtype=torch.long, device=device),
            "expert_age": torch.zeros(
                num_envs, dtype=torch.long, device=device),
        }
        _store_history(state, history)
        if training_opponent_kind is not None:
            if training_opponent_kind.shape != (num_envs,):
                raise ValueError("training_opponent_kind must have [num_envs] shape")
            state["training_opponent_kind"] = training_opponent_kind.to(
                device=device, dtype=torch.long)
        return state

    def _sample_response_candidates(
            self,
            history_context: torch.Tensor,
            action_input: torch.Tensor,
            current_response: torch.Tensor,
            coupling_base_batch: int | None,
    ) -> dict[str, torch.Tensor]:
        if coupling_base_batch is None:
            return self.response["model"].sample_candidates(
                history_context,
                action_input,
                samples_per_member=self.samples_per_member,
                variance_scale=self.response["variance_scale"],
                target_mean=self.response["target_mean"],
                target_scale=self.response["target_scale"],
                current_response=current_response)
        batch = history_context.shape[0]
        if coupling_base_batch <= 0 or batch % coupling_base_batch:
            raise ValueError("invalid response common-random-number group")
        groups = batch // coupling_base_batch
        means, variances = self.response["model"](
            history_context, action_input)
        means, variances = calibrated_member_parameters(
            means, variances, self.response["variance_scale"])
        members, _, dimension = means.shape
        half = self.samples_per_member // 2
        epsilon_base = torch.randn(
            members, coupling_base_batch, half, dimension,
            device=means.device, dtype=means.dtype)
        epsilon = epsilon_base.repeat(1, groups, 1, 1)
        epsilon = torch.cat((epsilon, -epsilon), 2)
        standardized = (
            means[:, :, None, :] + variances.sqrt()[:, :, None, :] * epsilon)
        flat = standardized.permute(1, 0, 2, 3).reshape(
            batch, members * self.samples_per_member, dimension)
        delta = (
            flat * self.response["target_scale"]
            + self.response["target_mean"])
        return {
            "delta_standardized": flat,
            "weights": torch.full(
                (batch, members * self.samples_per_member),
                1.0 / (members * self.samples_per_member),
                device=means.device, dtype=means.dtype),
            "delta": delta,
            "continuous": current_response[:, None, :] + delta,
        }

    @torch.no_grad()
    def prepare_step(
            self,
            env: TorchWVRVecEnv,
            state: MutableMapping[str, torch.Tensor],
            active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, MutableMapping[str, torch.Tensor]]:
        del active  # terminal rows are masked by the branch teacher itself
        num_envs = env.config.num_envs
        env_index = torch.arange(num_envs, device=env.device)
        learner_side = state["learner_side"].long()
        observation = env.observations()
        learner_observation = observation[env_index, learner_side]
        belief, next_hidden = self.model.belief_step(
            learner_observation, state["previous_action"], state["hidden"],
            state["reset"])
        provisional = self.model.conditioned_distributions(
            belief, state["active_option"], learner_observation)
        provisional_action = torch.stack([
            distribution.probs.argmax(-1) for distribution in provisional
        ], -1)
        history = _history_buffer_from_state(state, num_envs, env.device)
        preview = history.preview(
            learner_observation, provisional_action, reset=state["reset"])
        current_response, _ = extract_response_state(learner_observation)
        coupling = state.get("response_coupling_base_batch")
        coupling_base_batch = None if coupling is None else int(coupling.item())
        candidates = self._sample_response_candidates(
            preview.context, preview.action_input, current_response,
            coupling_base_batch)
        response_features = response_summary(candidates)
        samples = candidates["continuous"].shape[1]
        provisional_one_hot = multi_discrete_to_one_hot(provisional_action)
        raw_context = torch.cat((
            belief[:, None, :].expand(-1, samples, -1),
            provisional_one_hot[:, None, :].expand(-1, samples, -1),
            candidates["continuous"],
        ), -1)
        legal_options = macro_option_legal_mask(learner_observation)
        flat_legal = legal_options[:, None, :].expand(
            -1, samples, -1).reshape(-1, NUM_MACRO_OPTIONS)
        flat_context = raw_context.reshape(-1, raw_context.shape[-1])
        q_a = option_values(
            self.q_folds[0], flat_context, flat_legal
        ).reshape(num_envs, samples, NUM_MACRO_OPTIONS)
        q_b = option_values(
            self.q_folds[1], flat_context, flat_legal
        ).reshape(num_envs, samples, NUM_MACRO_OPTIONS)
        prior = confidence_gated_option_prior(
            q_a, q_b, candidates["weights"], legal_options,
            gap_temperature=self.gap_temperature)
        option_decision = state["reset"] | (
            state["option_age"] >= self.option_commitment)
        option_logits = self.model.option_logits(
            belief, response_features, legal_options,
            prior_logits=prior["prior_logits"],
            prior_coefficient=self.option_prior_coefficient)
        proposed = option_logits.argmax(-1)
        active_option = torch.where(
            option_decision, proposed, state["active_option"])
        distributions = self.model.conditioned_distributions(
            belief, active_option, learner_observation)
        learned_action = torch.stack([
            distribution.probs.argmax(-1) for distribution in distributions
        ], -1)
        intercept_action = lead_intercept_expert_actions(learner_observation)
        selected_expert = state["selected_expert"].long()
        learner_action = torch.where(
            (selected_expert == INTERCEPT_EXPERT)[:, None],
            intercept_action, learned_action)

        # The frozen V9 state follows the action actually executed. This keeps
        # its hidden/history state valid when control later returns from LOS.
        history.append(
            learner_observation, learner_action, reset=state["reset"])
        if "training_opponent_kind" in state:
            executed = fixed_mixture_actions(
                observation, state["training_opponent_kind"])
        else:
            executed = opponent_hypothesis_actions(
                observation, state["opponent_hypothesis"], env.episode_step)
        executed[env_index, learner_side] = learner_action
        controls, next_lowlevel_hidden = lowlevel_controls(
            self.lowlevel, state["lowlevel_hidden"], observation, executed,
            torch_speed_adapter=True)

        updated = dict(state)
        updated["hidden"] = next_hidden
        updated["active_option"] = active_option
        updated["pending_option_decision"] = option_decision
        updated["pending_previous_action"] = multi_discrete_to_one_hot(
            learner_action)
        updated["pending_lowlevel_hidden"] = next_lowlevel_hidden
        updated["last_belief"] = belief
        updated["last_response_features"] = response_features
        updated["last_prior_confidence"] = prior["confidence"]
        updated["last_learned_action"] = learned_action
        updated["last_intercept_action"] = intercept_action
        updated["last_executed_action"] = learner_action
        _store_history(updated, history)
        return (
            controls.reshape(num_envs, 2, 4),
            executed[..., 3] == 1,
            updated,
        )

    @torch.no_grad()
    def post_step(
            self,
            env: TorchWVRVecEnv,
            state: MutableMapping[str, torch.Tensor],
            done: torch.Tensor,
            info: dict[str, torch.Tensor],
    ) -> MutableMapping[str, torch.Tensor]:
        del env, info
        updated = dict(state)
        pending_action = state["pending_previous_action"]
        updated["previous_action"] = torch.where(
            done[:, None], torch.zeros_like(pending_action), pending_action)
        age = torch.where(
            state["pending_option_decision"],
            torch.zeros_like(state["option_age"]), state["option_age"])
        updated["option_age"] = torch.where(
            done, torch.zeros_like(age), age + 1)
        updated["active_option"] = torch.where(
            done, torch.zeros_like(state["active_option"]),
            state["active_option"])
        next_lowlevel = state["pending_lowlevel_hidden"]
        updated["lowlevel_hidden"] = torch.where(
            done.repeat_interleave(2)[:, None, None],
            torch.zeros_like(next_lowlevel), next_lowlevel)
        updated["reset"] = done
        updated["expert_age"] = torch.where(
            done, torch.zeros_like(state["expert_age"]),
            state["expert_age"] + 1)
        return updated


def load_frozen_v9_execution(
        checkpoint: str | Path,
        response_checkpoint: str | Path,
        option_q_checkpoint: str | Path,
        device: torch.device | str,
        samples_per_member: int = 4,
        gap_temperature: float = 0.25,
) -> V9ExecutionBundle:
    """Load the exact V9 policy dependencies used by the published evaluator."""
    device = torch.device(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("algorithm") != "coach_rhppo":
        raise ValueError("checkpoint is not a COACH-RHPPO policy")
    model = AdaptiveOptionRecurrentActorCritic(
        obs_dim=int(payload["obs_dim"]),
        hidden_size=int(payload["hidden_size"]),
        response_feature_dim=int(payload["response_feature_dim"]),
        num_options=int(payload["num_options"]),
        option_embedding_size=int(payload["option_embedding_size"]),
    ).to(device).eval()
    model.load_state_dict(payload["model"])
    model.requires_grad_(False)
    response = load_history_response(
        Path(response_checkpoint), int(payload["response_fold"]), device)
    _, q_folds = load_option_q(Path(option_q_checkpoint), device)
    lowlevel = BaselineActor().to(device).eval()
    lowlevel.load_state_dict(torch.load(
        get_root_dir() + "/model/baseline_model.pt",
        map_location=device, weights_only=True))
    lowlevel.requires_grad_(False)
    lowlevel.tpdv = {"dtype": torch.float32, "device": device}
    history = payload.get("history", [])
    if not history or "option_prior_coefficient" not in history[-1]:
        raise ValueError("V9 checkpoint lacks its final prior coefficient")
    executor = V9ExecutionAdapter(
        model=model,
        response=response,
        q_folds=q_folds,
        lowlevel=lowlevel,
        option_prior_coefficient=float(
            history[-1]["option_prior_coefficient"]),
        option_commitment=int(payload["option_commitment"]),
        samples_per_member=samples_per_member,
        gap_temperature=gap_temperature,
    )
    return V9ExecutionBundle(executor=executor, checkpoint_payload=payload)
