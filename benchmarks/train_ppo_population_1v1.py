"""Stratified historical-self-play PPO baseline for 1v1 missile combat.

The learner controls the first aircraft.  The second aircraft is controlled
by frozen historical policies.  Opponent sampling always covers the founding
anchor and latest snapshot, then fills the remaining slots with a recency-
weighted archive sample.  This is an auditable baseline, not PSRO: claims of
population-game convergence still require payoff-matrix evaluation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from functools import partial
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from benchmarks.train_ppo_1v1_shoot_smoke import (
    DEFAULT_REWARD_VARIANT,
    MultiDiscreteActorCritic,
    Rollout,
    make_environment,
    parameter_vector,
    ppo_update,
    select_device,
    REWARD_VARIANT_OVERRIDES,
    SCENARIO,
)
from benchmarks.evaluate_ppo_1v1_acceptance import (
    OPPONENT_POLICIES,
    scripted_evasive_fire_break_skill,
)
from envs.env_wrappers import DummyVecEnv, SubprocVecEnv


@dataclass(frozen=True)
class PolicySnapshot:
    snapshot_id: int
    source_update: int
    state_dict: dict[str, torch.Tensor]
    parameter_vector: torch.Tensor


class HistoricalPolicyPopulation:
    """Bounded, immutable CPU policy snapshots used as opponents."""

    def __init__(self, capacity: int):
        if capacity < 2:
            raise ValueError("population capacity must be at least two")
        self.capacity = int(capacity)
        self.snapshots: list[PolicySnapshot] = []
        self._next_snapshot_id = 0

    def add(self, model: MultiDiscreteActorCritic, source_update: int) -> int:
        state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }
        snapshot = PolicySnapshot(
            snapshot_id=self._next_snapshot_id,
            source_update=int(source_update),
            state_dict=state,
            parameter_vector=parameter_vector(model).clone(),
        )
        self._next_snapshot_id += 1
        self.snapshots.append(snapshot)
        if len(self.snapshots) > self.capacity:
            # Retain the immutable founding anchor and replace the oldest
            # non-anchor snapshot.  This prevents FIFO drift from silently
            # deleting the only fixed training reference.
            self.snapshots.pop(1)
        return snapshot.snapshot_id

    def sample_indices(self, generator: np.random.Generator,
                       count: int, max_unique: int = 0) -> np.ndarray:
        if not self.snapshots:
            raise RuntimeError("cannot sample an empty policy population")
        count = int(count)
        population_size = len(self.snapshots)
        unique_count = population_size
        if max_unique > 0:
            unique_count = min(unique_count, int(max_unique))
        if unique_count >= population_size:
            active = np.arange(population_size, dtype=np.int64)
        else:
            mandatory = [population_size - 1]
            if unique_count >= 2:
                mandatory.insert(0, 0)
            remaining = unique_count - len(mandatory)
            candidates = np.asarray([
                index for index in range(population_size)
                if index not in mandatory
            ], dtype=np.int64)
            if remaining > 0:
                # Linear recency weights retain archive diversity without
                # treating every stale policy as equally informative.
                weights = (candidates.astype(np.float64) + 1.0)
                weights /= weights.sum()
                sampled = generator.choice(
                    candidates, size=remaining, replace=False, p=weights)
                mandatory.extend(int(value) for value in sampled)
            active = np.asarray(sorted(mandatory), dtype=np.int64)
        full_cycles, remainder = divmod(count, unique_count)
        indices = np.tile(
            active, full_cycles)
        if remainder:
            extra = generator.choice(active, size=remainder, replace=False)
            indices = np.concatenate((indices, extra.astype(np.int64)))
        generator.shuffle(indices)
        return indices

    def materialize(self, index: int, obs_dim: int, action_dims: list[int],
                    hidden_size: int, device: torch.device,
                    fire_mask_obs_index: int | None = None,
                    altitude_safety_obs_index: int | None = None,
                    altitude_descend_mask_threshold: float = 0.8,
                    altitude_force_climb_threshold: float = 0.65):
        snapshot = self.snapshots[int(index)]
        model = MultiDiscreteActorCritic(
            obs_dim=obs_dim, action_dims=action_dims,
            hidden_size=hidden_size,
            fire_mask_obs_index=fire_mask_obs_index,
            altitude_safety_obs_index=altitude_safety_obs_index,
            altitude_descend_mask_threshold=altitude_descend_mask_threshold,
            altitude_force_climb_threshold=altitude_force_climb_threshold,
        ).to(device)
        model.load_state_dict(snapshot.state_dict)
        model.eval()
        model.requires_grad_(False)
        return model

    def state_dict(self):
        return {
            "capacity": self.capacity,
            "next_snapshot_id": self._next_snapshot_id,
            "snapshots": [
                {
                    "snapshot_id": snapshot.snapshot_id,
                    "source_update": snapshot.source_update,
                    "state_dict": snapshot.state_dict,
                    "parameter_vector": snapshot.parameter_vector,
                }
                for snapshot in self.snapshots
            ],
        }

    def load_state_dict(self, payload):
        if int(payload["capacity"]) != self.capacity:
            raise ValueError("checkpoint population capacity does not match")
        self._next_snapshot_id = int(payload["next_snapshot_id"])
        self.snapshots = [
            PolicySnapshot(
                snapshot_id=int(item["snapshot_id"]),
                source_update=int(item["source_update"]),
                state_dict={
                    name: tensor.detach().cpu().clone()
                    for name, tensor in item["state_dict"].items()
                },
                parameter_vector=item["parameter_vector"].detach().cpu().clone(),
            )
            for item in payload["snapshots"]
        ]
        if not self.snapshots:
            raise ValueError("checkpoint population is empty")


@dataclass
class PopulationStats:
    learner_launches: int
    opponent_launches: int
    wins: int
    draws: int
    losses: int
    learner_weapon_events: dict[str, int] = field(default_factory=dict)
    opponent_weapon_events: dict[str, int] = field(default_factory=dict)
    fire_requests: int = 0
    rejected_requests: int = 0
    legal_opportunities: int = 0
    timeouts: int = 0
    tactical_timeout_decisions: int = 0
    termination_reasons: dict[str, int] = field(default_factory=dict)
    learner_termination_conditions: dict[str, int] = field(
        default_factory=dict)
    mean_reward_components: dict[str, float] = field(default_factory=dict)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--start-method", choices=("spawn", "forkserver"),
                        default="spawn")
    parser.add_argument(
        "--envs-per-worker", type=int, default=1,
        help=("number of independent environments stepped serially by each "
              "worker process; this changes only process topology"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=1.0e-3)
    parser.add_argument("--max-grad-norm", type=float, default=2.0)
    parser.add_argument("--scenario", default=SCENARIO)
    parser.add_argument(
        "--reward-variant", choices=tuple(REWARD_VARIANT_OVERRIDES),
        default=DEFAULT_REWARD_VARIANT)
    parser.add_argument("--population-capacity", type=int, default=16)
    parser.add_argument("--snapshot-interval", type=int, default=1)
    parser.add_argument("--max-opponents-per-update", type=int, default=0)
    parser.add_argument(
        "--scripted-anchor-fraction", type=float, default=0.0,
        help="fraction of environments reserved for fixed scripted anchors")
    parser.add_argument(
        "--scripted-anchor-final-fraction", type=float,
        help=("optional final anchor fraction; linearly interpolated from "
              "--scripted-anchor-fraction over the configured decay updates"))
    parser.add_argument(
        "--scripted-anchor-decay-updates", type=int, default=0,
        help="number of updates used for linear anchor-fraction decay")
    parser.add_argument(
        "--scripted-anchor-ramp-start-update", type=int, default=1,
        help=("global PPO update at which anchor interpolation starts; "
              "set this to the first resumed update for a continuation curriculum"))
    parser.add_argument(
        "--scripted-anchor-policies", nargs="+",
        choices=tuple(OPPONENT_POLICIES),
        default=("pursuit_fire", "evasive_fire", "standoff_fire"))
    parser.add_argument(
        "--altitude-safety-mask", action=argparse.BooleanOptionalAction,
        default=False,
        help=("mask descend below the safe altitude and force climb below "
              "the critical altitude in the PPO distribution"))
    parser.add_argument(
        "--altitude-descend-mask-m", type=float, default=4000.0)
    parser.add_argument(
        "--altitude-force-climb-m", type=float, default=3250.0)
    parser.add_argument(
        "--threat-observation-v2", action=argparse.BooleanOptionalAction,
        default=False,
        help=("append an explicit missile-warning flag and time-to-go "
              "urgency to the legacy 24-D observation"))
    parser.add_argument(
        "--heading-command-span-deg", type=float, default=30.0,
        help=("maximum absolute high-level heading command; 30 reproduces "
              "the LAG-style baseline and 60/90 test wider evasive turns"))
    parser.add_argument(
        "--threat-relative-break-skill",
        action=argparse.BooleanOptionalAction, default=False,
        help=("under missile warning interpret extreme heading actions as "
              "threat-relative break-away/cut-toward skills"))
    parser.add_argument(
        "--threat-break-option",
        action=argparse.BooleanOptionalAction, default=False,
        help=("make break-away stateful with minimum duration, warning-clear "
              "hysteresis, and opponent reacquisition"))
    parser.add_argument("--threat-break-min-steps", type=int, default=10)
    parser.add_argument("--threat-break-clear-steps", type=int, default=5)
    parser.add_argument("--threat-break-recovery-steps", type=int, default=10)
    parser.add_argument(
        "--threat-break-recovery-ta-max-deg", type=float, default=180.0,
        help=("run forced opponent reacquisition only when target aspect "
              "is at or below this limit; 180 disables the gate"))
    parser.add_argument(
        "--threat-break-pulse",
        action=argparse.BooleanOptionalAction, default=False,
        help=("hold a high-urgency break for a short fixed pulse, then "
              "return control to PPO without warning-clear or recovery"))
    parser.add_argument("--threat-break-pulse-steps", type=int, default=4)
    parser.add_argument(
        "--threat-break-pulse-urgency", type=float, default=0.5)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--archive-interval", type=int, default=250)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--allow-curriculum-override-on-resume",
        action=argparse.BooleanOptionalAction, default=False,
        help=("allow only opponent_curriculum to differ from a resume "
              "checkpoint; all PPO, reward, observation and environment "
              "contract fields remain immutable"))
    parser.add_argument(
        "--allow-action-interface-override-on-resume",
        action=argparse.BooleanOptionalAction, default=False,
        help=("allow only action_interface to differ from a resume "
              "checkpoint; intended for isolated command-span ablations"))
    parser.add_argument(
        "--allow-reward-override-on-resume",
        action=argparse.BooleanOptionalAction, default=False,
        help=("allow only the named reward contract fields to differ from a "
              "resume checkpoint; intended for isolated reward ablations"))
    parser.add_argument("--seed", type=int, default=85)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"),
                        default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction,
                        default=True)
    return parser.parse_args()


def experiment_config(args, obs_dim: int, action_dims: list[int], env_config):
    """Fields that must stay identical when resuming a formal experiment."""
    config = {
        "schema": "WVRPopulationPPOBaselineV2",
        "scenario": args.scenario,
        "reward_variant": args.reward_variant,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "rollout_steps": args.rollout_steps,
        "epochs": args.epochs,
        "minibatch_size": args.minibatch_size,
        "hidden_size": args.hidden_size,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_ratio": args.clip_ratio,
        "value_coef": args.value_coef,
        "entropy_coef": args.entropy_coef,
        "max_grad_norm": args.max_grad_norm,
        "population_capacity": args.population_capacity,
        "snapshot_interval": args.snapshot_interval,
        "max_opponents_per_update": args.max_opponents_per_update,
        "obs_dim": obs_dim,
        "action_dims": list(action_dims),
        "fire_mask_obs_index": 23,
        "threat_observation_v2": bool(args.threat_observation_v2),
        "threat_warning_obs_index": (
            24 if args.threat_observation_v2 else None),
        "threat_urgency_obs_index": (
            25 if args.threat_observation_v2 else None),
        "altitude_safety_mask": bool(args.altitude_safety_mask),
        "altitude_safety_obs_index": 0,
        "altitude_descend_mask_threshold": (
            float(args.altitude_descend_mask_m) / 5000.0),
        "altitude_force_climb_threshold": (
            float(args.altitude_force_climb_m) / 5000.0),
        "environment_contract": {
            "reward_contract": str(env_config.FormalRewardContract_version),
            "max_steps": int(env_config.max_steps),
            "max_attack_angle": float(env_config.max_attack_angle),
            "max_attack_distance": float(env_config.max_attack_distance),
            "min_attack_interval": int(env_config.min_attack_interval),
            "initial_range_m": [
                float(env_config.InitialEngagement_min_range_m),
                float(env_config.InitialEngagement_max_range_m),
            ],
            "initial_aspect_probabilities": [
                float(value) for value in
                env_config.InitialEngagement_aspect_probabilities
            ],
            "timeout_adjudication": bool(
                env_config.TimeoutAdjudication_enabled),
            "resolve_pending_missiles_after_elimination": bool(
                env_config.ResolvePendingMissilesAfterElimination_enabled),
            "shoot_penalty_scale": float(
                env_config.ShootPenaltyReward_scale),
            "team_outcome_scale": float(env_config.TeamOutcomeReward_scale),
            "timeout_outcome_scale": float(
                getattr(env_config, "TimeoutOutcomeReward_scale", 0.0)),
            "reward_variant": str(env_config.RewardVariant),
        },
    }
    final_anchor_fraction = (
        args.scripted_anchor_fraction
        if args.scripted_anchor_final_fraction is None
        else args.scripted_anchor_final_fraction)
    if args.scripted_anchor_fraction > 0.0 or final_anchor_fraction > 0.0:
        config["opponent_curriculum"] = {
            "scripted_anchor_fraction": float(
                args.scripted_anchor_fraction),
            "scripted_anchor_final_fraction": float(
                final_anchor_fraction),
            "scripted_anchor_decay_updates": int(
                args.scripted_anchor_decay_updates),
            "scripted_anchor_ramp_start_update": int(
                args.scripted_anchor_ramp_start_update),
            "scripted_anchor_policies": list(
                args.scripted_anchor_policies),
        }
    if (float(args.heading_command_span_deg) != 30.0 or
            args.threat_relative_break_skill or args.threat_break_option or
            args.threat_break_pulse):
        config["action_interface"] = {
            "heading_command_span_deg": float(
                args.heading_command_span_deg),
            "heading_command_bins_deg": [
                float(value) for value in np.linspace(
                    -args.heading_command_span_deg,
                    args.heading_command_span_deg, 5)
            ],
            "threat_relative_break_skill": bool(
                args.threat_relative_break_skill),
            "threat_break_option": bool(args.threat_break_option),
            "threat_break_min_steps": int(args.threat_break_min_steps),
            "threat_break_clear_steps": int(args.threat_break_clear_steps),
            "threat_break_recovery_steps": int(
                args.threat_break_recovery_steps),
            "threat_break_recovery_ta_max_deg": float(
                args.threat_break_recovery_ta_max_deg),
            "threat_break_pulse": bool(args.threat_break_pulse),
            "threat_break_pulse_steps": int(args.threat_break_pulse_steps),
            "threat_break_pulse_urgency": float(
                args.threat_break_pulse_urgency),
        }
    return config


def anchor_fraction_for_update(args, update: int) -> float:
    """Return the deterministic scripted-anchor fraction for one update."""
    start = float(args.scripted_anchor_fraction)
    if args.scripted_anchor_final_fraction is None:
        return start
    end = float(args.scripted_anchor_final_fraction)
    decay = int(args.scripted_anchor_decay_updates)
    if decay <= 1:
        return end
    ramp_start = int(getattr(args, "scripted_anchor_ramp_start_update", 1))
    progress = np.clip(
        (int(update) - ramp_start) / (decay - 1), 0.0, 1.0)
    return float(start + progress * (end - start))


def build_opponent_schedule(population, generator, count, max_unique,
                            scripted_anchor_fraction,
                            scripted_anchor_policies):
    """Stratify historical self-play and fixed anchors across environments."""

    count = int(count)
    anchor_count = int(round(count * float(scripted_anchor_fraction)))
    anchor_count = min(count, max(0, anchor_count))
    opponent_indices = np.full(count, -1, dtype=np.int64)
    scripted_assignments = np.full(count, "", dtype=object)
    slots = generator.permutation(count)
    anchor_slots = slots[:anchor_count]
    historical_slots = slots[anchor_count:]
    if len(historical_slots):
        opponent_indices[historical_slots] = population.sample_indices(
            generator, len(historical_slots), max_unique=max_unique)
    if anchor_count:
        policies = tuple(str(name) for name in scripted_anchor_policies)
        if not policies:
            raise ValueError(
                "scripted anchor fraction requires at least one policy")
        names = np.asarray([
            policies[index % len(policies)]
            for index in range(anchor_count)
        ], dtype=object)
        generator.shuffle(names)
        scripted_assignments[anchor_slots] = names
    return opponent_indices, scripted_assignments


def atomic_json(path: Path, payload):
    def json_safe(value):
        if isinstance(value, dict):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        if isinstance(value, (float, np.floating)):
            numeric = float(value)
            return numeric if np.isfinite(numeric) else None
        if isinstance(value, (int, np.integer)):
            return int(value)
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            json_safe(payload), stream, indent=2, sort_keys=True,
            allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False))
        stream.write("\n")
        stream.flush()


def move_optimizer_state(optimizer, device):
    for state in optimizer.state.values():
        for key, value in tuple(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def save_checkpoint(path: Path, archive_path: Path | None, *, config,
                    update, total_steps, total_launches, total_completed,
                    total_fire_requests, total_rejected_requests,
                    total_legal_opportunities, total_timeouts,
                    total_tactical_timeout_decisions,
                    total_termination_reasons,
                    learner, optimizer, scaler, population, generator):
    payload = {
        "schema": "WVRPopulationPPOCheckpointV2",
        "config": config,
        "update": int(update),
        "total_steps": int(total_steps),
        "total_launches": int(total_launches),
        "total_completed": int(total_completed),
        "total_fire_requests": int(total_fire_requests),
        "total_rejected_requests": int(total_rejected_requests),
        "total_legal_opportunities": int(total_legal_opportunities),
        "total_timeouts": int(total_timeouts),
        "total_tactical_timeout_decisions": int(
            total_tactical_timeout_decisions),
        "total_termination_reasons": {
            str(name): int(count)
            for name, count in total_termination_reasons.items()
        },
        "learner": learner.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "population": population.state_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (torch.cuda.get_rng_state_all()
                           if torch.cuda.is_available() else None),
        "population_rng_state": generator.bit_generator.state,
        "saved_unix_time": time.time(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    if archive_path is not None:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_temporary = archive_path.with_name(archive_path.name + ".tmp")
        torch.save(payload, archive_temporary)
        os.replace(archive_temporary, archive_path)


def resume_configs_compatible(saved_config, expected_config,
                              allow_curriculum_override=False,
                              allow_action_interface_override=False,
                              allow_reward_override=False):
    def normalize_legacy_defaults(config):
        """Make old manifests explicit without weakening contract checks."""
        normalized = dict(config)
        action_interface = normalized.get("action_interface")
        if isinstance(action_interface, dict):
            action_interface = dict(action_interface)
            action_interface.setdefault("threat_relative_break_skill", False)
            action_interface.setdefault("threat_break_option", False)
            action_interface.setdefault("threat_break_min_steps", 10)
            action_interface.setdefault("threat_break_clear_steps", 5)
            action_interface.setdefault("threat_break_recovery_steps", 10)
            action_interface.setdefault(
                "threat_break_recovery_ta_max_deg", 180.0)
            action_interface.setdefault("threat_break_pulse", False)
            action_interface.setdefault("threat_break_pulse_steps", 4)
            action_interface.setdefault("threat_break_pulse_urgency", 0.5)
            normalized["action_interface"] = action_interface
        return normalized

    saved_config = normalize_legacy_defaults(saved_config)
    expected_config = normalize_legacy_defaults(expected_config)
    if saved_config == expected_config:
        return True
    allowed = set()
    if allow_curriculum_override:
        allowed.add("opponent_curriculum")
    if allow_action_interface_override:
        allowed.add("action_interface")
    if allow_reward_override:
        allowed.add("reward_variant")
    if not allowed:
        return False
    immutable_saved = {
        key: value for key, value in saved_config.items()
        if key not in allowed
    }
    immutable_expected = {
        key: value for key, value in expected_config.items()
        if key not in allowed
    }
    if allow_reward_override:
        reward_contract_fields = {
            "reward_contract", "reward_variant", "timeout_outcome_scale"
        }
        for config in (immutable_saved, immutable_expected):
            environment = config.get("environment_contract")
            if isinstance(environment, dict):
                config["environment_contract"] = {
                    key: value for key, value in environment.items()
                    if key not in reward_contract_fields
                }
    return immutable_saved == immutable_expected


def load_checkpoint(path: Path, *, expected_config, learner, optimizer,
                    scaler, population, generator, device,
                    allow_curriculum_override=False,
                    allow_action_interface_override=False,
                    allow_reward_override=False):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "WVRPopulationPPOCheckpointV2":
        raise ValueError("unsupported training checkpoint schema")
    if not resume_configs_compatible(
            payload.get("config", {}), expected_config,
            allow_curriculum_override=allow_curriculum_override,
            allow_action_interface_override=(
                allow_action_interface_override),
            allow_reward_override=allow_reward_override):
        raise ValueError("checkpoint experiment configuration does not match")
    learner.load_state_dict(payload["learner"])
    optimizer.load_state_dict(payload["optimizer"])
    move_optimizer_state(optimizer, device)
    scaler.load_state_dict(payload["scaler"])
    population.load_state_dict(payload["population"])
    generator.bit_generator.state = payload["population_rng_state"]
    return payload


def restore_rng_states(payload):
    random.setstate(payload["python_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    if torch.cuda.is_available() and payload.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])


def _weapon_events_by_shooter(infos, learner_id: str, opponent_id: str):
    event_types = (
        "launch", "hit", "redundant_hit", "miss", "invalidated")
    learner = {name: 0 for name in event_types}
    opponent = {name: 0 for name in event_types}
    for info in np.asarray(infos, dtype=object).reshape(-1):
        if not isinstance(info, dict):
            continue
        for event in info.get("weapon_events", ()):
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("event_type", ""))
            if event_type not in learner:
                continue
            if str(event.get("shooter_id")) == learner_id:
                learner[event_type] += 1
            elif str(event.get("shooter_id")) == opponent_id:
                opponent[event_type] += 1
    return learner, opponent


def _outcomes(infos, completed: np.ndarray, learner_id: str):
    wins = draws = losses = timeouts = tactical_timeouts = 0
    termination_reasons = {}
    learner_termination_conditions = {}
    flat_infos = np.asarray(infos, dtype=object).reshape(-1)
    for index in np.flatnonzero(completed):
        info = flat_infos[int(index)]
        outcome = info.get("team_outcome", {}) if isinstance(info, dict) else {}
        result = int(outcome.get("ego_result", 0))
        reason = str(outcome.get("reason", ""))
        termination_reasons[reason] = termination_reasons.get(reason, 0) + 1
        termination = info.get("termination", {}).get(learner_id, {})
        condition = str(termination.get("reason", "unknown"))
        learner_termination_conditions[condition] = (
            learner_termination_conditions.get(condition, 0) + 1)
        is_timeout = reason.startswith("timeout")
        # Training W/D/L reports the kill objective. Tactical adjudication is
        # retained separately and never masquerades as a weapon win.
        wins += int((not is_timeout) and result > 0)
        draws += int(is_timeout or result == 0)
        losses += int((not is_timeout) and result < 0)
        timeouts += int(is_timeout)
        tactical_timeouts += int(reason == "timeout_tactical_advantage")
    return (wins, draws, losses, timeouts, tactical_timeouts,
            termination_reasons, learner_termination_conditions)


def _weapon_step_counts(infos):
    requests = rejected = opportunities = 0
    for info in np.asarray(infos, dtype=object).reshape(-1):
        if not isinstance(info, dict):
            continue
        requests += int(info.get("weapon_fire_requests", 0))
        rejected += int(info.get("weapon_rejected_requests", 0))
        opportunities += int(info.get("weapon_legal_opportunities", 0))
    return requests, rejected, opportunities


def _reward_component_sums(infos):
    totals = {}
    for info in np.asarray(infos, dtype=object).reshape(-1):
        if not isinstance(info, dict):
            continue
        for name, values in info.get("team_reward_components", {}).items():
            if not isinstance(values, (list, tuple)) or not values:
                continue
            totals[str(name)] = totals.get(str(name), 0.0) + float(values[0])
    return totals


def collect_population_rollout(
        env, learner, opponent_models, opponent_indices,
        scripted_assignments, obs, args, device,
        learner_id: str, opponent_id: str):
    observations = []
    actions = []
    log_probs = []
    values = []
    rewards = []
    dones = []
    episode_returns = np.zeros(env.num_envs, dtype=np.float64)
    completed_returns = []
    completed_episodes = 0
    learner_launches = opponent_launches = 0
    learner_weapon_events = {
        name: 0 for name in (
            "launch", "hit", "redundant_hit", "miss", "invalidated")}
    opponent_weapon_events = dict(learner_weapon_events)
    wins = draws = losses = 0
    fire_requests = rejected_requests = legal_opportunities = 0
    timeouts = tactical_timeout_decisions = 0
    termination_reasons = {}
    learner_termination_conditions = {}
    reward_component_totals = {}
    zero_sum_error = 0.0

    for _ in range(args.rollout_steps):
        learner_obs = torch.as_tensor(
            obs[:, 0], dtype=torch.float32, device=device)
        with torch.inference_mode():
            learner_action, learner_log_prob, learner_value = learner.act(
                learner_obs)

        opponent_action = torch.empty(
            (env.num_envs, learner_action.shape[-1]), dtype=torch.long,
            device=device)
        with torch.inference_mode():
            for population_index, model in opponent_models.items():
                selected = np.flatnonzero(
                    opponent_indices == population_index)
                selected_tensor = torch.as_tensor(
                    selected, dtype=torch.long, device=device)
                sampled_action, _, _ = model.act(torch.as_tensor(
                    obs[selected, 1], dtype=torch.float32, device=device))
                opponent_action.index_copy_(
                    0, selected_tensor, sampled_action)
            for policy_name in sorted(set(
                    str(name) for name in scripted_assignments if name)):
                selected = np.flatnonzero(
                    scripted_assignments == policy_name)
                selected_tensor = torch.as_tensor(
                    selected, dtype=torch.long, device=device)
                scripted_policy = OPPONENT_POLICIES[policy_name]
                if (args.threat_relative_break_skill and
                        policy_name == "evasive_fire"):
                    scripted_policy = scripted_evasive_fire_break_skill
                scripted_action = torch.as_tensor(
                    scripted_policy(obs[selected, 1]),
                    dtype=torch.long, device=device)
                opponent_action.index_copy_(
                    0, selected_tensor, scripted_action)

        joint_action = torch.stack(
            (learner_action, opponent_action), dim=1).cpu().numpy()
        next_obs, reward, done, infos = env.step(joint_action)
        side_events = _weapon_events_by_shooter(
            infos, learner_id, opponent_id)
        for name in learner_weapon_events:
            learner_weapon_events[name] += side_events[0][name]
            opponent_weapon_events[name] += side_events[1][name]
        learner_launches += side_events[0]["launch"]
        opponent_launches += side_events[1]["launch"]
        weapon_counts = _weapon_step_counts(infos)
        fire_requests += weapon_counts[0]
        rejected_requests += weapon_counts[1]
        legal_opportunities += weapon_counts[2]
        for name, value in _reward_component_sums(infos).items():
            reward_component_totals[name] = (
                reward_component_totals.get(name, 0.0) + value)

        reward = np.asarray(reward, dtype=np.float32).reshape(
            env.num_envs, env.num_agents)
        done = np.asarray(done, dtype=np.bool_).reshape(
            env.num_envs, env.num_agents)
        learner_reward = reward[:, 0]
        learner_done = done[:, 0]
        zero_sum_error = max(
            zero_sum_error,
            float(np.max(np.abs(reward.sum(axis=-1)))),
        )
        episode_returns += learner_reward

        observations.append(np.asarray(obs[:, 0], dtype=np.float32))
        actions.append(learner_action.cpu().numpy().astype(np.int64))
        log_probs.append(learner_log_prob.cpu().numpy())
        values.append(learner_value.cpu().numpy())
        rewards.append(learner_reward)
        dones.append(learner_done)

        obs = next_obs
        completed = np.all(done, axis=1)
        if np.any(completed):
            completed_returns.extend(episode_returns[completed].tolist())
            completed_episodes += int(np.count_nonzero(completed))
            result = _outcomes(infos, completed, learner_id)
            wins += result[0]
            draws += result[1]
            losses += result[2]
            timeouts += result[3]
            tactical_timeout_decisions += result[4]
            for reason, count in result[5].items():
                termination_reasons[reason] = (
                    termination_reasons.get(reason, 0) + count)
            for condition, count in result[6].items():
                learner_termination_conditions[condition] = (
                    learner_termination_conditions.get(condition, 0) + count)
            episode_returns[completed] = 0.0

    with torch.inference_mode():
        next_value = learner.value(torch.as_tensor(
            obs[:, 0], dtype=torch.float32, device=device)).cpu().numpy()

    rewards_array = np.asarray(rewards, dtype=np.float32)
    dones_array = np.asarray(dones, dtype=np.float32)
    values_array = np.asarray(values, dtype=np.float32)
    advantages = np.zeros_like(rewards_array)
    gae = np.zeros(env.num_envs, dtype=np.float32)
    for step in reversed(range(args.rollout_steps)):
        following_value = (
            next_value if step == args.rollout_steps - 1
            else values_array[step + 1]
        )
        nonterminal = 1.0 - dones_array[step]
        delta = (rewards_array[step]
                 + args.gamma * following_value * nonterminal
                 - values_array[step])
        gae = delta + args.gamma * args.gae_lambda * nonterminal * gae
        advantages[step] = gae

    returns = advantages + values_array
    flat_advantages = advantages.reshape(-1)
    flat_advantages = (
        flat_advantages - flat_advantages.mean()
    ) / (flat_advantages.std() + 1.0e-8)
    rollout = Rollout(
        observations=torch.as_tensor(
            np.asarray(observations).reshape(-1, obs.shape[-1]),
            dtype=torch.float32, device=device),
        actions=torch.as_tensor(
            np.asarray(actions).reshape(-1, joint_action.shape[-1]),
            dtype=torch.long, device=device),
        old_log_probs=torch.as_tensor(
            np.asarray(log_probs).reshape(-1), dtype=torch.float32,
            device=device),
        old_values=torch.as_tensor(
            values_array.reshape(-1), dtype=torch.float32, device=device),
        returns=torch.as_tensor(
            returns.reshape(-1), dtype=torch.float32, device=device),
        advantages=torch.as_tensor(
            flat_advantages, dtype=torch.float32, device=device),
        mean_reward=float(rewards_array.mean()),
        mean_abs_reward=float(np.abs(rewards_array).mean()),
        zero_sum_error=float(zero_sum_error),
        completed_episodes=completed_episodes,
        mean_abs_completed_return=(float(np.mean(np.abs(completed_returns)))
                                   if completed_returns else float("nan")),
        accepted_launches=learner_launches + opponent_launches,
    )
    stats = PopulationStats(
        learner_launches=learner_launches,
        opponent_launches=opponent_launches,
        learner_weapon_events=learner_weapon_events,
        opponent_weapon_events=opponent_weapon_events,
        wins=wins, draws=draws, losses=losses,
        fire_requests=fire_requests,
        rejected_requests=rejected_requests,
        legal_opportunities=legal_opportunities,
        timeouts=timeouts,
        tactical_timeout_decisions=tactical_timeout_decisions,
        termination_reasons=termination_reasons,
        learner_termination_conditions=learner_termination_conditions,
        mean_reward_components={
            name: value / (args.rollout_steps * env.num_envs)
            for name, value in sorted(reward_component_totals.items())
        },
    )
    return rollout, obs, stats


def main():
    args = parse_args()
    if (args.updates <= 0 or args.rollout_steps <= 0 or args.epochs <= 0 or
            args.num_envs <= 0 or args.minibatch_size <= 0):
        raise ValueError("training sizes must be positive")
    if args.envs_per_worker <= 0:
        raise ValueError("envs-per-worker must be positive")
    if args.num_envs > 1 and args.num_envs % args.envs_per_worker:
        raise ValueError("num-envs must be divisible by envs-per-worker")
    if (args.population_capacity < 2 or args.snapshot_interval <= 0 or
            args.checkpoint_interval <= 0 or args.archive_interval <= 0 or
            args.max_opponents_per_update < 0):
        raise ValueError("population and persistence intervals are invalid")
    if not 0.0 <= args.scripted_anchor_fraction <= 1.0:
        raise ValueError("scripted anchor fraction must be in [0, 1]")
    if (args.scripted_anchor_final_fraction is not None and
            not 0.0 <= args.scripted_anchor_final_fraction <= 1.0):
        raise ValueError("final scripted anchor fraction must be in [0, 1]")
    if (args.scripted_anchor_final_fraction is not None and
            args.scripted_anchor_decay_updates <= 0):
        raise ValueError(
            "scripted anchor decay updates must be positive when a final "
            "fraction is supplied")
    if args.scripted_anchor_ramp_start_update <= 0:
        raise ValueError("scripted anchor ramp start update must be positive")
    if (args.altitude_force_climb_m <= 2500.0 or
            args.altitude_descend_mask_m <=
            args.altitude_force_climb_m):
        raise ValueError(
            "altitude safety thresholds must satisfy termination < force "
            "climb < descend mask")
    if (not np.isfinite(args.heading_command_span_deg) or
            not 15.0 <= args.heading_command_span_deg <= 90.0):
        raise ValueError("heading command span must be in [15, 90] degrees")
    if args.threat_relative_break_skill and not args.threat_observation_v2:
        raise ValueError(
            "threat-relative break skill requires threat observation V2")
    if args.threat_break_option and not args.threat_relative_break_skill:
        raise ValueError("threat break option requires threat-relative skill")
    if args.threat_break_pulse and not args.threat_relative_break_skill:
        raise ValueError("threat break pulse requires threat-relative skill")
    if args.threat_break_option and args.threat_break_pulse:
        raise ValueError("threat break option and pulse are mutually exclusive")
    if args.threat_break_pulse_steps <= 0:
        raise ValueError("threat break pulse duration must be positive")
    if (not np.isfinite(args.threat_break_pulse_urgency) or
            not 0.0 <= args.threat_break_pulse_urgency <= 1.0):
        raise ValueError("threat break pulse urgency must be in [0, 1]")
    if min(args.threat_break_min_steps, args.threat_break_clear_steps,
           args.threat_break_recovery_steps) <= 0:
        raise ValueError("threat break option durations must be positive")
    if (not np.isfinite(args.threat_break_recovery_ta_max_deg) or
            not 0.0 <= args.threat_break_recovery_ta_max_deg <= 180.0):
        raise ValueError(
            "threat break recovery TA limit must be in [0, 180] degrees")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    generator = np.random.default_rng(args.seed + 7919)
    device = select_device(args.device)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    probe = make_environment(
        args.seed, scenario=args.scenario,
        reward_variant=args.reward_variant,
        threat_observation_v2=args.threat_observation_v2,
        heading_command_span_deg=args.heading_command_span_deg,
        threat_relative_break_skill=args.threat_relative_break_skill,
        threat_break_option=args.threat_break_option,
        threat_break_min_steps=args.threat_break_min_steps,
        threat_break_clear_steps=args.threat_break_clear_steps,
        threat_break_recovery_steps=args.threat_break_recovery_steps,
        threat_break_recovery_ta_max_deg=(
            args.threat_break_recovery_ta_max_deg),
        threat_break_pulse=args.threat_break_pulse,
        threat_break_pulse_steps=args.threat_break_pulse_steps,
        threat_break_pulse_urgency=args.threat_break_pulse_urgency)
    probe_obs = probe.reset()
    agent_ids = [str(agent_id) for agent_id in probe.agents.keys()]
    action_dims = [*probe.action_space[0].nvec.tolist(),
                   int(probe.action_space[1].n)]
    obs_dim = int(probe_obs.shape[-1])
    if obs_dim not in (24, 26) or not np.all(
            np.isin(probe_obs[:, 23], [0.0, 1.0])):
        raise RuntimeError(
            "formal trainer requires the 24-D/26-D weapon-aware "
            "observation contract")
    lowlevel_initial = parameter_vector(probe.task.lowlevel_policy)
    if probe.task.lowlevel_policy.training or any(
            parameter.requires_grad
            for parameter in probe.task.lowlevel_policy.parameters()):
        raise RuntimeError("the hierarchical low-level controller is not frozen")

    learner = MultiDiscreteActorCritic(
        obs_dim=obs_dim, action_dims=action_dims,
        hidden_size=args.hidden_size,
        fire_mask_obs_index=23,
        altitude_safety_obs_index=(
            0 if args.altitude_safety_mask else None),
        altitude_descend_mask_threshold=(
            args.altitude_descend_mask_m / 5000.0),
        altitude_force_climb_threshold=(
            args.altitude_force_climb_m / 5000.0),
    ).to(device)
    optimizer = torch.optim.Adam(
        learner.parameters(), lr=args.learning_rate, eps=1.0e-5)
    learner_initial = parameter_vector(learner)
    population = HistoricalPolicyPopulation(args.population_capacity)
    population.add(learner, source_update=0)
    config = experiment_config(args, obs_dim, action_dims, probe.config)
    checkpoint = None
    completed_update = 0
    total_steps = total_launches = total_completed = 0
    total_fire_requests = total_rejected_requests = 0
    total_legal_opportunities = total_timeouts = 0
    total_tactical_timeout_decisions = 0
    total_termination_reasons = {}
    if args.resume is not None:
        checkpoint = load_checkpoint(
            args.resume, expected_config=config, learner=learner,
            optimizer=optimizer, scaler=scaler, population=population,
            generator=generator, device=device,
            allow_curriculum_override=(
                args.allow_curriculum_override_on_resume),
            allow_action_interface_override=(
                args.allow_action_interface_override_on_resume),
            allow_reward_override=(
                args.allow_reward_override_on_resume))
        completed_update = int(checkpoint["update"])
        total_steps = int(checkpoint["total_steps"])
        total_launches = int(checkpoint["total_launches"])
        total_completed = int(checkpoint["total_completed"])
        total_fire_requests = int(checkpoint["total_fire_requests"])
        total_rejected_requests = int(checkpoint["total_rejected_requests"])
        total_legal_opportunities = int(checkpoint["total_legal_opportunities"])
        total_timeouts = int(checkpoint["total_timeouts"])
        total_tactical_timeout_decisions = int(
            checkpoint["total_tactical_timeout_decisions"])
        total_termination_reasons = {
            str(name): int(count) for name, count in
            checkpoint.get("total_termination_reasons", {}).items()
        }
        if completed_update >= args.updates:
            raise ValueError("checkpoint already reached the requested updates")

    resume_seed_offset = completed_update * 1_000_003
    factories = [
        partial(
            make_environment,
            args.seed + resume_seed_offset + index * 1000,
            scenario=args.scenario,
            reward_variant=args.reward_variant,
            threat_observation_v2=args.threat_observation_v2,
            heading_command_span_deg=args.heading_command_span_deg,
            threat_relative_break_skill=args.threat_relative_break_skill,
            threat_break_option=args.threat_break_option,
            threat_break_min_steps=args.threat_break_min_steps,
            threat_break_clear_steps=args.threat_break_clear_steps,
            threat_break_recovery_steps=args.threat_break_recovery_steps,
            threat_break_recovery_ta_max_deg=(
                args.threat_break_recovery_ta_max_deg),
            threat_break_pulse=args.threat_break_pulse,
            threat_break_pulse_steps=args.threat_break_pulse_steps,
            threat_break_pulse_urgency=args.threat_break_pulse_urgency,
        )
        for index in range(args.num_envs)
    ]
    env = (DummyVecEnv(factories) if args.num_envs == 1 else
           SubprocVecEnv(
               factories, context=args.start_method,
               in_series=args.envs_per_worker))
    obs = env.reset()
    if checkpoint is not None:
        restore_rng_states(checkpoint)

    founding_snapshot = population.snapshots[0]
    founding_parameters = founding_snapshot.parameter_vector.clone()
    run_dir = args.run_dir
    target_steps = args.updates * args.rollout_steps * args.num_envs
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = run_dir / "experiment_manifest.json"
        manifest = {
            **config,
            "target_updates": args.updates,
            "target_steps": target_steps,
            "device": str(device),
            "amp": amp_enabled,
        }
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing != manifest:
                raise ValueError("run directory manifest does not match")
        else:
            atomic_json(manifest_path, manifest)

    print(
        "mode=stratified_historical_selfplay scenario={0} variant={1} "
        "envs={2} envs_per_worker={3} learner_side={4} "
        "opponent_side={5} device={6} amp={7} population={8} "
        "resume_update={9} target_updates={10}".format(
            args.scenario, args.reward_variant, args.num_envs,
            args.envs_per_worker, agent_ids[0], agent_ids[1], device,
            amp_enabled, len(population.snapshots), completed_update,
            args.updates),
        flush=True)
    start = time.perf_counter()
    session_start_steps = total_steps
    last_metrics = None
    last_stats = None
    try:
        for update in range(completed_update + 1, args.updates + 1):
            population_before = len(population.snapshots)
            opponent_indices, scripted_assignments = build_opponent_schedule(
                population, generator, args.num_envs,
                args.max_opponents_per_update,
                anchor_fraction_for_update(args, update),
                args.scripted_anchor_policies)
            opponent_models = {
                int(index): population.materialize(
                    int(index), obs_dim, action_dims, args.hidden_size, device,
                    fire_mask_obs_index=23,
                    altitude_safety_obs_index=(
                        0 if args.altitude_safety_mask else None),
                    altitude_descend_mask_threshold=(
                        args.altitude_descend_mask_m / 5000.0),
                    altitude_force_climb_threshold=(
                        args.altitude_force_climb_m / 5000.0))
                for index in np.unique(opponent_indices)
                if int(index) >= 0
            }
            sampled_ids = [
                population.snapshots[int(index)].snapshot_id
                for index in opponent_indices
                if int(index) >= 0
            ]
            histogram = {
                f"snapshot:{snapshot_id}": sampled_ids.count(snapshot_id)
                for snapshot_id in sorted(set(sampled_ids))
            }
            for policy_name in sorted(set(
                    str(name) for name in scripted_assignments if name)):
                histogram[f"scripted:{policy_name}"] = int(np.count_nonzero(
                    scripted_assignments == policy_name))
            rollout, obs, stats = collect_population_rollout(
                env, learner, opponent_models, opponent_indices,
                scripted_assignments, obs, args, device,
                agent_ids[0], agent_ids[1])
            metrics = ppo_update(
                learner, optimizer, rollout, args, scaler, amp_enabled)
            new_snapshot_id = None
            if update % args.snapshot_interval == 0:
                new_snapshot_id = population.add(
                    learner, source_update=update)
            total_steps += args.rollout_steps * args.num_envs
            total_launches += stats.learner_launches + stats.opponent_launches
            total_completed += rollout.completed_episodes
            total_fire_requests += stats.fire_requests
            total_rejected_requests += stats.rejected_requests
            total_legal_opportunities += stats.legal_opportunities
            total_timeouts += stats.timeouts
            total_tactical_timeout_decisions += (
                stats.tactical_timeout_decisions)
            for reason, count in stats.termination_reasons.items():
                total_termination_reasons[reason] = (
                    total_termination_reasons.get(reason, 0) + count)
            last_metrics = metrics
            last_stats = stats
            print(
                "update={0}/{1} samples={2} population={3}->{4} sampled={5} "
                "new_snapshot={6} reward={7:+.6f} launches={8}/{9} "
                "wdl={10}/{11}/{12} timeout={13} fire={14}/{15} "
                "legal_opportunities={16} policy_loss={17:+.5f} "
                "value_loss={18:.5f} entropy={19:.4f} kl={20:.6f} "
                "zero_sum_err={21:.3e}".format(
                    update, args.updates, rollout.observations.shape[0],
                    population_before, len(population.snapshots), histogram,
                    new_snapshot_id, rollout.mean_reward,
                    stats.learner_launches, stats.opponent_launches,
                    stats.wins, stats.draws, stats.losses,
                    stats.timeouts, stats.fire_requests,
                    stats.rejected_requests, stats.legal_opportunities,
                    metrics["policy_loss"], metrics["value_loss"],
                    metrics["entropy"], metrics["approx_kl"],
                    rollout.zero_sum_error), flush=True)
            del opponent_models

            session_elapsed = time.perf_counter() - start
            session_steps = total_steps - session_start_steps
            throughput = session_steps / max(session_elapsed, 1.0e-9)
            progress = {
                "schema": "WVRPopulationPPOProgressV1",
                "status": "running",
                "pid": os.getpid(),
                "update": update,
                "target_updates": args.updates,
                "total_steps": total_steps,
                "target_steps": target_steps,
                "percent": 100.0 * update / args.updates,
                "session_elapsed_s": session_elapsed,
                "session_throughput_steps_s": throughput,
                "eta_s": (target_steps - total_steps) /
                         max(throughput, 1.0e-9),
                "population_size": len(population.snapshots),
                "population_snapshot_ids": [
                    snapshot.snapshot_id for snapshot in population.snapshots
                ],
                "sampled_opponents": histogram,
                "reward": rollout.mean_reward,
                "zero_sum_error": rollout.zero_sum_error,
                "learner_launches": stats.learner_launches,
                "opponent_launches": stats.opponent_launches,
                "learner_weapon_events": stats.learner_weapon_events,
                "opponent_weapon_events": stats.opponent_weapon_events,
                "wins": stats.wins,
                "draws": stats.draws,
                "losses": stats.losses,
                "timeouts": stats.timeouts,
                "tactical_timeout_decisions": stats.tactical_timeout_decisions,
                "termination_reasons": stats.termination_reasons,
                "learner_termination_conditions": (
                    stats.learner_termination_conditions),
                "fire_requests": stats.fire_requests,
                "rejected_fire_requests": stats.rejected_requests,
                "legal_fire_opportunities": stats.legal_opportunities,
                "mean_reward_components": stats.mean_reward_components,
                "episodes_this_update": rollout.completed_episodes,
                "total_episodes": total_completed,
                "total_launches": total_launches,
                "total_fire_requests": total_fire_requests,
                "total_rejected_fire_requests": total_rejected_requests,
                "total_legal_fire_opportunities": total_legal_opportunities,
                "total_timeouts": total_timeouts,
                "total_tactical_timeout_decisions": (
                    total_tactical_timeout_decisions),
                "total_termination_reasons": total_termination_reasons,
                "metrics": metrics,
                "updated_unix_time": time.time(),
            }
            if run_dir is not None:
                atomic_json(run_dir / "progress.json", progress)
                append_jsonl(run_dir / "metrics.jsonl", progress)
                if (update % args.checkpoint_interval == 0
                        or update == args.updates):
                    archive_path = None
                    if (update % args.archive_interval == 0
                            or update == args.updates):
                        archive_path = (
                            run_dir / "checkpoints" /
                            "checkpoint_{:06d}.pt".format(update))
                    save_checkpoint(
                        run_dir / "latest.pt", archive_path,
                        config=config, update=update,
                        total_steps=total_steps,
                        total_launches=total_launches,
                        total_completed=total_completed,
                        total_fire_requests=total_fire_requests,
                        total_rejected_requests=total_rejected_requests,
                        total_legal_opportunities=total_legal_opportunities,
                        total_timeouts=total_timeouts,
                        total_tactical_timeout_decisions=(
                            total_tactical_timeout_decisions),
                        total_termination_reasons=(
                            total_termination_reasons),
                        learner=learner, optimizer=optimizer, scaler=scaler,
                        population=population, generator=generator)
    finally:
        env.close()
        probe.close()

    learner_change = torch.linalg.vector_norm(
        parameter_vector(learner) - learner_initial).item()
    founding_change = torch.linalg.vector_norm(
        founding_snapshot.parameter_vector - founding_parameters).item()
    lowlevel_change = torch.linalg.vector_norm(
        parameter_vector(probe.task.lowlevel_policy) - lowlevel_initial).item()
    if not np.isfinite(learner_change) or learner_change <= 0.0:
        raise RuntimeError("PPO did not update the learner")
    if founding_change != 0.0:
        raise RuntimeError("training mutated a frozen population snapshot")
    if lowlevel_change != 0.0:
        raise RuntimeError("training mutated the frozen low-level controller")
    if total_launches <= 0:
        raise RuntimeError("population training did not exercise a launch")
    elapsed = time.perf_counter() - start
    if run_dir is not None:
        completion = {
            "schema": "WVRPopulationPPOProgressV1",
            "status": "completed",
            "pid": os.getpid(),
            "update": args.updates,
            "target_updates": args.updates,
            "total_steps": total_steps,
            "target_steps": target_steps,
            "percent": 100.0,
            "session_elapsed_s": elapsed,
            "session_throughput_steps_s": (
                (total_steps - session_start_steps) /
                max(elapsed, 1.0e-9)),
            "population_size": len(population.snapshots),
            "total_episodes": total_completed,
            "total_launches": total_launches,
            "total_fire_requests": total_fire_requests,
            "total_rejected_fire_requests": total_rejected_requests,
            "total_legal_fire_opportunities": total_legal_opportunities,
            "total_timeouts": total_timeouts,
            "timeout_rate": (
                total_timeouts / total_completed if total_completed else None),
            "legal_launch_acceptance_rate": (
                (total_fire_requests - total_rejected_requests) /
                total_fire_requests if total_fire_requests else None),
            "legal_opportunity_exploitation_rate": (
                total_launches / total_legal_opportunities
                if total_legal_opportunities else None),
            "total_tactical_timeout_decisions": (
                total_tactical_timeout_decisions),
            "total_termination_reasons": total_termination_reasons,
            "metrics": last_metrics,
            "last_wdl": ({
                "wins": last_stats.wins,
                "draws": last_stats.draws,
                "losses": last_stats.losses,
                "timeouts": last_stats.timeouts,
            } if last_stats is not None else None),
            "last_mean_reward_components": (
                last_stats.mean_reward_components
                if last_stats is not None else None),
            "updated_unix_time": time.time(),
        }
        atomic_json(run_dir / "progress.json", completion)
        atomic_json(run_dir / "COMPLETED.json", completion)
    print(
        "PASS decisions={0} learner_transitions={0} elapsed_s={1:.3f} "
        "decisions_per_s={2:.1f} population={3} episodes={4} launches={5} "
        "learner_parameter_l2_change={6:.6f} "
        "founding_snapshot_l2_change={7:.6f} "
        "lowlevel_parameter_l2_change={8:.6f}".format(
            total_steps, elapsed,
            (total_steps - session_start_steps) / max(elapsed, 1.0e-9),
            len(population.snapshots), total_completed, total_launches,
            learner_change, founding_change, lowlevel_change), flush=True)


if __name__ == "__main__":
    main()
