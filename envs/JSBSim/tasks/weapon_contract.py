"""Auditable simple-missile action, event and team-outcome contracts."""

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ShotLegality(object):
    shooter_id: str
    target_id: Optional[str]
    requested: bool
    can_fire: bool
    accepted: bool
    reason: str
    violations: Tuple[str, ...]
    distance_m: Optional[float]
    attack_angle_deg: Optional[float]
    cooldown_steps: int
    remaining_missiles: int

    def to_dict(self):
        result = asdict(self)
        result["violations"] = list(self.violations)
        result["shoot_action_mask"] = [True, bool(self.can_fire)]
        return result


@dataclass(frozen=True)
class WeaponEvent(object):
    event_type: str
    step: int
    shooter_id: str
    target_id: Optional[str]
    missile_id: Optional[str]
    reason: str

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TeamOutcome(object):
    winner: Optional[str]
    ego_result: int
    enemy_result: int
    label: str
    reason: str
    terminated: bool
    truncated: bool
    episode_steps: int
    ego_alive: int
    enemy_alive: int
    ego_tactical_score: Optional[float] = None
    enemy_tactical_score: Optional[float] = None

    def to_dict(self):
        return asdict(self)


def team_outcome_terminal_rewards(config, outcome):
    """Map an evaluation outcome to the zero-sum learning utility.

    Tactical timeout adjudication remains available in ``TeamOutcome`` for
    diagnostics, but a time-limit geometry advantage is not a weapon kill.
    Its learning utility therefore has an independent scale whose safe
    default is zero.
    """

    if not outcome.terminated:
        return [0.0, 0.0]
    if str(outcome.reason).startswith("timeout"):
        scale = float(getattr(config, "TimeoutOutcomeReward_scale", 0.0))
    else:
        scale = float(getattr(config, "TeamOutcomeReward_scale", 200.0))
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("team outcome reward scales must be finite and non-negative")
    return [scale * outcome.ego_result, scale * outcome.enemy_result]


def _uid(sim):
    return str(getattr(sim, "uid", ""))


def nearest_alive_enemy(agent):
    """Return the nearest living enemy, with uid as a deterministic tie-break."""
    position = np.asarray(agent.get_position(), dtype=np.float64)
    candidates = [enemy for enemy in getattr(agent, "enemies", ())
                  if bool(getattr(enemy, "is_alive", False))]
    if not candidates:
        return None
    return min(candidates, key=lambda enemy: (
        float(np.linalg.norm(np.asarray(enemy.get_position(), dtype=np.float64) - position)),
        _uid(enemy)))


def explicit_enemy_slots(agent):
    """Return stable enemy slots for target-conditioned weapon actions.

    Slots are sorted by immutable aircraft uid and deliberately retain dead
    aircraft.  Repacking a slot after a death would silently change the
    meaning of a recorded BRIC intervention and make a branch label depend on
    events that occur after the actor selected its target.
    """
    return tuple(sorted(getattr(agent, "enemies", ()), key=_uid))


def enemy_by_id(agent, target_id):
    """Resolve an explicit enemy id without falling back to a nearer target."""
    requested = str(target_id)
    for enemy in explicit_enemy_slots(agent):
        if _uid(enemy) == requested:
            return enemy
    return None


def evaluate_shot(agent, requested, remaining_missiles, current_step,
                  last_shot_step, max_attack_angle, max_attack_distance,
                  min_attack_interval, target_id=None):
    """Evaluate a fire request without mutating the environment.

    Historical PBGR callers omit ``target_id`` and retain the nearest-living
    enemy contract.  BRIC callers provide an explicit id; an invalid or dead
    selection fails closed and is never replaced with a different enemy.
    """
    explicit_target = target_id is not None
    target = (enemy_by_id(agent, target_id) if explicit_target
              else nearest_alive_enemy(agent))
    distance = None
    attack_angle = None
    violations = []
    if not bool(getattr(agent, "is_alive", False)):
        violations.append("shooter_not_alive")
    if target is None:
        violations.append("unknown_target" if explicit_target else "no_alive_target")
    elif not bool(getattr(target, "is_alive", False)):
        violations.append("target_not_alive")
    else:
        relative = (np.asarray(target.get_position(), dtype=np.float64) -
                    np.asarray(agent.get_position(), dtype=np.float64))
        heading = np.asarray(agent.get_velocity(), dtype=np.float64)
        distance = float(np.linalg.norm(relative))
        speed = float(np.linalg.norm(heading))
        if distance <= 1e-8 or speed <= 1e-8:
            attack_angle = 0.0 if distance <= 1e-8 else 180.0
        else:
            cosine = np.dot(relative, heading) / (distance * speed)
            attack_angle = float(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0))))
    if int(remaining_missiles) <= 0:
        violations.append("no_ammunition")
    cooldown = int(current_step) - int(last_shot_step)
    if cooldown < int(min_attack_interval):
        violations.append("cooldown")
    if distance is not None and distance > float(max_attack_distance):
        violations.append("out_of_range")
    if attack_angle is not None and attack_angle > float(max_attack_angle):
        violations.append("outside_firing_cone")
    can_fire = not violations
    accepted = bool(requested) and can_fire
    if accepted:
        reason = "accepted"
    elif not bool(requested):
        reason = "not_requested"
    else:
        reason = violations[0]
    return ShotLegality(
        shooter_id=_uid(agent), target_id=(str(target_id) if target is None and
                                           explicit_target else
                                           None if target is None else _uid(target)),
        requested=bool(requested), can_fire=bool(can_fire), accepted=bool(accepted),
        reason=reason, violations=tuple(violations), distance_m=distance,
        attack_angle_deg=attack_angle, cooldown_steps=cooldown,
        remaining_missiles=int(remaining_missiles)), target


def _pending_missile_against(env, source_ids, alive_target_ids):
    sources = set(source_ids)
    targets = set(alive_target_ids)
    for missile in getattr(env, "_tempsims", {}).values():
        parent = getattr(missile, "parent_aircraft", None)
        target = getattr(missile, "target_aircraft", None)
        if (bool(getattr(missile, "is_alive", False)) and
                _uid(parent) in sources and _uid(target) in targets):
            return True
    return False


def _team_tactical_score(env, team_ids, opponent_ids, distance_scale):
    """Return a bounded timeout score based only on live attack geometry.

    A score is high when a live aircraft is close to, and nose-on toward, a
    live opponent.  Ammunition is deliberately excluded so conserving every
    missile cannot become a timeout-winning strategy.
    """

    opponents = [env.agents[uid] for uid in opponent_ids
                 if bool(env.agents[uid].is_alive)]
    if not opponents:
        return 1.0
    scores = []
    for uid in team_ids:
        agent = env.agents[uid]
        if not bool(agent.is_alive):
            scores.append(0.0)
            continue
        position = np.asarray(agent.get_position(), dtype=np.float64)
        velocity = np.asarray(agent.get_velocity(), dtype=np.float64)
        speed = float(np.linalg.norm(velocity))
        best = 0.0
        for opponent in opponents:
            relative = np.asarray(
                opponent.get_position(), dtype=np.float64) - position
            distance = float(np.linalg.norm(relative))
            if distance <= 1.0e-8 or speed <= 1.0e-8:
                nose_on = 1.0 if distance <= 1.0e-8 else 0.0
            else:
                cosine = float(np.dot(relative, velocity) /
                               (distance * speed))
                nose_on = 0.5 * (1.0 + np.clip(cosine, -1.0, 1.0))
            range_factor = float(np.exp(-distance / distance_scale))
            best = max(best, nose_on * range_factor)
        scores.append(best)
    return float(np.mean(scores)) if scores else 0.0


def resolve_team_outcome(env, timeout=False):
    """Resolve a zero-sum ego-team W/D/L result from living aircraft.

    A live missile launched by the eliminated team against a still-living
    opponent is allowed to resolve before an elimination win is declared.
    The configured time limit is a task draw, not a value-bootstrap
    truncation.
    """
    ego_ids = tuple(env.ego_ids)
    enemy_ids = tuple(env.enm_ids)
    if not ego_ids or not enemy_ids:
        raise ValueError("team outcome requires two non-empty teams")
    ego_alive_ids = tuple(uid for uid in ego_ids if env.agents[uid].is_alive)
    enemy_alive_ids = tuple(uid for uid in enemy_ids if env.agents[uid].is_alive)
    ego_alive = len(ego_alive_ids)
    enemy_alive = len(enemy_alive_ids)
    step = int(env.current_step)
    if ego_alive == 0 and enemy_alive == 0:
        return TeamOutcome(None, 0, 0, "draw", "simultaneous_elimination",
                           True, False, step, ego_alive, enemy_alive)
    resolve_pending = bool(getattr(
        env.config, "ResolvePendingMissilesAfterElimination_enabled", True))
    if ego_alive == 0 and (not resolve_pending or not _pending_missile_against(
            env, ego_ids, enemy_alive_ids)):
        return TeamOutcome("enemy", -1, 1, "ego_loss", "ego_team_eliminated",
                           True, False, step, ego_alive, enemy_alive)
    if enemy_alive == 0 and (not resolve_pending or not _pending_missile_against(
            env, enemy_ids, ego_alive_ids)):
        return TeamOutcome("ego", 1, -1, "ego_win", "enemy_team_eliminated",
                           True, False, step, ego_alive, enemy_alive)
    if bool(timeout):
        if bool(getattr(env.config, "TimeoutAdjudication_enabled", False)):
            distance_scale = float(getattr(
                env.config, "TimeoutAdjudication_distance_scale_m", 20000.0))
            margin = float(getattr(
                env.config, "TimeoutAdjudication_margin", 0.05))
            if distance_scale <= 0.0 or margin < 0.0:
                raise ValueError("invalid timeout adjudication configuration")
            ego_score = _team_tactical_score(
                env, ego_ids, enemy_ids, distance_scale)
            enemy_score = _team_tactical_score(
                env, enemy_ids, ego_ids, distance_scale)
            advantage = ego_score - enemy_score
            if advantage > margin:
                return TeamOutcome(
                    "ego", 1, -1, "ego_win",
                    "timeout_tactical_advantage", True, False, step,
                    ego_alive, enemy_alive, ego_score, enemy_score)
            if advantage < -margin:
                return TeamOutcome(
                    "enemy", -1, 1, "ego_loss",
                    "timeout_tactical_advantage", True, False, step,
                    ego_alive, enemy_alive, ego_score, enemy_score)
            return TeamOutcome(
                None, 0, 0, "draw", "timeout_tactical_tie",
                True, False, step, ego_alive, enemy_alive,
                ego_score, enemy_score)
        return TeamOutcome(None, 0, 0, "draw", "timeout",
                           True, False, step, ego_alive, enemy_alive)
    return TeamOutcome(None, 0, 0, "ongoing", "ongoing",
                       False, False, step, ego_alive, enemy_alive)
