"""Fixed-slot, mask-explicit entity observations for multi-aircraft PBGR.

The adapter deliberately exposes existence, last-observed life, visibility,
observation age and slot reset as separate signals. Hidden simulator state is
never read into policy inputs: an invisible entity retains only its cached
last observation.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Tuple

import numpy as np

from .provenance import strict_nonnegative_int


class EntityKind(IntEnum):
    EGO = 0
    TEAMMATE = 1
    ENEMY = 2
    MISSILE = 3


ENTITY_FEATURE_NAMES = (
    "relative_north_10km", "relative_east_10km", "relative_up_10km",
    "relative_v_north_mach", "relative_v_east_mach", "relative_v_up_mach",
    "range_10km", "speed_mach",
    # Private resources are zero when unknown.  The paired observed bit makes
    # that value distinguishable from a genuine zero without leaking the
    # simulator's exact teammate/enemy health or ammunition.
    "health_fraction", "health_observed",
    "ammo_fraction", "ammo_observed",
    "cooldown_fraction", "cooldown_observed",
    "shoot_legal", "shoot_legal_observed",
    "is_incoming_missile", "is_ego",
    # Exact own-aircraft flight state.  These values are populated only on
    # the ego row; every teammate, enemy, and missile row remains zero so the
    # extension cannot reveal privileged opponent state.
    "ego_altitude_5km", "ego_roll_sin", "ego_roll_cos",
    "ego_pitch_sin", "ego_pitch_cos",
    "ego_heading_sin", "ego_heading_cos",
    "ego_body_u_mach", "ego_body_v_mach", "ego_body_w_mach",
    "ego_airspeed_mach",
)

LEGACY_ENTITY_FEATURE_DIM = 18
EGO_FLIGHT_FEATURE_NAMES = ENTITY_FEATURE_NAMES[LEGACY_ENTITY_FEATURE_DIM:]


class EntityCapacityError(ValueError):
    """Raised before a real entity could be hidden by fixed-slot truncation."""


@dataclass
class EntityObservation(object):
    """One agent's fixed-slot entity view.

    ``alive_mask`` is the last *observed* life state, never omniscient truth.
    ``alive_observed_mask`` tells the learner whether that value was refreshed
    this step. ``age`` is time since the last observation and ``slot_reset``
    marks a new identity entering a slot.
    """

    features: np.ndarray
    entity_type: np.ndarray
    entity_mask: np.ndarray
    alive_mask: np.ndarray
    alive_observed_mask: np.ndarray
    visible_mask: np.ndarray
    age: np.ndarray
    slot_reset: np.ndarray
    entity_ids: Tuple[str, ...]

    def __post_init__(self):
        self.features = np.asarray(self.features, dtype=np.float32)
        self.entity_type = np.asarray(self.entity_type, dtype=np.int64)
        self.entity_mask = np.asarray(self.entity_mask, dtype=np.float32)
        self.alive_mask = np.asarray(self.alive_mask, dtype=np.float32)
        self.alive_observed_mask = np.asarray(self.alive_observed_mask, dtype=np.float32)
        self.visible_mask = np.asarray(self.visible_mask, dtype=np.float32)
        self.age = np.asarray(self.age, dtype=np.float32)
        self.slot_reset = np.asarray(self.slot_reset, dtype=np.float32)
        if self.features.ndim != 2:
            raise ValueError("features must have shape [entities, features]")
        slots = self.features.shape[0]
        for name in ("entity_type", "entity_mask", "alive_mask",
                     "alive_observed_mask", "visible_mask", "age", "slot_reset"):
            if getattr(self, name).shape != (slots,):
                raise ValueError("{0} must have shape [entities]".format(name))
        if len(self.entity_ids) != slots:
            raise ValueError("entity_ids must contain one id per slot")
        if not np.all(np.isfinite(self.features)) or not np.all(np.isfinite(self.age)):
            raise ValueError("entity observations must be finite")
        for name in ("entity_mask", "alive_mask", "alive_observed_mask",
                     "visible_mask", "slot_reset"):
            values = getattr(self, name)
            if not np.all((values == 0.0) | (values == 1.0)):
                raise ValueError("{0} must be binary".format(name))
        if (np.any(self.alive_mask > self.entity_mask) or
                np.any(self.visible_mask > self.entity_mask) or
                np.any(self.alive_observed_mask > self.visible_mask)):
            raise ValueError("life/visibility masks violate the entity contract")
        if np.any(self.features[self.entity_mask == 0.0] != 0.0):
            raise ValueError("padded feature rows must be zero")

    @property
    def num_entities(self):
        return int(self.features.shape[0])

    @property
    def feature_dim(self):
        return int(self.features.shape[1])

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "features": self.features.copy(),
            "entity_type": self.entity_type.copy(),
            "entity_mask": self.entity_mask.copy(),
            "alive_mask": self.alive_mask.copy(),
            "alive_observed_mask": self.alive_observed_mask.copy(),
            "visible_mask": self.visible_mask.copy(),
            "age": self.age.copy(),
            "slot_reset": self.slot_reset.copy(),
        }


class FlatEntityCodec(object):
    """Lossless fixed-width bridge between Gym Box rows and entity tensors.

    Each slot is encoded as ``features, entity_type, entity_mask,
    alive_mask, alive_observed_mask, visible_mask, age, slot_reset``.  The
    per-slot layout keeps a WSL subprocess vector environment on ordinary
    NumPy arrays while retaining every mask required by PBGR.
    """

    metadata_width = 7

    def __init__(self, num_entities, feature_dim=len(ENTITY_FEATURE_NAMES)):
        self.num_entities = strict_nonnegative_int(num_entities, "num_entities")
        self.feature_dim = strict_nonnegative_int(feature_dim, "feature_dim")
        if self.num_entities == 0 or self.feature_dim == 0:
            raise ValueError("entity codec dimensions must be positive")
        self.slot_width = self.feature_dim + self.metadata_width
        self.flat_dim = self.num_entities * self.slot_width

    def pack(self, observation):
        if not isinstance(observation, EntityObservation):
            raise TypeError("observation must be an EntityObservation")
        if (observation.num_entities != self.num_entities or
                observation.feature_dim != self.feature_dim):
            raise ValueError("entity observation shape does not match codec")
        rows = np.column_stack((
            observation.features,
            observation.entity_type.astype(np.float32),
            observation.entity_mask,
            observation.alive_mask,
            observation.alive_observed_mask,
            observation.visible_mask,
            observation.age,
            observation.slot_reset,
        )).astype(np.float32, copy=False)
        return rows.reshape(self.flat_dim).copy()

    def unpack_array(self, values):
        array = np.asarray(values, dtype=np.float32)
        if array.shape[-1] != self.flat_dim:
            raise ValueError("flat entity observation width mismatch")
        rows = array.reshape(array.shape[:-1] + (self.num_entities, self.slot_width))
        offset = self.feature_dim
        return {
            "features": rows[..., :offset],
            "entity_type": rows[..., offset].astype(np.int64),
            "entity_mask": rows[..., offset + 1],
            "alive_mask": rows[..., offset + 2],
            "alive_observed_mask": rows[..., offset + 3],
            "visible_mask": rows[..., offset + 4],
            "age": rows[..., offset + 5],
            "slot_reset": rows[..., offset + 6],
        }


class JSBSimEntityAdapter(object):
    """Build a deterministic fixed-slot entity view from a JSBSim env.

    Slot order is ego, UID-sorted teammates, UID-sorted enemies, then stable
    incoming-missile slots. Aircraft never move between slots merely because
    range or life state changes.
    """

    feature_dim = len(ENTITY_FEATURE_NAMES)

    def __init__(self, max_teammates, max_enemies, max_missiles,
                 visibility_range_m=np.inf, missile_age_scale_s=60.0):
        if float(visibility_range_m) <= 0.0 or float(missile_age_scale_s) <= 0.0:
            raise ValueError("range and age scale must be positive")
        self.max_teammates = strict_nonnegative_int(
            max_teammates, "max_teammates")
        self.max_enemies = strict_nonnegative_int(
            max_enemies, "max_enemies")
        self.max_missiles = strict_nonnegative_int(
            max_missiles, "max_missiles")
        self.visibility_range_m = float(visibility_range_m)
        self.missile_age_scale_s = float(missile_age_scale_s)
        self._observer_cache = {}

    @property
    def num_entities(self):
        return 1 + self.max_teammates + self.max_enemies + self.max_missiles

    @staticmethod
    def _uid(sim):
        return str(getattr(sim, "uid", ""))

    @staticmethod
    def _vector(env, sim, method):
        getter = getattr(env, "get_simulator_vector", None)
        if getter is not None:
            return getter(sim, method)
        value = np.asarray(getattr(sim, method)(), dtype=np.float64)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError("{0} must return a finite 3-vector".format(method))
        return value

    def _distance(self, env, ego, sim):
        return float(np.linalg.norm(
            self._vector(env, sim, "get_position")
            - self._vector(env, ego, "get_position")
        ))

    def reset(self, agent_id=None):
        """Clear last-known observations at an episode boundary."""
        if agent_id is None:
            self._observer_cache.clear()
        else:
            self._observer_cache.pop(str(agent_id), None)

    @staticmethod
    def _initial_ammunition(sim):
        try:
            return strict_nonnegative_int(
                getattr(sim, "num_missiles", 0),
                "num_missiles for {0}".format(JSBSimEntityAdapter._uid(sim)),
            )
        except ValueError as error:
            raise EntityCapacityError(
                str(error)
            ) from error

    def validate_environment_capacity(self, env):
        """Fail if any observer can outgrow the configured fixed slots.

        The incoming-missile requirement is computed per observer as the
        total initial ammunition carried by that observer's opponents.  This
        is deliberately not the sum of both teams' ammunition.
        """
        requirements = {}
        for agent_id, agent in env.agents.items():
            teammates = tuple(getattr(agent, "partners", ()))
            enemies = tuple(getattr(agent, "enemies", ()))
            opponent_ammunition = sum(
                self._initial_ammunition(enemy) for enemy in enemies
            )
            required = {
                "teammates": len(teammates),
                "enemies": len(enemies),
                "incoming_missiles": opponent_ammunition,
            }
            capacities = {
                "teammates": self.max_teammates,
                "enemies": self.max_enemies,
                "incoming_missiles": self.max_missiles,
            }
            shortages = [
                "{0}={1} < required_{0}={2}".format(
                    name, capacities[name], required[name])
                for name in ("teammates", "enemies", "incoming_missiles")
                if capacities[name] < required[name]
            ]
            if shortages:
                raise EntityCapacityError(
                    "PBGR entity capacity is insufficient for observer "
                    "{0}: {1}; incoming requirement equals opponent total "
                    "initial ammunition".format(
                        agent_id, "; ".join(shortages)))
            requirements[str(agent_id)] = required
        return requirements

    def _new_cache(self):
        return {
            "slot_ids": [""] * self.num_entities,
            "features": np.zeros((self.num_entities, self.feature_dim), dtype=np.float32),
            "alive": np.zeros(self.num_entities, dtype=np.float32),
            "last_seen_step": np.full(self.num_entities, -1, dtype=np.int64),
            "missile_ids": [""] * self.max_missiles,
            "missile_objects": [None] * self.max_missiles,
        }

    def _stable_missiles(self, env, ego):
        missiles = {self._uid(missile): missile
                    for missile in getattr(ego, "under_missiles", ())}
        cache = self._observer_cache.setdefault(self._uid(ego), self._new_cache())
        assigned = cache["missile_ids"]
        assigned_objects = cache["missile_objects"]

        # Discovery is an observation event, not a query of simulator truth.
        # Check geometry first and inspect life state only for missiles that are
        # currently inside the sensor envelope. An unseen far-away missile must
        # therefore have no policy-visible effect whether it is alive or dead.
        discovered = []
        for uid, missile in missiles.items():
            if uid in assigned:
                continue
            if self._distance(env, ego, missile) > self.visibility_range_m:
                continue
            if bool(getattr(missile, "is_alive", False)):
                discovered.append(uid)

        # A discovered UID owns its slot for the rest of the episode. Do not
        # free or replace it merely because the missile leaves sensor range,
        # dies outside coverage, or disappears from the simulator collection.
        new_ids = sorted(discovered)
        available = sum(not uid for uid in assigned)
        if len(new_ids) > available:
            raise EntityCapacityError(
                "incoming missile slots exhausted for observer {0}: "
                "capacity={1}, occupied={2}, newly_discovered={3}".format(
                    self._uid(ego), self.max_missiles,
                    self.max_missiles - available, new_ids))
        for index, uid in enumerate(assigned):
            if not uid and new_ids:
                discovered_uid = new_ids.pop(0)
                assigned[index] = discovered_uid
                assigned_objects[index] = missiles[discovered_uid]
        return list(assigned_objects)

    def _aircraft_features(self, env, ego, sim, is_ego):
        position = self._vector(env, sim, "get_position")
        velocity = self._vector(env, sim, "get_velocity")
        ego_position = self._vector(env, ego, "get_position")
        ego_velocity = self._vector(env, ego, "get_velocity")
        relative_position = (position - ego_position) / 10000.0
        relative_velocity = (velocity - ego_velocity) / 340.0
        distance = np.linalg.norm(position - ego_position) / 10000.0
        speed = np.linalg.norm(velocity) / 340.0
        health = 0.0
        health_observed = 0.0
        ammo_fraction = 0.0
        ammo_observed = 0.0
        cooldown_fraction = 0.0
        cooldown_observed = 0.0
        shoot_legal = 0.0
        shoot_legal_observed = 0.0
        ego_flight = np.zeros(len(EGO_FLIGHT_FEATURE_NAMES), dtype=np.float32)
        if is_ego:
            # An actor may inspect only its own exact resource state.  Team and
            # enemy rows intentionally do not even read these simulator fields.
            health = np.clip(
                float(getattr(sim, "bloods", 0.0)) / 100.0, 0.0, 1.0)
            health_observed = 1.0
            # Exact weapon resources are an explicit PBGR-task capability;
            # merely running the historical shoot task must not enable them.
            get_weapon_state = getattr(
                getattr(env, "task", None), "get_weapon_action_state", None)
            if get_weapon_state is not None:
                weapon_state = get_weapon_state(env, self._uid(sim))
                ammo_fraction = float(weapon_state["ammo_fraction"])
                cooldown_fraction = float(weapon_state["cooldown_fraction"])
                shoot_legal = float(bool(weapon_state["shoot_legal"]))
                ammo_observed = 1.0
                cooldown_observed = 1.0
                shoot_legal_observed = 1.0
            ego_flight = self._ego_flight_features(env, sim)
        return np.asarray([*relative_position, *relative_velocity, distance, speed,
                           health, health_observed,
                           ammo_fraction, ammo_observed,
                           cooldown_fraction, cooldown_observed,
                           shoot_legal, shoot_legal_observed,
                           0.0, float(is_ego), *ego_flight], dtype=np.float32)

    @staticmethod
    def _ego_flight_features(env, sim):
        """Return eleven non-privileged ego flight-state features.

        Real JSBSim aircraft expose these values through the property catalog.
        Lightweight test doubles may only expose world position/velocity; the
        deterministic fallback preserves the contract without reading any
        teammate or opponent state.
        """

        try:
            from envs.JSBSim.core.catalog import Catalog as c

            props = [
                c.position_h_sl_m,
                c.attitude_roll_rad,
                c.attitude_pitch_rad,
                c.attitude_heading_true_rad,
                c.velocities_u_mps,
                c.velocities_v_mps,
                c.velocities_w_mps,
                c.velocities_vc_mps,
            ]
            getter = getattr(env, "get_agent_property_values", None)
            values = (
                getter(sim.uid, props)
                if getter is not None
                else sim.get_property_values(props)
            )
            state = np.asarray(values, dtype=np.float64)
            if state.shape != (8,) or not np.all(np.isfinite(state)):
                raise ValueError("invalid JSBSim ego flight state")
            altitude, roll, pitch, heading, body_u, body_v, body_w, airspeed = state
        except (AttributeError, TypeError):
            position = JSBSimEntityAdapter._vector(env, sim, "get_position")
            velocity = JSBSimEntityAdapter._vector(env, sim, "get_velocity")
            altitude = position[2]
            roll = pitch = heading = 0.0
            body_u, body_v, body_w = velocity
            airspeed = float(np.linalg.norm(velocity))
        return np.asarray(
            [
                altitude / 5000.0,
                np.sin(roll),
                np.cos(roll),
                np.sin(pitch),
                np.cos(pitch),
                np.sin(heading),
                np.cos(heading),
                body_u / 340.0,
                body_v / 340.0,
                body_w / 340.0,
                airspeed / 340.0,
            ],
            dtype=np.float32,
        )

    def _missile_features(self, env, ego, missile):
        position = self._vector(env, missile, "get_position")
        velocity = self._vector(env, missile, "get_velocity")
        ego_position = self._vector(env, ego, "get_position")
        ego_velocity = self._vector(env, ego, "get_velocity")
        return np.asarray([
            *((position - ego_position) / 10000.0),
            *((velocity - ego_velocity) / 340.0),
            np.linalg.norm(position - ego_position) / 10000.0,
            np.linalg.norm(velocity) / 340.0,
            # Missile life/visibility is represented by the explicit masks;
            # aircraft-only private resource fields remain unknown here.
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            1.0, 0.0,
            *([0.0] * len(EGO_FLIGHT_FEATURE_NAMES)),
        ], dtype=np.float32)

    def build(self, env, agent_id):
        if agent_id not in env.agents:
            raise KeyError("unknown agent_id: {0}".format(agent_id))
        ego = env.agents[agent_id]
        teammates = sorted(getattr(ego, "partners", ()), key=self._uid)
        enemies = sorted(getattr(ego, "enemies", ()), key=self._uid)
        if len(teammates) > self.max_teammates:
            raise EntityCapacityError(
                "teammate slots exhausted for observer {0}: capacity={1}, "
                "actual={2}".format(
                    self._uid(ego), self.max_teammates, len(teammates)))
        if len(enemies) > self.max_enemies:
            raise EntityCapacityError(
                "enemy slots exhausted for observer {0}: capacity={1}, "
                "actual={2}".format(
                    self._uid(ego), self.max_enemies, len(enemies)))
        missiles = self._stable_missiles(env, ego)
        teammate_slots = list(teammates[:self.max_teammates])
        teammate_slots += [None] * (self.max_teammates - len(teammate_slots))
        enemy_slots = list(enemies[:self.max_enemies])
        enemy_slots += [None] * (self.max_enemies - len(enemy_slots))
        slots = [(ego, EntityKind.EGO)]
        slots += [(sim, EntityKind.TEAMMATE) for sim in teammate_slots]
        slots += [(sim, EntityKind.ENEMY) for sim in enemy_slots]
        slots += [(sim, EntityKind.MISSILE) for sim in missiles]

        observer_id = self._uid(ego)
        cache = self._observer_cache.setdefault(observer_id, self._new_cache())
        features = np.zeros((self.num_entities, self.feature_dim), dtype=np.float32)
        entity_type = np.zeros(self.num_entities, dtype=np.int64)
        entity_mask = np.zeros(self.num_entities, dtype=np.float32)
        alive_mask = np.zeros(self.num_entities, dtype=np.float32)
        alive_observed_mask = np.zeros(self.num_entities, dtype=np.float32)
        visible_mask = np.zeros(self.num_entities, dtype=np.float32)
        age = np.ones(self.num_entities, dtype=np.float32)
        slot_reset = np.zeros(self.num_entities, dtype=np.float32)
        entity_ids = []
        current_step = int(getattr(env, "current_step", 0))
        time_interval = float(getattr(env, "time_interval", 1.0))
        for index, (sim, kind) in enumerate(slots):
            entity_type[index] = int(kind)
            uid = "" if sim is None else self._uid(sim)
            entity_ids.append(uid)
            identity_changed = uid != cache["slot_ids"][index]
            slot_reset[index] = float(identity_changed)
            if identity_changed:
                cache["slot_ids"][index] = uid
                cache["features"][index].fill(0.0)
                cache["alive"][index] = 0.0
                cache["last_seen_step"][index] = -1
            if sim is None:
                continue
            entity_mask[index] = 1.0
            # Detection range uses geometry only. Crucially, life state is not
            # consulted until the entity is observable.
            distance = self._distance(env, ego, sim)
            visible = bool(kind == EntityKind.EGO or
                           distance <= self.visibility_range_m)
            visible_mask[index] = float(visible)
            if visible:
                alive = bool(getattr(sim, "is_alive", False))
                alive_observed_mask[index] = 1.0
                cache["alive"][index] = float(alive)
                cache["last_seen_step"][index] = current_step
                if kind == EntityKind.MISSILE:
                    row = self._missile_features(env, ego, sim)
                else:
                    row = self._aircraft_features(env, ego, sim, kind == EntityKind.EGO)
                cache["features"][index] = np.clip(row, -10.0, 10.0)
            features[index] = cache["features"][index]
            alive_mask[index] = cache["alive"][index]
            last_seen = int(cache["last_seen_step"][index])
            if last_seen >= 0:
                age[index] = np.clip((current_step - last_seen) * time_interval /
                                     self.missile_age_scale_s, 0.0, 1.0)
        return EntityObservation(
            features, entity_type, entity_mask, alive_mask,
            alive_observed_mask, visible_mask, age, slot_reset,
            tuple(entity_ids))
