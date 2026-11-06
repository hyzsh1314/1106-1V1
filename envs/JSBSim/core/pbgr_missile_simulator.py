"""Numerically guarded missile simulator for the opt-in PBGR protocol."""

import numpy as np

from .simulatior import AircraftSimulator, MissileSimulator


class PBGRMissileSimulator(MissileSimulator):
    """Preserve terminal hits and fail closed on degenerate guidance states.

    The historical :class:`MissileSimulator` is deliberately untouched.  This
    subclass is instantiated only by the PBGR-specific task so legacy scenario
    trajectories retain their original implementation semantics.
    """

    _EPS = 1e-8

    @classmethod
    def create(cls, parent: AircraftSimulator, target: AircraftSimulator,
               uid: str, missile_model: str = "simple_kinematic_PN_R1"):
        assert parent.dt == target.dt, "integration timestep must be same!"
        missile = cls(uid, parent.color, missile_model, parent.dt)
        missile.launch(parent)
        missile.target(target)
        return missile

    def _set_status(self, status):
        # MissileSimulator keeps its state private.  The opt-in subclass uses
        # the base class's name-mangled storage so inherited status properties
        # and rendering continue to work without changing the legacy class.
        self._MissileSimulator__status = status

    def run(self, defer_target_effect=False, target_segment_start=None):
        """Use the shared absorbing, segment-capture and deferred-effect flow."""

        return super().run(
            defer_target_effect=defer_target_effect,
            target_segment_start=target_segment_start,
        )

    def _guidance(self):
        """Return finite proportional-navigation commands or neutral control."""
        missile_position = np.asarray(
            self.get_position(), dtype=np.float64)
        missile_velocity = np.asarray(
            self.get_velocity(), dtype=np.float64)
        target_position = np.asarray(
            self.target_aircraft.get_position(), dtype=np.float64)
        target_velocity = np.asarray(
            self.target_aircraft.get_velocity(), dtype=np.float64)
        values = np.concatenate((
            missile_position, missile_velocity,
            target_position, target_velocity,
        ))
        if not np.all(np.isfinite(values)):
            return np.zeros(2, dtype=np.float64), np.inf

        relative = target_position - missile_position
        r_xy = float(np.linalg.norm(relative[:2]))
        distance = float(np.linalg.norm(relative))
        speed = float(np.linalg.norm(missile_velocity))
        if (speed <= self._EPS or r_xy <= self._EPS or
                distance <= self._EPS):
            return np.zeros(2, dtype=np.float64), distance

        theta_m = np.arcsin(np.clip(
            missile_velocity[2] / speed, -1.0, 1.0))
        relative_velocity = target_velocity - missile_velocity
        dbeta = (
            relative_velocity[1] * relative[0] -
            relative_velocity[0] * relative[1]
        ) / (r_xy ** 2)
        deps = (
            relative_velocity[2] * (r_xy ** 2) -
            relative[2] * np.dot(relative[:2], relative_velocity[:2])
        ) / ((distance ** 2) * r_xy)
        ny = self.K * speed / self._g * np.cos(theta_m) * dbeta
        nz = self.K * speed / self._g * deps + np.cos(theta_m)
        action = np.asarray([ny, nz], dtype=np.float64)
        if not np.all(np.isfinite(action)):
            action = np.zeros(2, dtype=np.float64)
        return np.clip(
            action, -self._nyz_max, self._nyz_max), distance


class PBGRGenericWVRMissileSimulator(PBGRMissileSimulator):
    """Versioned, opt-in generic WVR point-mass missile.

    This model is intentionally not named after a real weapon.  Its public
    parameters are part of the evidence contract, unknown model identifiers
    fail closed, and capture is evaluated along the relative integration
    segment to avoid tunnelling through a small capture radius.
    """

    MODEL_ID = "PBGR_GENERIC_WVR_PN_R2"
    MODEL_REGISTRY = {
        MODEL_ID: {
            "navigation_constant": 3.0,
            "maximum_normal_load_g": 30.0,
            "thrust_duration_s": 3.0,
            "maximum_flight_time_s": 40.0,
            "minimum_speed_mps": 150.0,
            "capture_radius_m": 50.0,
        },
    }

    @classmethod
    def create(cls, parent: AircraftSimulator, target: AircraftSimulator,
               uid: str, missile_model: str = MODEL_ID,
               parameters=None):
        assert parent.dt == target.dt, "integration timestep must be same!"
        model = str(missile_model)
        if model not in cls.MODEL_REGISTRY:
            raise ValueError("unknown V23 missile model: {0}".format(model))
        values = dict(cls.MODEL_REGISTRY[model])
        if parameters is not None:
            unknown = set(parameters) - set(values)
            if unknown:
                raise ValueError("unknown V23 missile parameters: {0}".format(
                    sorted(unknown)))
            values.update({key: float(value) for key, value in parameters.items()})
        cls._validate_parameters(values)
        missile = cls(uid, parent.color, model, parent.dt)
        missile._apply_parameters(values)
        missile.launch(parent)
        missile.target(target)
        missile.resolution_reason = "pending"
        missile.resolution_role = "pending"
        missile.capture_fraction = None
        missile.closest_approach_m = None
        missile._deferred_target_effect = False
        return missile

    @staticmethod
    def _validate_parameters(values):
        positive = (
            "navigation_constant", "maximum_normal_load_g",
            "maximum_flight_time_s", "minimum_speed_mps", "capture_radius_m",
        )
        for name in positive:
            value = float(values[name])
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("{0} must be positive and finite".format(name))
        thrust = float(values["thrust_duration_s"])
        if not np.isfinite(thrust) or thrust < 0.0:
            raise ValueError("thrust_duration_s must be finite and non-negative")
        if thrust > float(values["maximum_flight_time_s"]):
            raise ValueError("thrust duration cannot exceed maximum flight time")

    def _apply_parameters(self, values):
        self._K = float(values["navigation_constant"])
        self._nyz_max = float(values["maximum_normal_load_g"])
        self._t_thrust = float(values["thrust_duration_s"])
        self._t_max = float(values["maximum_flight_time_s"])
        self._v_min = float(values["minimum_speed_mps"])
        self._Rc = float(values["capture_radius_m"])
        self.model_parameters = dict(values)

    @property
    def K(self):
        """Use the registered constant PN gain; do not decay it with age."""

        return float(self._K)

    @staticmethod
    def relative_segment_closest_approach(relative_start, relative_end):
        """Return minimum distance on a linearly interpolated relative segment."""

        distance, _ = PBGRGenericWVRMissileSimulator.relative_segment_metrics(
            relative_start, relative_end)
        return distance

    @staticmethod
    def relative_segment_metrics(relative_start, relative_end):
        """Return closest distance and its fraction along a relative segment."""

        start = np.asarray(relative_start, dtype=np.float64)
        end = np.asarray(relative_end, dtype=np.float64)
        if start.shape != (3,) or end.shape != (3,):
            raise ValueError("relative segments must be three-dimensional")
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
            return np.inf, 0.0
        delta = end - start
        denominator = float(np.dot(delta, delta))
        if denominator <= 1e-16:
            return float(np.linalg.norm(start)), 0.0
        fraction = float(np.clip(-np.dot(start, delta) / denominator, 0.0, 1.0))
        return float(np.linalg.norm(start + fraction * delta)), fraction

    def _finish(self, status, reason):
        self._set_status(status)
        self.resolution_reason = str(reason)

    def _capture(self, target, reason, fraction, closest, defer_target_effect):
        self._finish(MissileSimulator.HIT, reason)
        self.capture_fraction = float(fraction)
        self.closest_approach_m = float(closest)
        if defer_target_effect:
            self._deferred_target_effect = True
            return
        self.resolution_role = "kill"
        target.shotdown()

    @classmethod
    def resolve_deferred_target_effects(cls, missiles):
        """Apply one synchronized kill per target after every missile advances.

        Every missile that geometrically captures the target remains a hit.
        Kill attribution uses continuous capture fraction, then closest
        approach, with immutable uid only as the exact-tie breaker.  It is
        therefore independent of the temporary-simulator insertion order.
        """

        grouped = {}
        for missile in missiles:
            if not bool(getattr(missile, "_deferred_target_effect", False)):
                continue
            target = getattr(missile, "target_aircraft", None)
            if target is None:
                missile._deferred_target_effect = False
                continue
            grouped.setdefault(id(target), (target, []))[1].append(missile)

        for target, captures in grouped.values():
            ranked = sorted(captures, key=lambda missile: (
                float(getattr(missile, "capture_fraction", np.inf)),
                float(getattr(missile, "closest_approach_m", np.inf)),
                str(missile.uid),
            ))
            winner = ranked[0] if bool(getattr(target, "is_alive", False)) else None
            if winner is not None:
                winner.resolution_role = "kill"
                target.shotdown()
            for missile in ranked:
                missile._deferred_target_effect = False
                if missile is winner:
                    continue
                missile.resolution_role = "redundant_hit"
                missile.resolution_reason = "simultaneous_redundant_capture"

    def run(self, defer_target_effect=False, target_segment_start=None):
        if self.is_done:
            return
        target = self.target_aircraft
        if target is None:
            self._finish(MissileSimulator.MISS, "missing_target")
            return
        if not target.is_alive:
            self._finish(MissileSimulator.MISS, "target_already_resolved")
            return

        missile_start = np.asarray(self.get_position(), dtype=np.float64).copy()
        target_current = np.asarray(target.get_position(), dtype=np.float64).copy()
        target_velocity = np.asarray(target.get_velocity(), dtype=np.float64).copy()
        # V23 supplies the exact pre-aircraft position for synchronized
        # environment stepping. Direct simulator users retain a guarded
        # constant-velocity fallback for backward compatibility.
        if target_segment_start is None:
            target_start = target_current - target_velocity * float(self.dt)
        else:
            target_start = np.asarray(
                target_segment_start, dtype=np.float64).copy()
            if target_start.shape != (3,) or not np.all(np.isfinite(target_start)):
                self._finish(MissileSimulator.MISS, "invalid_target_segment")
                return
        relative_start = target_start - missile_start
        start_distance = float(np.linalg.norm(relative_start))
        if start_distance <= self._Rc:
            self._capture(
                target, "capture", 0.0, start_distance,
                bool(defer_target_effect))
            return

        self._t += self.dt
        action, distance = self._guidance()
        action = np.asarray(action, dtype=np.float64)
        if not np.isfinite(distance) or not np.all(np.isfinite(action)):
            self._finish(MissileSimulator.MISS, "non_finite_guidance")
            return
        self._distance_increment.append(distance > self._distance_pre)
        self._distance_pre = distance

        speed = float(np.linalg.norm(self.get_velocity()))
        if (self._t > self._t_max or not np.isfinite(speed) or
                speed < self._v_min or
                np.sum(self._distance_increment) >= self._distance_increment.maxlen):
            self._finish(MissileSimulator.MISS, "expired")
            return

        pitch = float(self.get_rpy()[1])
        if not np.isfinite(pitch) or abs(float(np.cos(pitch))) <= self._EPS:
            self._finish(MissileSimulator.MISS, "degenerate_posture")
            return
        previous_position = self.get_position().copy()
        previous_velocity = self.get_velocity().copy()
        previous_posture = self.get_rpy().copy()
        previous_geodetic = self.get_geodetic().copy()
        previous_mass = self._m
        previous_dtheta, previous_dphi = self._dtheta, self._dphi
        self._state_trans(action)
        finite_state = (
            np.all(np.isfinite(self.get_position())) and
            np.all(np.isfinite(self.get_velocity())) and
            np.all(np.isfinite(self.get_rpy())) and
            np.all(np.isfinite(self.get_geodetic()))
        )
        if not finite_state:
            self._position[:] = previous_position
            self._velocity[:] = previous_velocity
            self._posture[:] = previous_posture
            self._geodetic[:] = previous_geodetic
            self._m = previous_mass
            self._dtheta, self._dphi = previous_dtheta, previous_dphi
            self._finish(MissileSimulator.MISS, "non_finite_state")
            return

        relative_end = target_current - np.asarray(
            self.get_position(), dtype=np.float64)
        closest, fraction = self.relative_segment_metrics(
            relative_start, relative_end)
        if closest <= self._Rc and target.is_alive:
            self._capture(
                target, "segment_capture", fraction, closest,
                bool(defer_target_effect))
            return
        if not target.is_alive:
            self._finish(MissileSimulator.MISS, "target_already_resolved")
