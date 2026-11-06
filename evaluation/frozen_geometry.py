"""Hash-bound, seed-replayable 2v2 initial-geometry sampling.

The legacy R3 scenario uses one fixed head-on geometry.  This module keeps
geometry randomization outside policy code and makes every sampled episode a
pure function of a frozen protocol, a geometry seed, and an explicit side
assignment.  Paired ``native``/``swapped`` episodes therefore share exactly
the same physical geometry while exchanging which side is controlled by the
candidate policy.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Real

import numpy as np

GEOMETRY_SCHEMA = "PBGRFrozenInitialGeometryV1"
INDEXED_GEOMETRY_SCHEMA = "PBGRIndexedInitialGeometryV1"
SIDE_ASSIGNMENTS = ("native", "swapped")
_EARTH_RADIUS_M = 6_371_000.0


def _canonical_sha256(payload: object) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _finite(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a real numeric scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _range(values: Sequence[float], name: str) -> tuple[float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError(f"{name} must contain [minimum, maximum]")
    low, high = (_finite(values[0], name), _finite(values[1], name))
    if high < low:
        raise ValueError(f"{name} maximum must be >= minimum")
    return low, high


def indexed_geometry_assignment(
    environment_seed: int,
    reset_index: int,
    side_policy: str = "uniform_native_swapped",
) -> tuple[int, str]:
    """Derive one R4 geometry assignment without consuming mutable RNG state.

    ``environment_seed`` identifies a vector slot and ``reset_index`` is that
    slot's persistent count of automatically sampled resets.  The mapping is
    deliberately hash based: recreating a vector environment after an
    interruption can restore only the next reset index and continue the exact
    geometry/side sequence without replaying the start of the RNG stream.
    """

    for value, name in (
        (environment_seed, "environment_seed"),
        (reset_index, "reset_index"),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(f"{name} must be an integer")
        if int(value) < 0:
            raise ValueError(f"{name} must be non-negative")
    policy = str(side_policy).strip().lower()
    if policy not in ("uniform_native_swapped", "paired_native_swapped"):
        raise ValueError("unsupported indexed geometry side policy")
    if policy == "paired_native_swapped":
        pair_index = int(reset_index) // 2
        key = (
            f"{INDEXED_GEOMETRY_SCHEMA}|paired|{int(environment_seed)}|"
            f"{pair_index}"
        ).encode()
        digest = hashlib.sha256(key).digest()
        geometry_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
        geometry_seed &= int(np.iinfo(np.int64).max)
        side_assignment = "native" if int(reset_index) % 2 == 0 else "swapped"
        return geometry_seed, side_assignment
    key = (
        f"{INDEXED_GEOMETRY_SCHEMA}|{int(environment_seed)}|{int(reset_index)}"
    ).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    geometry_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    geometry_seed &= int(np.iinfo(np.int64).max)
    side_assignment = "native" if digest[8] % 2 == 0 else "swapped"
    return geometry_seed, side_assignment


@dataclass(frozen=True)
class FrozenGeometryProtocol:
    protocol_id: str
    center_longitude_deg: float
    center_latitude_deg: float
    center_offset_north_m: tuple[float, float]
    center_offset_east_m: tuple[float, float]
    engagement_range_m: tuple[float, float]
    formation_spacing_m: tuple[float, float]
    altitude_ft: tuple[float, float]
    team_altitude_delta_ft: tuple[float, float]
    individual_altitude_jitter_ft: tuple[float, float]
    speed_fps: tuple[float, float]
    heading_jitter_deg: tuple[float, float]
    bearing_deg: tuple[float, float] = (0.0, 360.0)
    minimum_aircraft_separation_m: float = 500.0
    schema_version: str = GEOMETRY_SCHEMA
    paired_side_swap: bool = True

    def __post_init__(self) -> None:
        if self.schema_version != GEOMETRY_SCHEMA:
            raise ValueError("unsupported frozen geometry schema")
        protocol_id = str(self.protocol_id).strip()
        if not protocol_id:
            raise ValueError("protocol_id is required")
        if not isinstance(self.paired_side_swap, bool):
            raise ValueError("paired_side_swap must be boolean")
        if not self.paired_side_swap:
            raise ValueError("R4 requires paired native/swapped evaluation")
        object.__setattr__(self, "protocol_id", protocol_id)
        for name in (
            "center_offset_north_m",
            "center_offset_east_m",
            "engagement_range_m",
            "formation_spacing_m",
            "altitude_ft",
            "team_altitude_delta_ft",
            "individual_altitude_jitter_ft",
            "speed_fps",
            "heading_jitter_deg",
            "bearing_deg",
        ):
            object.__setattr__(self, name, _range(getattr(self, name), name))
        object.__setattr__(
            self,
            "center_longitude_deg",
            _finite(self.center_longitude_deg, "center_longitude_deg"),
        )
        latitude = _finite(self.center_latitude_deg, "center_latitude_deg")
        if not -89.0 <= latitude <= 89.0:
            raise ValueError("center_latitude_deg must lie in [-89, 89]")
        object.__setattr__(self, "center_latitude_deg", latitude)
        minimum = _finite(
            self.minimum_aircraft_separation_m,
            "minimum_aircraft_separation_m",
        )
        if minimum <= 0.0:
            raise ValueError("minimum_aircraft_separation_m must be positive")
        if self.engagement_range_m[0] <= minimum:
            raise ValueError("engagement range must exceed minimum separation")
        if self.formation_spacing_m[0] <= minimum:
            raise ValueError("formation spacing must exceed minimum separation")
        if self.altitude_ft[0] <= 0.0 or self.speed_fps[0] <= 0.0:
            raise ValueError("altitude and speed ranges must be positive")
        object.__setattr__(self, "minimum_aircraft_separation_m", minimum)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> FrozenGeometryProtocol:
        if not isinstance(payload, Mapping):
            raise ValueError("geometry protocol must be a mapping")
        expected = {field.name for field in cls.__dataclass_fields__.values()}
        if set(payload) != expected:
            raise ValueError("geometry protocol fields do not match the schema")
        return cls(**dict(payload))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def protocol_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    def sample(
        self,
        geometry_seed: int,
        side_assignment: str,
        ego_ids: Sequence[str],
        opponent_ids: Sequence[str],
    ) -> dict[str, object]:
        if isinstance(geometry_seed, (bool, np.bool_)) or not isinstance(
            geometry_seed, (int, np.integer)
        ):
            raise ValueError("geometry_seed must be an integer")
        if int(geometry_seed) < 0:
            raise ValueError("geometry_seed must be non-negative")
        side = str(side_assignment).strip().lower()
        if side not in SIDE_ASSIGNMENTS:
            raise ValueError(f"side_assignment must be one of {SIDE_ASSIGNMENTS}")
        ego = tuple(str(value) for value in ego_ids)
        opponent = tuple(str(value) for value in opponent_ids)
        if len(ego) != 2 or len(opponent) != 2 or len(set(ego + opponent)) != 4:
            raise ValueError("frozen R4 geometry requires two distinct 2-aircraft teams")

        rng = np.random.default_rng(int(geometry_seed))

        def uniform(bounds):
            return float(rng.uniform(bounds[0], bounds[1]))

        bearing = uniform(self.bearing_deg) % 360.0
        theta = math.radians(bearing)
        forward = np.asarray([math.cos(theta), math.sin(theta)], dtype=np.float64)
        lateral = np.asarray([-forward[1], forward[0]], dtype=np.float64)
        center = np.asarray(
            [
                uniform(self.center_offset_north_m),
                uniform(self.center_offset_east_m),
            ],
            dtype=np.float64,
        )
        engagement_range = uniform(self.engagement_range_m)
        spacing = uniform(self.formation_spacing_m)
        base_altitude = uniform(self.altitude_ft)
        altitude_delta = uniform(self.team_altitude_delta_ft)

        native_centers = (
            center - 0.5 * engagement_range * forward,
            center + 0.5 * engagement_range * forward,
        )
        native_headings = (bearing, (bearing + 180.0) % 360.0)
        native_altitudes = (
            base_altitude - 0.5 * altitude_delta,
            base_altitude + 0.5 * altitude_delta,
        )
        # Draw slot perturbations in physical-role order, not policy-team
        # order.  Replaying the seed with ``swapped`` then preserves the exact
        # four physical initial conditions and only exchanges their owners.
        native_states = []
        for native_index in range(2):
            team_center = native_centers[native_index]
            heading = native_headings[native_index]
            altitude = native_altitudes[native_index]
            slot_states = []
            for slot in range(2):
                sign = -1.0 if slot == 0 else 1.0
                north_m, east_m = team_center + sign * 0.5 * spacing * lateral
                lon, lat = _offset_to_lon_lat(
                    north_m,
                    east_m,
                    self.center_longitude_deg,
                    self.center_latitude_deg,
                )
                slot_states.append(
                    {
                        "ic_long_gc_deg": lon,
                        "ic_lat_geod_deg": lat,
                        "ic_h_sl_ft": altitude + uniform(self.individual_altitude_jitter_ft),
                        "ic_psi_true_deg": (heading + uniform(self.heading_jitter_deg)) % 360.0,
                        "ic_u_fps": uniform(self.speed_fps),
                    }
                )
            native_states.append(tuple(slot_states))

        if side == "native":
            assigned = ((ego, 0), (opponent, 1))
        else:
            assigned = ((ego, 1), (opponent, 0))
        states: dict[str, dict[str, float]] = {}
        for team_ids, native_index in assigned:
            for slot, aircraft_id in enumerate(team_ids):
                states[aircraft_id] = dict(native_states[native_index][slot])

        payload = {
            "schema_version": "PBGRFrozenGeometrySampleV1",
            "protocol_id": self.protocol_id,
            "protocol_sha256": self.protocol_sha256,
            "geometry_seed": int(geometry_seed),
            "side_assignment": side,
            "engagement_range_m": engagement_range,
            "formation_spacing_m": spacing,
            "bearing_deg": bearing,
            "aircraft_states": states,
        }
        payload["sample_sha256"] = _canonical_sha256(payload)
        return payload


def _offset_to_lon_lat(
    north_m: float,
    east_m: float,
    center_longitude_deg: float,
    center_latitude_deg: float,
) -> tuple[float, float]:
    latitude = center_latitude_deg + math.degrees(north_m / _EARTH_RADIUS_M)
    longitude = center_longitude_deg + math.degrees(
        east_m / (_EARTH_RADIUS_M * math.cos(math.radians(center_latitude_deg)))
    )
    return float(longitude), float(latitude)


def apply_frozen_geometry(env, sample: Mapping[str, object]) -> None:
    """Install one already-sampled geometry before ``env.reset()``."""

    setter = getattr(env, "set_frozen_initial_geometry", None)
    if setter is not None:
        setter(dict(sample))
        return
    states = sample.get("aircraft_states")
    if not isinstance(states, Mapping):
        raise ValueError("geometry sample lacks aircraft_states")
    if set(states) != set(env.agents):
        raise ValueError("geometry sample aircraft IDs do not match the environment")
    for aircraft_id, state in states.items():
        if not isinstance(state, Mapping):
            raise ValueError("aircraft geometry state must be a mapping")
        merged = dict(env.agents[aircraft_id].init_state)
        merged.update({str(key): float(value) for key, value in state.items()})
        env.agents[aircraft_id].init_state = merged
