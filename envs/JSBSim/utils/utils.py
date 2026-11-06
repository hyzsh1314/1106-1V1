import os
from functools import lru_cache

import yaml
import pymap3d
import numpy as np


_WGS84_A_M = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_B_M = _WGS84_A_M * (1.0 - _WGS84_F)
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)
_WGS84_EP2 = (_WGS84_A_M ** 2 - _WGS84_B_M ** 2) / _WGS84_B_M ** 2


class WGS84LocalFrame:
    """Cached WGS-84 geodetic <-> local north/east/up transform.

    The reference implementation used :mod:`pymap3d` for every conversion.
    This object caches the origin ECEF coordinates and the local rotation
    terms while keeping the same ellipsoid and per-physics-substep semantics.
    """

    __slots__ = (
        "lon0", "lat0", "alt0", "_x0", "_y0", "_z0",
        "_sin_lon0", "_cos_lon0", "_sin_lat0", "_cos_lat0",
    )

    def __init__(self, lon0=120.0, lat0=60.0, alt0=0.0):
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        self.alt0 = float(alt0)
        lon0_rad = np.deg2rad(self.lon0)
        lat0_rad = np.deg2rad(self.lat0)
        self._sin_lon0 = float(np.sin(lon0_rad))
        self._cos_lon0 = float(np.cos(lon0_rad))
        self._sin_lat0 = float(np.sin(lat0_rad))
        self._cos_lat0 = float(np.cos(lat0_rad))
        self._x0, self._y0, self._z0 = self._geodetic_to_ecef(
            self.lon0, self.lat0, self.alt0
        )

    @staticmethod
    def _geodetic_to_ecef(lon, lat, alt):
        lon_rad = np.deg2rad(lon)
        lat_rad = np.deg2rad(lat)
        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        radius = _WGS84_A_M / np.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
        x = (radius + alt) * cos_lat * np.cos(lon_rad)
        y = (radius + alt) * cos_lat * np.sin(lon_rad)
        z = (radius * (1.0 - _WGS84_E2) + alt) * sin_lat
        return float(x), float(y), float(z)

    @staticmethod
    def _ecef_to_geodetic(x, y, z):
        # Bowring's closed-form initial solution followed by two inexpensive
        # Newton-style refinements. The refinement keeps altitude accurate for
        # the WVR operating envelope without calling a general-purpose API.
        p = float(np.hypot(x, y))
        lon = float(np.arctan2(y, x))
        if p < 1e-12:
            lat = np.copysign(np.pi / 2.0, z)
            alt = abs(float(z)) - _WGS84_B_M
            return np.rad2deg(lon), np.rad2deg(lat), alt

        theta = np.arctan2(z * _WGS84_A_M, p * _WGS84_B_M)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        lat = np.arctan2(
            z + _WGS84_EP2 * _WGS84_B_M * sin_theta ** 3,
            p - _WGS84_E2 * _WGS84_A_M * cos_theta ** 3,
        )
        for _ in range(2):
            sin_lat = np.sin(lat)
            radius = _WGS84_A_M / np.sqrt(
                1.0 - _WGS84_E2 * sin_lat * sin_lat
            )
            alt = p / np.cos(lat) - radius
            lat = np.arctan2(
                z,
                p * (1.0 - _WGS84_E2 * radius / (radius + alt)),
            )
        sin_lat = np.sin(lat)
        radius = _WGS84_A_M / np.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
        alt = p / np.cos(lat) - radius
        return float(np.rad2deg(lon)), float(np.rad2deg(lat)), float(alt)

    def lla_to_neu(self, lon, lat, alt, out=None):
        x, y, z = self._geodetic_to_ecef(lon, lat, alt)
        dx, dy, dz = x - self._x0, y - self._y0, z - self._z0
        east = -self._sin_lon0 * dx + self._cos_lon0 * dy
        north = (
            -self._sin_lat0 * self._cos_lon0 * dx
            - self._sin_lat0 * self._sin_lon0 * dy
            + self._cos_lat0 * dz
        )
        up = (
            self._cos_lat0 * self._cos_lon0 * dx
            + self._cos_lat0 * self._sin_lon0 * dy
            + self._sin_lat0 * dz
        )
        if out is None:
            return np.asarray([north, east, up], dtype=np.float64)
        out[:] = north, east, up
        return out

    def neu_to_lla(self, north, east, up, out=None):
        dx = (
            -self._sin_lat0 * self._cos_lon0 * north
            - self._sin_lon0 * east
            + self._cos_lat0 * self._cos_lon0 * up
        )
        dy = (
            -self._sin_lat0 * self._sin_lon0 * north
            + self._cos_lon0 * east
            + self._cos_lat0 * self._sin_lon0 * up
        )
        dz = self._cos_lat0 * north + self._sin_lat0 * up
        lon, lat, alt = self._ecef_to_geodetic(
            self._x0 + dx, self._y0 + dy, self._z0 + dz
        )
        if out is None:
            return np.asarray([lon, lat, alt], dtype=np.float64)
        out[:] = lon, lat, alt
        return out


@lru_cache(maxsize=64)
def get_wgs84_local_frame(lon0=120.0, lat0=60.0, alt0=0.0):
    """Return an immutable-by-convention cached local WGS-84 frame."""
    return WGS84LocalFrame(float(lon0), float(lat0), float(alt0))


def parse_config(filename):
    """Parse JSBSim config file.

    Args:
        config (str): config file name

    Returns:
        (EnvConfig): a custom class which parsing dict into object.
    """
    filepath = os.path.join(get_root_dir(), 'configs', f'{filename}.yaml')
    assert os.path.exists(filepath), \
        f'config path {filepath} does not exist. Please pass in a string that represents the file path to the config yaml.'
    with open(filepath, 'r', encoding='utf-8') as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)

    return type('EnvConfig', (object,), config_data)


def get_root_dir():
    return os.path.join(os.path.split(os.path.realpath(__file__))[0], '..')


def LLA2NEU(lon, lat, alt, lon0=120.0, lat0=60.0, alt0=0):
    """Convert from Geodetic Coordinate System to NEU Coordinate System.

    Args:
        lon, lat, alt (float): target geodetic lontitude(°), latitude(°), altitude(m)
        lon, lat, alt (float): observer geodetic lontitude(°), latitude(°), altitude(m); Default=`(120°E, 60°N, 0m)`

    Returns:
        (np.array): (North, East, Up), unit: m
    """
    return get_wgs84_local_frame(lon0, lat0, alt0).lla_to_neu(lon, lat, alt)


def NEU2LLA(n, e, u, lon0=120.0, lat0=60.0, alt0=0):
    """Convert from NEU Coordinate System to Geodetic Coordinate System.

    Args:
        n, e, u (float): target relative position w.r.t. North, East, Down
        lon, lat, alt (float): observer geodetic lontitude(°), latitude(°), altitude(m); Default=`(120°E, 60°N, 0m)`

    Returns:
        (np.array): (lon, lat, alt), unit: °, °, m
    """
    return get_wgs84_local_frame(lon0, lat0, alt0).neu_to_lla(n, e, u)


def LLA2NEU_reference(lon, lat, alt, lon0=120.0, lat0=60.0, alt0=0):
    """Original pymap3d path retained for differential tests and audits."""
    n, e, d = pymap3d.geodetic2ned(lat, lon, alt, lat0, lon0, alt0)
    return np.asarray([n, e, -d], dtype=np.float64)


def NEU2LLA_reference(n, e, u, lon0=120.0, lat0=60.0, alt0=0):
    """Original pymap3d path retained for differential tests and audits."""
    lat, lon, h = pymap3d.ned2geodetic(n, e, -u, lat0, lon0, alt0)
    return np.asarray([lon, lat, h], dtype=np.float64)


def get_AO_TA_R(ego_feature, enm_feature, return_side=False):
    """Get AO & TA angles and relative distance between two agent.

    Args:
        ego_feature & enemy_feature (tuple): (north, east, down, vn, ve, vd)

    Returns:
        (tuple): ego_AO, ego_TA, R
    """
    ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz = ego_feature
    ego_v = np.linalg.norm([ego_vx, ego_vy, ego_vz])
    enm_x, enm_y, enm_z, enm_vx, enm_vy, enm_vz = enm_feature
    enm_v = np.linalg.norm([enm_vx, enm_vy, enm_vz])
    delta_x, delta_y, delta_z = enm_x - ego_x, enm_y - ego_y, enm_z - ego_z
    R = np.linalg.norm([delta_x, delta_y, delta_z])

    proj_dist = delta_x * ego_vx + delta_y * ego_vy + delta_z * ego_vz
    ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))
    proj_dist = delta_x * enm_vx + delta_y * enm_vy + delta_z * enm_vz
    ego_TA = np.arccos(np.clip(proj_dist / (R * enm_v + 1e-8), -1, 1))

    if not return_side:
        return ego_AO, ego_TA, R
    else:
        # Scalar 2-D determinant. This is mathematically identical to the
        # former ``np.cross([x, y], [x, y])`` expression, whose 2-D-vector
        # form was removed in NumPy 2.x.
        side_flag = np.sign(ego_vx * delta_y - ego_vy * delta_x)
        return ego_AO, ego_TA, R, side_flag


def get2d_AO_TA_R(ego_feature, enm_feature, return_side=False):
    ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz = ego_feature
    ego_v = np.linalg.norm([ego_vx, ego_vy])
    enm_x, enm_y, enm_z, enm_vx, enm_vy, enm_vz = enm_feature
    enm_v = np.linalg.norm([enm_vx, enm_vy])
    delta_x, delta_y, delta_z = enm_x - ego_x, enm_y - ego_y, enm_z - ego_z
    R = np.linalg.norm([delta_x, delta_y])

    proj_dist = delta_x * ego_vx + delta_y * ego_vy
    ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))
    proj_dist = delta_x * enm_vx + delta_y * enm_vy
    ego_TA = np.arccos(np.clip(proj_dist / (R * enm_v + 1e-8), -1, 1))

    if not return_side:
        return ego_AO, ego_TA, R
    else:
        side_flag = np.sign(ego_vx * delta_y - ego_vy * delta_x)
        return ego_AO, ego_TA, R, side_flag


def in_range_deg(angle):
    """ Given an angle in degrees, normalises in (-180, 180] """
    angle = angle % 360
    if angle > 180:
        angle -= 360
    return angle


def in_range_rad(angle):
    """ Given an angle in rads, normalises in (-pi, pi] """
    angle = angle % (2 * np.pi)
    if angle > np.pi:
        angle -= 2 * np.pi
    return angle
