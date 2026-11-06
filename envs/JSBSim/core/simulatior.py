import os
import logging
import numpy as np
from collections import deque
from abc import ABC, abstractmethod
from typing import Literal, Union, List

import jsbsim
from .catalog import Property, Catalog
from ..utils.utils import get_root_dir, get_wgs84_local_frame

TeamColors = Literal["Red", "Blue", "Green", "Violet", "Orange"]


class BaseSimulator(ABC):

    def __init__(self, uid: str, color: TeamColors, dt: float):
        """Constructor. Creates an instance of simulator, initialize all the available properties.

        Args:
            uid (str): 5-digits hexadecimal numbers for unique identification.
            color (TeamColors): use different color strings to represent diferent teams
            dt (float): simulation timestep. Default = `1 / 60`.
        """
        self.__uid = uid
        self.__color = color
        self.__dt = dt
        self.model = ""
        self._geodetic = np.zeros(3)
        self._position = np.zeros(3)
        self._posture = np.zeros(3)
        self._velocity = np.zeros(3)
        logging.debug(f"{self.__class__.__name__}:{self.__uid} is created!")

    @property
    def uid(self) -> str:
        return self.__uid

    @property
    def color(self) -> str:
        return self.__color

    @property
    def dt(self) -> float:
        return self.__dt

    def get_geodetic(self):
        """(lontitude, latitude, altitude), unit: °, m"""
        return self._geodetic

    def get_position(self):
        """(north, east, up), unit: m"""
        return self._position

    def get_rpy(self):
        """(roll, pitch, yaw), unit: rad"""
        return self._posture

    def get_velocity(self):
        """(v_north, v_east, v_up), unit: m/s"""
        return self._velocity

    def reload(self):
        self._geodetic = np.zeros(3)
        self._position = np.zeros(3)
        self._posture = np.zeros(3)
        self._velocity = np.zeros(3)

    @abstractmethod
    def run(self, **kwargs):
        pass

    def log(self):
        lon, lat, alt = self.get_geodetic()
        roll, pitch, yaw = self.get_rpy() * 180 / np.pi
        log_msg = f"{self.uid},T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},"
        log_msg += f"Name={self.model.upper()},"
        log_msg += f"Color={self.color}"
        return log_msg

    @abstractmethod
    def close(self):
        pass

    def __del__(self):
        logging.debug(f"{self.__class__.__name__}:{self.uid} is deleted!")


class AircraftSimulator(BaseSimulator):
    """A class which wraps an instance of JSBSim and manages communication with it.
    """

    ALIVE = 0
    CRASH = 1       # low altitude / extreme state / overload
    SHOTDOWN = 2    # missile attack

    def __init__(self,
                 uid: str = "A0100",
                 color: TeamColors = "Red",
                 model: str = 'f16',
                 init_state: dict = {},
                 origin: tuple = (120.0, 60.0, 0.0),
                 sim_freq: int = 60,
                 reuse_fdm: bool = False, **kwargs):
        """Constructor. Creates an instance of JSBSim, loads an aircraft and sets initial conditions.

        Args:
            uid (str): 5-digits hexadecimal numbers for unique identification. Default = `"A0100"`.
            color (TeamColors): use different color strings to represent diferent teams
            model (str): name of aircraft to be loaded. Default = `"f16"`.
                model path: './data/aircraft_name/aircraft_name.xml'
            init_state (dict): dict mapping properties to their initial values. Input empty dict to use a default set of initial props.
            origin (tuple): origin point (longitude, latitude, altitude) of the Global Combat Field. Default = `(120.0, 60.0, 0.0)`
            sim_freq (int): JSBSim integration frequency. Default = `60`.
        """
        super().__init__(uid, color, 1 / sim_freq)
        self.model = model
        self.init_state = init_state
        self.reuse_fdm = bool(reuse_fdm)
        self.lon0, self.lat0, self.alt0 = origin
        self._local_frame = get_wgs84_local_frame(*origin)
        self.bloods = 100
        self.__status = AircraftSimulator.ALIVE
        for key, value in kwargs.items():
            if key == 'num_missiles':
                self.num_missiles = value  # type: int
                self.num_left_missiles = self.num_missiles  # type: int
        # fixed simulator links
        self.partners = []  # type: List[AircraftSimulator]
        self.enemies = []   # type: List[AircraftSimulator]
        # temp simulator links
        self.launch_missiles = []   # type: List[MissileSimulator]
        self.under_missiles = []    # type: List[MissileSimulator]
        # initialize simulator
        self.reload()

    @property
    def is_alive(self):
        return self.__status == AircraftSimulator.ALIVE

    @property
    def is_crash(self):
        return self.__status == AircraftSimulator.CRASH

    @property
    def is_shotdown(self):
        return self.__status == AircraftSimulator.SHOTDOWN

    def crash(self):
        self.__status = AircraftSimulator.CRASH

    def shotdown(self):
        self.__status = AircraftSimulator.SHOTDOWN

    def reload(self, new_state: Union[dict, None] = None, new_origin: Union[tuple, None] = None):
        """Reload aircraft simulator
        """
        super().reload()

        # reset temp simulator links
        self.bloods = 100
        self.__status = AircraftSimulator.ALIVE
        self.launch_missiles.clear()
        self.under_missiles.clear()
        self.num_left_missiles = self.num_missiles

        # Load once or reset the existing FDM while preserving the model.
        can_reuse = self.reuse_fdm and getattr(
            self, "jsbsim_exec", None
        ) is not None
        if can_reuse:
            # Skip the implicit RunIC so the new initial conditions can be
            # installed before the single explicit run_ic() below.
            self.jsbsim_exec.reset_to_initial_conditions(2)
        else:
            self.jsbsim_exec = jsbsim.FGFDMExec(
                os.path.join(get_root_dir(), 'data')
            )
            self.jsbsim_exec.set_debug_level(0)
            self.jsbsim_exec.load_model(self.model)
            Catalog.add_jsbsim_props(
                self.jsbsim_exec.query_property_catalog("")
            )
            self.jsbsim_exec.set_dt(self.dt)
        self.clear_defalut_condition()

        # assign new properties
        if new_state is not None:
            self.init_state = new_state
        if new_origin is not None:
            self.lon0, self.lat0, self.alt0 = new_origin
            self._local_frame = get_wgs84_local_frame(*new_origin)
        for key, value in self.init_state.items():
            self.set_property_value(Catalog[key], value)
        success = self.jsbsim_exec.run_ic()
        if not success:
            raise RuntimeError("JSBSim failed to init simulation conditions.")

        # propulsion init running
        propulsion = self.jsbsim_exec.get_propulsion()
        n = propulsion.get_num_engines()
        for j in range(n):
            propulsion.get_engine(j).init_running()
        propulsion.get_steady_state()
        # update inner property
        self._update_properties()

    def clear_defalut_condition(self):
        default_condition = {
            Catalog.ic_long_gc_deg: 120.0,  # geodesic longitude [deg]
            Catalog.ic_lat_geod_deg: 60.0,  # geodesic latitude  [deg]
            Catalog.ic_h_sl_ft: 20000,      # altitude above mean sea level [ft]
            Catalog.ic_psi_true_deg: 0.0,   # initial (true) heading [deg] (0, 360)
            Catalog.ic_u_fps: 800.0,        # body frame x-axis velocity [ft/s]  (-2200, 2200)
            Catalog.ic_v_fps: 0.0,          # body frame y-axis velocity [ft/s]  (-2200, 2200)
            Catalog.ic_w_fps: 0.0,          # body frame z-axis velocity [ft/s]  (-2200, 2200)
            Catalog.ic_p_rad_sec: 0.0,      # roll rate  [rad/s]  (-2 * pi, 2 * pi)
            Catalog.ic_q_rad_sec: 0.0,      # pitch rate [rad/s]  (-2 * pi, 2 * pi)
            Catalog.ic_r_rad_sec: 0.0,      # yaw rate   [rad/s]  (-2 * pi, 2 * pi)
            Catalog.ic_roc_fpm: 0.0,        # initial rate of climb [ft/min]
            Catalog.ic_terrain_elevation_ft: 0,
        }
        for prop, value in default_condition.items():
            self.set_property_value(prop, value)

    def run(self):
        """Runs JSBSim simulation until the agent interacts and update custom properties.

        JSBSim monitors the simulation and detects whether it thinks it should
        end, e.g. because a simulation time was specified. False is returned
        if JSBSim termination criteria are met.

        Returns:
            (bool): False if sim has met JSBSim termination criteria else True.
        """
        if self.is_alive:
            if self.bloods <= 0:
                self.shotdown()
            result = self.jsbsim_exec.run()
            if not result:
                raise RuntimeError("JSBSim failed.")
            self._update_properties()
            return result
        else:
            return True

    def close(self):
        """ Closes the simulation and any plots. """
        if self.jsbsim_exec:
            self.jsbsim_exec = None
        self.partners = []
        self.enemies = []

    def _update_properties(self):
        # Read native JSBSim fields directly in the 60 Hz hot path. The public
        # metre-based Catalog fields are Python callbacks over native feet
        # values, so perform the same exact 0.3048 conversion here.
        read = self.jsbsim_exec.get_property_value
        self._geodetic[:] = (
            read(Catalog.position_long_gc_deg.name_jsbsim),
            read(Catalog.position_lat_geod_deg.name_jsbsim),
            read(Catalog.position_h_sl_ft.name_jsbsim) * 0.3048,
        )
        self._local_frame.lla_to_neu(*self._geodetic, out=self._position)
        self._posture[:] = (
            read(Catalog.attitude_roll_rad.name_jsbsim),
            read(Catalog.attitude_pitch_rad.name_jsbsim),
            read(Catalog.attitude_heading_true_rad.name_jsbsim),
        )
        self._velocity[:] = (
            read(Catalog.velocities_v_north_fps.name_jsbsim) * 0.3048,
            read(Catalog.velocities_v_east_fps.name_jsbsim) * 0.3048,
            read(Catalog.velocities_v_down_fps.name_jsbsim) * 0.3048,
        )
        # v_down -> v_up
        self._velocity[2] = -self._velocity[2]

    def get_sim_time(self):
        """ Gets the simulation time from JSBSim, a float. """
        return self.jsbsim_exec.get_sim_time()

    def get_property_values(self, props):
        """Get the values of the specified properties

        :param props: list of Properties

        : return: NamedTupl e with properties name and their values
        """
        return [self.get_property_value(prop) for prop in props]

    def set_property_values(self, props, values):
        """Set the values of the specified properties

        :param props: list of Properties

        :param values: list of float
        """
        if not len(props) == len(values):
            raise ValueError("mismatch between properties and values size")
        for prop, value in zip(props, values):
            self.set_property_value(prop, value)

    def get_property_value(self, prop):
        """Get the value of the specified property from the JSBSim simulation

        :param prop: Property

        :return : float
        """
        if isinstance(prop, Property):
            if prop.access == "R":
                if prop.update:
                    prop.update(self)
            return self.jsbsim_exec.get_property_value(prop.name_jsbsim)
        else:
            raise ValueError(f"prop type unhandled: {type(prop)} ({prop})")

    def set_property_value(self, prop, value):
        """Set the values of the specified property

        :param prop: Property

        :param value: float
        """
        # set value in property bounds
        if isinstance(prop, Property):
            if value < prop.min:
                value = prop.min
            elif value > prop.max:
                value = prop.max

            self.jsbsim_exec.set_property_value(prop.name_jsbsim, value)

            if "W" in prop.access:
                if prop.update:
                    prop.update(self)
        else:
            raise ValueError(f"prop type unhandled: {type(prop)} ({prop})")

    def check_missile_warning(self):
        for missile in self.under_missiles:
            if missile.is_alive:
                return missile
        return None


class MissileSimulator(BaseSimulator):

    INACTIVE = -1
    LAUNCHED = 0
    HIT = 1
    MISS = 2

    @classmethod
    def create(cls, parent: AircraftSimulator, target: AircraftSimulator, uid: str, missile_model: str = "AIM-9L"):
        assert parent.dt == target.dt, "integration timestep must be same!"
        missile = MissileSimulator(uid, parent.color, missile_model, parent.dt)
        missile.launch(parent)
        missile.target(target)
        return missile

    def __init__(self,
                 uid="A0101",
                 color="Red",
                 model="AIM-9L",
                 dt=1 / 12):
        super().__init__(uid, color, dt)
        self.__status = MissileSimulator.INACTIVE
        self.model = model
        self.parent_aircraft = None  # type: AircraftSimulator
        self.target_aircraft = None  # type: AircraftSimulator
        self.render_explosion = False

        # missile parameters (for AIM-9L)
        self._g = 9.81      # gravitational acceleration
        self._t_max = 60    # time limitation of missile life
        self._t_thrust = 3  # time limitation of engine
        self._Isp = 120     # average specific impulse
        self._Length = 2.87
        self._Diameter = 0.127
        self._cD = 0.4      # aerodynamic drag factor
        self._m0 = 84       # mass, unit: kg
        self._dm = 6        # mass loss rate, unit: kg/s
        self._K = 3         # proportionality constant of proportional navigation
        self._nyz_max = 30  # max overload
        self._Rc = 300      # radius of explosion, unit: m
        self._v_min = 150   # minimun velocity, unit: m/s

    @property
    def is_alive(self):
        """Missile is still flying"""
        return self.__status == MissileSimulator.LAUNCHED

    @property
    def is_success(self):
        """Missile has hit the target"""
        return self.__status == MissileSimulator.HIT

    @property
    def is_done(self):
        """Missile is already exploded"""
        return self.__status == MissileSimulator.HIT \
            or self.__status == MissileSimulator.MISS

    @property
    def Isp(self):
        return self._Isp if self._t < self._t_thrust else 0

    @property
    def K(self):
        """Proportional Guidance Coefficient"""
        # return self._K
        return max(self._K * (self._t_max - self._t) / self._t_max, 0)

    @property
    def S(self):
        """Cross-Sectional area, unit m^2"""
        S0 = np.pi * (self._Diameter / 2)**2
        S0 += np.linalg.norm([np.sin(self._dtheta), np.sin(self._dphi)]) * self._Diameter * self._Length
        return S0

    @property
    def rho(self):
        """Air Density, unit: kg/m^3"""
        # approximate expression
        return 1.225 * np.exp(-self._geodetic[-1] / 9300)
        # exact expression (Reference: https://www.cnblogs.com/pathjh/p/9127352.html)
        rho0, T0, h = 1.225, 288.15, self._geodetic[-1]
        if h <= 11000:  # Troposphere
            T = T0 - 0.0065 * h
            return rho0 * (T / T0)**4.25588
        elif h <= 20000:  # Lower Stratosphere
            T = 216.65
            return 0.36392 * np.exp((11000 - h) / 6341.62)
        else:  # Upper Stratosphere
            T = 216.65 + 0.001 * (h - 20000)
            return 0.088035 * (T / 216.65)**(-35.1632)

    @property
    def target_distance(self) -> float:
        return np.linalg.norm(self.target_aircraft.get_position() - self.get_position())

    def launch(self, parent: AircraftSimulator):
        # inherit kinetic parameters from parent aricraft
        self.parent_aircraft = parent
        self.parent_aircraft.launch_missiles.append(self)
        self._geodetic[:] = parent.get_geodetic()
        self._position[:] = parent.get_position()
        self._velocity[:] = parent.get_velocity()
        self._posture[:] = parent.get_rpy()
        self._posture[0] = 0  # missile's roll remains zero
        self.lon0, self.lat0, self.alt0 = parent.lon0, parent.lat0, parent.alt0
        self._local_frame = getattr(
            parent,
            "_local_frame",
            get_wgs84_local_frame(self.lon0, self.lat0, self.alt0),
        )
        # init status
        self._t = 0
        self._m = self._m0
        self._dtheta, self._dphi = 0, 0
        self.__status = MissileSimulator.LAUNCHED
        self.resolution_reason = "pending"
        self.resolution_role = "pending"
        self.capture_fraction = None
        self.closest_approach_m = None
        self._deferred_target_effect = False
        self._distance_pre = np.inf
        self._distance_increment = deque(maxlen=int(5 / self.dt))  # 5s of distance increment -- can't hit
        self._left_t = int(1 / self.dt)  # remove missile 1s after its destroying

    def target(self, target: AircraftSimulator):
        self.target_aircraft = target  # TODO: change target?
        self.target_aircraft.under_missiles.append(self)

    def _set_status(self, status):
        self.__status = status

    def _finish(self, status, reason):
        self._set_status(status)
        self.resolution_reason = str(reason)

    @staticmethod
    def relative_segment_metrics(relative_start, relative_end):
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
        fraction = float(np.clip(
            -np.dot(start, delta) / denominator, 0.0, 1.0))
        return float(np.linalg.norm(start + fraction * delta)), fraction

    def _capture(self, target, reason, fraction, closest,
                 defer_target_effect):
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
        """Commit synchronized target effects independently of iteration order."""

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
        # HIT and MISS are absorbing states. This is essential when one agent
        # decision contains multiple physics substeps.
        if self.is_done:
            return
        target = self.target_aircraft
        if target is None:
            self._finish(MissileSimulator.MISS, "missing_target")
            return
        if not target.is_alive:
            self._finish(MissileSimulator.MISS, "target_already_resolved")
            return

        missile_start = np.asarray(
            self.get_position(), dtype=np.float64).copy()
        target_current = np.asarray(
            target.get_position(), dtype=np.float64).copy()
        target_velocity = np.asarray(
            target.get_velocity(), dtype=np.float64).copy()
        if target_segment_start is None:
            target_start = target_current - target_velocity * float(self.dt)
        else:
            target_start = np.asarray(
                target_segment_start, dtype=np.float64).copy()
            if target_start.shape != (3,) or not np.all(np.isfinite(target_start)):
                self._finish(MissileSimulator.MISS, "invalid_target_segment")
                return
        self._t += self.dt
        relative_start = target_start - missile_start
        start_distance = float(np.linalg.norm(relative_start))
        if start_distance <= self._Rc:
            self._capture(
                target, "capture", 0.0, start_distance,
                bool(defer_target_effect))
            return

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

    def log(self):
        if self.is_alive:
            log_msg = super().log()
        elif self.is_done and (not self.render_explosion):
            self.render_explosion = True
            # remove missile model
            log_msg = f"-{self.uid}\n"
            # add explosion
            lon, lat, alt = self.get_geodetic()
            roll, pitch, yaw = self.get_rpy() * 180 / np.pi
            log_msg += f"{self.uid}F,T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},"
            log_msg += f"Type=Misc+Explosion,Color={self.color},Radius={self._Rc}"
        else:
            log_msg = None
        return log_msg

    def close(self):
        self.target_aircraft = None

    def _guidance(self):
        """
        Guidance law, proportional navigation
        """
        x_m, y_m, z_m = self.get_position()
        dx_m, dy_m, dz_m = self.get_velocity()
        v_m = np.linalg.norm([dx_m, dy_m, dz_m])
        theta_m = np.arcsin(dz_m / v_m)
        x_t, y_t, z_t = self.target_aircraft.get_position()
        dx_t, dy_t, dz_t = self.target_aircraft.get_velocity()
        Rxy = np.linalg.norm([x_m - x_t, y_m - y_t])  # distance from missile to target project to X-Y plane
        Rxyz = np.linalg.norm([x_m - x_t, y_m - y_t, z_t - z_m])  # distance from missile to target
        # calculate beta & eps, but no need actually...
        # beta = np.arctan2(y_m - y_t, x_m - x_t)  # relative yaw
        # eps = np.arctan2(z_m - z_t, np.linalg.norm([x_m - x_t, y_m - y_t]))  # relative pitch
        dbeta = ((dy_t - dy_m) * (x_t - x_m) - (dx_t - dx_m) * (y_t - y_m)) / Rxy**2
        deps = ((dz_t - dz_m) * Rxy**2 - (z_t - z_m) * (
            (x_t - x_m) * (dx_t - dx_m) + (y_t - y_m) * (dy_t - dy_m))) / (Rxyz**2 * Rxy)
        ny = self.K * v_m / self._g * np.cos(theta_m) * dbeta
        nz = self.K * v_m / self._g * deps + np.cos(theta_m)
        return np.clip([ny, nz], -self._nyz_max, self._nyz_max), Rxyz

    def _state_trans(self, action):
        """
        State transition function
        """
        # update position & geodetic
        self._position[:] += self.dt * self.get_velocity()
        self._local_frame.neu_to_lla(*self.get_position(), out=self._geodetic)
        # update velocity & posture
        v = np.linalg.norm(self.get_velocity())
        theta, phi = self.get_rpy()[1:]
        T = self._g * self.Isp * self._dm
        D = 0.5 * self._cD * self.S * self.rho * v**2
        nx = (T - D) / (self._m * self._g)
        ny, nz = action

        dv = self._g * (nx - np.sin(theta))
        self._dphi = self._g / v * (ny / np.cos(theta))
        self._dtheta = self._g / v * (nz - np.cos(theta))

        v += self.dt * dv
        phi += self.dt * self._dphi
        theta += self.dt * self._dtheta
        self._velocity[:] = np.array([
            v * np.cos(theta) * np.cos(phi),
            v * np.cos(theta) * np.sin(phi),
            v * np.sin(theta)
        ])
        self._posture[:] = np.array([0, theta, phi])
        # update mass
        if self._t < self._t_thrust:
            self._m = self._m - self.dt * self._dm
