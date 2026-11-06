"""Vectorized differentiable six-degree-of-freedom F-16 approximation.

The state uses SI units and a local NED navigation frame:

``[north, east, down, u, v, w, qw, qx, qy, qz, p, q, r, da, de, dr,
throttle, roll_i, pitch_i, yaw_i]``

The quaternion rotates body-frame vectors into NED.  Controls are normalized
aileron, elevator, rudder and throttle commands.  This implementation is an
experimental airborne-only model: it contains rigid-body equations, an ISA
atmosphere, the principal JSBSim XML aerodynamic/F100 tables and airborne
flight-control channels, but not fuel, ground-reaction or auxiliary systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .tables import (
    TableData, load_jsbsim_function_tables, lookup_1d, lookup_2d,
)


@dataclass(frozen=True)
class TorchF16Parameters:
    mass_kg: float = 9295.0
    wing_area_m2: float = 27.8709
    wing_span_m: float = 9.144
    mean_chord_m: float = 3.450336
    ixx_kg_m2: float = 12874.84
    iyy_kg_m2: float = 75676.79
    izz_kg_m2: float = 85552.11
    ixz_kg_m2: float = -1331.38
    max_thrust_n: float = 129000.0
    surface_time_constant_s: float = 0.08
    throttle_time_constant_s: float = 0.50
    max_aileron_rad: float = 0.375
    max_elevator_rad: float = 0.436
    max_rudder_rad: float = 0.524
    gravity_m_s2: float = 9.80665


def _cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.cross(a, b, dim=-1)


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Return body-to-NED direction cosine matrices."""

    q = quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True).clamp_min(1.0e-9)
    w, x, y, z = q.unbind(-1)
    two = 2.0
    return torch.stack((
        1 - two * (y * y + z * z),
        two * (x * y - z * w),
        two * (x * z + y * w),
        two * (x * y + z * w),
        1 - two * (x * x + z * z),
        two * (y * z - x * w),
        two * (x * z - y * w),
        two * (y * z + x * w),
        1 - two * (x * x + y * y),
    ), dim=-1).reshape(*q.shape[:-1], 3, 3)


def quaternion_derivative(
        quaternion: torch.Tensor, rates: torch.Tensor) -> torch.Tensor:
    """Quaternion derivative for body rates p, q, r."""

    qw = quaternion[..., :1]
    qv = quaternion[..., 1:]
    scalar = -0.5 * (qv * rates).sum(dim=-1, keepdim=True)
    vector = 0.5 * (qw * rates + _cross(qv, rates))
    return torch.cat((scalar, vector), dim=-1)


class TorchF16BatchDynamics(torch.nn.Module):
    """Batched airborne 6-DoF dynamics with a fixed 60 Hz integration step."""

    state_size = 20
    control_size = 4

    def __init__(self, parameters: TorchF16Parameters | None = None,
                 dt: float = 1.0 / 60.0):
        super().__init__()
        self.parameters = parameters or TorchF16Parameters()
        self.dt = float(dt)
        if self.dt <= 0.0:
            raise ValueError("integration time step must be positive")
        p = self.parameters
        inertia = torch.tensor([
            [p.ixx_kg_m2, 0.0, p.ixz_kg_m2],
            [0.0, p.iyy_kg_m2, 0.0],
            [p.ixz_kg_m2, 0.0, p.izz_kg_m2],
        ], dtype=torch.float32)
        self.register_buffer("inertia", inertia)
        self.register_buffer("inverse_inertia", torch.linalg.inv(inertia))
        data_root = Path(__file__).resolve().parents[1] / "JSBSim" / "data"
        aero_tables = load_jsbsim_function_tables(
            data_root / "aircraft" / "f16" / "f16.xml")
        engine_tables = load_jsbsim_function_tables(
            data_root / "engine" / "F100-PW-229.xml")
        names = (
            "CDDh", "CDmach", "CDq", "CYb_M", "CYp", "CYr", "CLDh", "CLq",
            "Clb", "Clb_M", "Clp", "Clr", "Clda", "Clda_M", "Cldr_M",
            "Cldr", "CmDh", "Cma_M", "Cmq", "Cnb", "Cnb_M", "Cnp",
            "Cnr", "Cnda_M", "Cnda", "Cndr", "Cndr_M",
        )
        for name in names:
            self._register_table(name, aero_tables[f"aero/coefficient/{name}"])
        for name in ("IdleThrust", "MilThrust", "AugThrust"):
            self._register_table(name, engine_tables[name])

    def _register_table(self, name: str, table: TableData):
        self.register_buffer(f"{name}_row", table.row)
        self.register_buffer(f"{name}_values", table.values)
        if table.column is not None:
            self.register_buffer(f"{name}_column", table.column)

    def _one(self, name: str, value: torch.Tensor) -> torch.Tensor:
        return lookup_1d(
            value, getattr(self, f"{name}_row"),
            getattr(self, f"{name}_values"))

    def _two(self, name: str, row: torch.Tensor,
             column: torch.Tensor) -> torch.Tensor:
        return lookup_2d(
            row, column, getattr(self, f"{name}_row"),
            getattr(self, f"{name}_column"),
            getattr(self, f"{name}_values"))

    def _f100_thrust(self, throttle: torch.Tensor, mach: torch.Tensor,
                     altitude_m: torch.Tensor) -> torch.Tensor:
        """Tensorized F100-PW-229 XML deck, including augmentation."""

        density_altitude_ft = altitude_m * 3.280839895
        idle = 17800.0 * 4.4482216153 * self._two(
            "IdleThrust", mach, density_altitude_ft)
        military = 17800.0 * 4.4482216153 * self._two(
            "MilThrust", mach, density_altitude_ft)
        augmented = 29000.0 * 4.4482216153 * self._two(
            "AugThrust", mach, density_altitude_ft)
        dry_fraction = (2.0 * throttle).clamp(0.0, 1.0)
        augmented_fraction = (2.0 * throttle - 1.0).clamp(0.0, 1.0)
        dry = torch.lerp(idle, military, dry_fraction)
        return torch.where(throttle <= 0.5, dry,
                           torch.lerp(military, augmented,
                                      augmented_fraction))

    def _flight_control_commands(
            self, control: torch.Tensor, rates: torch.Tensor,
            alpha: torch.Tensor, mach: torch.Tensor,
            integrals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """GPU form of the airborne F-16 roll/pitch/yaw XML channels.

        Controller integrals are explicit state components, so environments
        remain independent and resets remain deterministic.
        """

        roll_command = control[..., 0].clamp(-1.0, 1.0)
        roll_error = roll_command - 0.31821 * rates[..., 0]
        aileron = (roll_command + 3.0 * roll_error
                   + 0.0005 * integrals[..., 0]).clamp(-1.0, 1.0)

        elevator_limited = control[..., 1].clamp(-1.0, 0.44)
        alpha_grid = torch.as_tensor(
            [-0.5236, -0.5, 0.0, 0.5, 0.5236],
            dtype=alpha.dtype, device=alpha.device)
        alpha_gain = torch.as_tensor(
            [0.0, 0.11, 1.0, 0.11, 0.0],
            dtype=alpha.dtype, device=alpha.device)
        scheduled = elevator_limited * lookup_1d(
            alpha, alpha_grid, alpha_gain)
        pitch_error = scheduled + 6.2 * rates[..., 1]
        elevator = (scheduled + 1.0472 * alpha + 0.3 * pitch_error
                    + 0.025 * integrals[..., 1]).clamp(-1.0, 1.0)

        speed_fps = mach * 1116.0
        yaw_gain = torch.where(
            speed_fps <= 80.0, torch.zeros_like(speed_fps),
            torch.where(
                speed_fps < 100.0, 0.75 * (speed_fps - 80.0),
                torch.where(speed_fps < 150.0,
                            15.0 + 1.7 * (speed_fps - 100.0),
                            torch.full_like(speed_fps, 100.0))))
        rudder_command = control[..., 2].clamp(-1.0, 1.0)
        yaw_error = rudder_command + yaw_gain * rates[..., 2]
        rudder = (rudder_command + 0.1055 * yaw_error
                  + 0.00001 * integrals[..., 2]).clamp(-1.0, 1.0)
        commands = torch.stack((
            aileron, elevator, rudder,
            control[..., 3].clamp(0.0, 1.0)), dim=-1)
        errors = torch.stack((roll_error, pitch_error, yaw_error), dim=-1)
        return commands, errors

    @staticmethod
    def level_state(position_ned: torch.Tensor, heading_rad: torch.Tensor,
                    speed_m_s: torch.Tensor) -> torch.Tensor:
        """Construct level-flight states from broadcast-compatible tensors."""

        position_ned = torch.as_tensor(position_ned)
        heading_rad = torch.as_tensor(
            heading_rad, device=position_ned.device,
            dtype=position_ned.dtype)
        speed_m_s = torch.as_tensor(
            speed_m_s, device=position_ned.device,
            dtype=position_ned.dtype)
        half = 0.5 * heading_rad
        quaternion = torch.stack((
            torch.cos(half), torch.zeros_like(half),
            torch.zeros_like(half), torch.sin(half)), dim=-1)
        velocity = torch.stack((
            speed_m_s, torch.zeros_like(speed_m_s),
            torch.zeros_like(speed_m_s)), dim=-1)
        rates = torch.zeros_like(velocity)
        actuators = torch.zeros(
            *velocity.shape[:-1], 7, device=velocity.device,
            dtype=velocity.dtype)
        actuators[..., 3] = 0.7
        return torch.cat(
            (position_ned, velocity, quaternion, rates, actuators), dim=-1)

    def _atmosphere(self, altitude_m: torch.Tensor):
        altitude = altitude_m.clamp(0.0, 20000.0)
        temperature = (288.15 - 0.0065 * altitude).clamp_min(180.0)
        pressure = 101325.0 * torch.pow(temperature / 288.15, 5.25588)
        density = pressure / (287.05287 * temperature)
        speed_of_sound = torch.sqrt(1.4 * 287.05287 * temperature)
        return density, speed_of_sound

    def derivatives(self, state: torch.Tensor, control: torch.Tensor):
        p = self.parameters
        position = state[..., 0:3]
        velocity = state[..., 3:6]
        quaternion = state[..., 6:10]
        rates = state[..., 10:13]
        actuators = state[..., 13:17]
        integrals = state[..., 17:20]
        u, v, w = velocity.unbind(-1)
        airspeed = torch.linalg.vector_norm(
            velocity, dim=-1).clamp_min(30.0)
        alpha = torch.atan2(w, u.clamp_min(1.0))
        beta = torch.asin((v / airspeed).clamp(-0.95, 0.95))
        altitude = -position[..., 2]
        density, speed_of_sound = self._atmosphere(altitude)
        mach = airspeed / speed_of_sound
        dynamic_pressure = 0.5 * density * airspeed.square()
        commands, integral_dot = self._flight_control_commands(
            control, rates, alpha, mach, integrals)
        surface_rates = torch.as_tensor(
            [2.0 / 0.3, 2.0 / 0.3, 2.0 / 0.4],
            dtype=state.dtype, device=state.device)
        surface_dot = ((commands[..., :3] - actuators[..., :3]) / self.dt
                       ).clamp(-surface_rates, surface_rates)
        throttle_dot = ((commands[..., 3:4] - actuators[..., 3:4])
                        / p.throttle_time_constant_s)
        actuator_dot = torch.cat((surface_dot, throttle_dot), dim=-1)

        aileron = actuators[..., 0] * p.max_aileron_rad
        elevator = actuators[..., 1] * p.max_elevator_rad
        rudder = actuators[..., 2] * p.max_rudder_rad
        throttle = actuators[..., 3].clamp(0.0, 1.0)
        phat = rates[..., 0] * p.wing_span_m / (2.0 * airspeed)
        qhat = rates[..., 1] * p.mean_chord_m / (2.0 * airspeed)
        rhat = rates[..., 2] * p.wing_span_m / (2.0 * airspeed)

        # Six aerodynamic axes reconstructed directly from the public JSBSim
        # F-16 XML coefficient tables.  Ground gear and speedbrake terms are
        # omitted because the reinforcement-learning envelope is airborne.
        lift_coefficient = (
            self._two("CLDh", alpha, elevator)
            + rates[..., 1] * p.mean_chord_m / (2.0 * airspeed)
            * self._one("CLq", alpha)).clamp(-1.5, 2.2)
        drag_coefficient = (
            self._two("CDDh", alpha, elevator)
            + self._one("CDmach", mach)
            + rates[..., 1] * p.mean_chord_m / (2.0 * airspeed)
            * self._one("CDq", alpha)).clamp(0.01, 1.8)
        side_coefficient = (
            -1.1460 * beta + beta * self._one("CYb_M", mach)
            - 0.0226 * aileron + 0.0860 * rudder
            + phat * self._one("CYp", alpha)
            + rhat * self._one("CYr", alpha)).clamp(-1.2, 1.2)
        roll_coefficient = (
            self._two("Clb", alpha, beta)
            + beta * self._one("Clb_M", mach)
            + phat * self._one("Clp", alpha)
            + rhat * self._one("Clr", alpha)
            + aileron * (self._two("Clda", alpha, beta)
                         + alpha * self._one("Clda_M", mach))
            + rudder * (self._two("Cldr", alpha, beta)
                        + alpha * self._one("Cldr_M", mach))).clamp(-0.5, 0.5)
        pitch_coefficient = (
            self._two("CmDh", alpha, elevator)
            + alpha * self._one("Cma_M", mach)
            + qhat * self._one("Cmq", alpha)).clamp(-1.2, 1.2)
        yaw_coefficient = (
            self._two("Cnb", alpha, beta)
            + beta * self._one("Cnb_M", mach)
            + phat * self._one("Cnp", alpha)
            + rhat * self._one("Cnr", alpha)
            + aileron * (self._two("Cnda", alpha, beta)
                         + self._one("Cnda_M", mach))
            + rudder * (self._two("Cndr", alpha, beta)
                        + alpha * self._one("Cndr_M", mach))).clamp(-0.5, 0.5)

        lift = dynamic_pressure * p.wing_area_m2 * lift_coefficient
        drag = dynamic_pressure * p.wing_area_m2 * drag_coefficient
        side = dynamic_pressure * p.wing_area_m2 * side_coefficient
        thrust = self._f100_thrust(throttle, mach, altitude)
        sin_alpha, cos_alpha = torch.sin(alpha), torch.cos(alpha)
        aerodynamic_force = torch.stack((
            -drag * cos_alpha + lift * sin_alpha + thrust,
            side,
            -drag * sin_alpha - lift * cos_alpha,
        ), dim=-1)

        rotation_body_to_ned = quaternion_to_matrix(quaternion)
        gravity_ned = torch.zeros_like(velocity)
        gravity_ned[..., 2] = p.gravity_m_s2
        gravity_body = torch.matmul(
            rotation_body_to_ned.transpose(-1, -2),
            gravity_ned.unsqueeze(-1)).squeeze(-1)
        velocity_dot = (
            aerodynamic_force / p.mass_kg + gravity_body
            - _cross(rates, velocity)
        )
        position_dot = torch.matmul(
            rotation_body_to_ned, velocity.unsqueeze(-1)).squeeze(-1)

        moments = torch.stack((
            dynamic_pressure * p.wing_area_m2 * p.wing_span_m
            * roll_coefficient,
            dynamic_pressure * p.wing_area_m2 * p.mean_chord_m
            * pitch_coefficient,
            dynamic_pressure * p.wing_area_m2 * p.wing_span_m
            * yaw_coefficient,
        ), dim=-1)
        angular_momentum = torch.matmul(
            self.inertia.to(dtype=state.dtype),
            rates.unsqueeze(-1)).squeeze(-1)
        rates_dot = torch.matmul(
            self.inverse_inertia.to(dtype=state.dtype),
            (moments - _cross(rates, angular_momentum)).unsqueeze(-1)
        ).squeeze(-1)
        quaternion_dot = quaternion_derivative(quaternion, rates)
        return torch.cat((
            position_dot, velocity_dot, quaternion_dot, rates_dot,
            actuator_dot, integral_dot), dim=-1)

    def _integrate_once(self, state: torch.Tensor,
                        control: torch.Tensor) -> torch.Tensor:
        # Heun integration is substantially more stable than Euler at the
        # fixed 60 Hz project rate while remaining inexpensive and batched.
        first = self.derivatives(state, control)
        predictor = state + self.dt * first
        predictor_quaternion = predictor[..., 6:10]
        predictor = torch.cat((
            predictor[..., :6],
            predictor_quaternion / torch.linalg.vector_norm(
                predictor_quaternion, dim=-1,
                keepdim=True).clamp_min(1.0e-9),
            predictor[..., 10:],
        ), dim=-1)
        second = self.derivatives(predictor, control)
        result = state + 0.5 * self.dt * (first + second)
        quaternion = result[..., 6:10]
        quaternion = quaternion / torch.linalg.vector_norm(
            quaternion, dim=-1, keepdim=True).clamp_min(1.0e-9)
        rates = result[..., 10:13].clamp(-4.0, 4.0)
        actuators = torch.cat((
            result[..., 13:16].clamp(-1.0, 1.0),
            result[..., 16:17].clamp(0.0, 1.0)), dim=-1)
        integrals = result[..., 17:20].clamp(-20.0, 20.0)
        return torch.cat((
            result[..., :6], quaternion, rates, actuators, integrals), dim=-1)

    def forward(self, state: torch.Tensor, control: torch.Tensor,
                substeps: int = 1) -> torch.Tensor:
        if state.shape[-1] != self.state_size:
            raise ValueError("Torch F-16 state must have 20 components")
        if control.shape[-1] != self.control_size:
            raise ValueError("Torch F-16 control must have four components")
        if substeps <= 0:
            raise ValueError("substeps must be positive")
        result = state
        for _ in range(int(substeps)):
            result = self._integrate_once(result, control)
        return result


class CompiledTorchF16Runner:
    """Compile one 60 Hz step and keep the 12-step decision loop outside.

    Compiling all table lookups unrolled across a complete 0.2 s decision
    creates an unnecessarily large Inductor graph.  A single compiled physics
    step is cached once and reused, which preserves GPU residency while making
    compilation and recovery practical.
    """

    def __init__(self, model: TorchF16BatchDynamics):
        self.model = model
        self.compiled_step = torch.compile(
            model._integrate_once, fullgraph=True, dynamic=False,
            options={"triton.cudagraphs": False})

    def __call__(self, state: torch.Tensor, control: torch.Tensor,
                 substeps: int = 1) -> torch.Tensor:
        result = state
        for _ in range(int(substeps)):
            result = self.compiled_step(result, control)
        return result
