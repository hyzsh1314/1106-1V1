"""Batched Torch implementation of the project's generic WVR missile."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TorchWVRMissileParameters:
    navigation_constant: float = 3.0
    maximum_normal_load_g: float = 30.0
    thrust_duration_s: float = 3.0
    maximum_flight_time_s: float = 40.0
    minimum_speed_mps: float = 150.0
    capture_radius_m: float = 50.0
    initial_mass_kg: float = 84.0
    mass_flow_kg_s: float = 6.0
    specific_impulse_s: float = 120.0
    drag_coefficient: float = 0.4
    length_m: float = 2.87
    diameter_m: float = 0.127
    gravity_m_s2: float = 9.81
    nonclosing_timeout_s: float = 5.0


class TorchWVRMissileBatch(torch.nn.Module):
    """Masked missile lifecycle and PN kinematics on CPU or GPU.

    State layout is ``[N,E,U,VN,VE,VU,mass,age,previous_range,
    nonclosing_time,dtheta,dphi,status]``. Status values are inactive=0, flying=1,
    hit=2 and miss=3.  HIT and MISS are absorbing states.
    """

    state_size = 13
    INACTIVE = 0
    FLYING = 1
    HIT = 2
    MISS = 3

    def __init__(self, parameters: TorchWVRMissileParameters | None = None,
                 dt: float = 1.0 / 60.0):
        super().__init__()
        self.parameters = parameters or TorchWVRMissileParameters()
        self.dt = float(dt)
        if self.dt <= 0.0:
            raise ValueError("integration time step must be positive")

    def launch(self, position_neu: torch.Tensor,
               velocity_neu: torch.Tensor) -> torch.Tensor:
        if position_neu.shape[-1] != 3 or velocity_neu.shape[-1] != 3:
            raise ValueError("missile position and velocity must be 3-D")
        p = self.parameters
        scalar_shape = position_neu.shape[:-1]
        mass = torch.full(
            (*scalar_shape, 1), p.initial_mass_kg,
            dtype=position_neu.dtype, device=position_neu.device)
        age = torch.zeros_like(mass)
        previous_range = torch.full_like(mass, torch.inf)
        nonclosing = torch.zeros_like(mass)
        angular_rates = torch.zeros(*scalar_shape, 2,
                                    dtype=position_neu.dtype,
                                    device=position_neu.device)
        status = torch.full_like(mass, float(self.FLYING))
        return torch.cat((position_neu, velocity_neu, mass, age,
                          previous_range, nonclosing, angular_rates,
                          status), dim=-1)

    @staticmethod
    def _segment_distance(relative_start: torch.Tensor,
                          relative_end: torch.Tensor) -> torch.Tensor:
        delta = relative_end - relative_start
        denominator = delta.square().sum(-1).clamp_min(1.0e-16)
        fraction = (-(relative_start * delta).sum(-1) /
                    denominator).clamp(0.0, 1.0)
        return torch.linalg.vector_norm(
            relative_start + fraction.unsqueeze(-1) * delta, dim=-1)

    def forward(self, state: torch.Tensor, target_position_neu: torch.Tensor,
                target_velocity_neu: torch.Tensor,
                target_alive: torch.Tensor | None = None) -> torch.Tensor:
        if state.shape[-1] != self.state_size:
            raise ValueError("Torch missile state must have 13 components")
        p = self.parameters
        position, velocity = state[..., :3], state[..., 3:6]
        mass, age = state[..., 6], state[..., 7]
        previous_range = state[..., 8]
        nonclosing = state[..., 9]
        previous_dtheta, previous_dheading = state[..., 10], state[..., 11]
        status = state[..., 12]
        was_flying = status == float(self.FLYING)
        target_valid = (torch.ones_like(was_flying) if target_alive is None
                        else target_alive.to(dtype=torch.bool))
        flying = was_flying & target_valid

        relative = target_position_neu - position
        relative_velocity = target_velocity_neu - velocity
        horizontal_range = torch.linalg.vector_norm(
            relative[..., :2], dim=-1).clamp_min(1.0e-6)
        distance = torch.linalg.vector_norm(relative, dim=-1).clamp_min(1.0e-6)
        speed = torch.linalg.vector_norm(velocity, dim=-1).clamp_min(1.0e-6)
        theta = torch.asin((velocity[..., 2] / speed).clamp(-1.0, 1.0))
        dbeta = (
            relative_velocity[..., 1] * relative[..., 0]
            - relative_velocity[..., 0] * relative[..., 1]
        ) / horizontal_range.square()
        deps = (
            relative_velocity[..., 2] * horizontal_range.square()
            - relative[..., 2]
            * (relative[..., :2] * relative_velocity[..., :2]).sum(-1)
        ) / (distance.square() * horizontal_range)
        ny = (p.navigation_constant * speed / p.gravity_m_s2
              * torch.cos(theta) * dbeta)
        nz = (p.navigation_constant * speed / p.gravity_m_s2 * deps
              + torch.cos(theta))
        ny = ny.clamp(-p.maximum_normal_load_g, p.maximum_normal_load_g)
        nz = nz.clamp(-p.maximum_normal_load_g, p.maximum_normal_load_g)

        horizontal_heading = torch.atan2(velocity[..., 1], velocity[..., 0])
        d_heading = p.gravity_m_s2 / speed * (
            ny / torch.cos(theta).clamp_min(1.0e-3))
        d_theta = p.gravity_m_s2 / speed * (nz - torch.cos(theta))
        base_area = torch.pi * (0.5 * p.diameter_m) ** 2
        projected_area = base_area + torch.sqrt(
            torch.sin(previous_dtheta).square()
            + torch.sin(previous_dheading).square()
        ) * p.diameter_m * p.length_m
        density = 1.225 * torch.exp(-position[..., 2] / 9300.0)
        drag = (0.5 * p.drag_coefficient * projected_area
                * density * speed.square())
        thrust = torch.where(
            age < p.thrust_duration_s,
            torch.full_like(age, p.gravity_m_s2 * p.specific_impulse_s
                            * p.mass_flow_kg_s),
            torch.zeros_like(age))
        acceleration = (thrust - drag) / mass.clamp_min(1.0)
        next_speed = (speed + self.dt * (
            acceleration - p.gravity_m_s2 * torch.sin(theta))).clamp_min(0.0)
        next_heading = horizontal_heading + self.dt * d_heading
        next_theta = theta + self.dt * d_theta
        next_velocity = torch.stack((
            next_speed * torch.cos(next_theta) * torch.cos(next_heading),
            next_speed * torch.cos(next_theta) * torch.sin(next_heading),
            next_speed * torch.sin(next_theta)), dim=-1)
        next_position = position + self.dt * velocity
        next_age = age + self.dt
        next_mass = torch.where(
            age < p.thrust_duration_s,
            (mass - self.dt * p.mass_flow_kg_s).clamp_min(1.0), mass)
        next_nonclosing = torch.where(
            distance > previous_range,
            nonclosing + self.dt, torch.zeros_like(nonclosing))

        target_start = target_position_neu - self.dt * target_velocity_neu
        closest = self._segment_distance(
            target_start - position, target_position_neu - next_position)
        hit = flying & (closest <= p.capture_radius_m)
        expired = flying & ~hit & (
            (next_age > p.maximum_flight_time_s)
            | (next_speed < p.minimum_speed_mps)
            | (next_nonclosing >= p.nonclosing_timeout_s)
            | ~torch.isfinite(next_position).all(-1)
            | ~torch.isfinite(next_velocity).all(-1))
        next_status = torch.where(
            hit, torch.full_like(status, float(self.HIT)),
            torch.where(expired, torch.full_like(status, float(self.MISS)),
                        status))
        candidate = torch.cat((
            next_position, next_velocity, next_mass.unsqueeze(-1),
            next_age.unsqueeze(-1), distance.unsqueeze(-1),
            next_nonclosing.unsqueeze(-1), d_theta.unsqueeze(-1),
            d_heading.unsqueeze(-1), next_status.unsqueeze(-1)), dim=-1)
        dead_target_state = torch.cat((
            state[..., :12],
            torch.full_like(state[..., 12:13], float(self.MISS))), dim=-1)
        active_candidate = torch.where(
            target_valid.unsqueeze(-1), candidate, dead_target_state)
        return torch.where(was_flying.unsqueeze(-1), active_candidate, state)
