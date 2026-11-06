"""Single-process, static-shape GPU vector environment for 1v1 WVR combat."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .f16_dynamics import (
    CompiledTorchF16Runner, TorchF16BatchDynamics, quaternion_to_matrix,
)
from .missile import TorchWVRMissileBatch


@dataclass(frozen=True)
class TorchWVRConfig:
    num_envs: int = 96
    sim_freq_hz: int = 60
    substeps_per_decision: int = 12
    max_episode_steps: int = 1000
    missiles_per_aircraft: int = 2
    min_launch_interval_steps: int = 25
    max_attack_distance_m: float = 14000.0
    max_attack_angle_deg: float = 45.0
    altitude_limit_m: float = 2500.0
    initial_min_range_m: float = 8000.0
    initial_max_range_m: float = 16000.0
    initial_min_altitude_m: float = 5000.0
    initial_max_altitude_m: float = 7500.0
    initial_max_altitude_delta_m: float = 600.0
    initial_min_speed_m_s: float = 198.12
    initial_max_speed_m_s: float = 274.32
    initial_aspect_probabilities: tuple[float, float, float] = (
        0.15, 0.35, 0.50)
    potential_gamma: float = 0.99
    terminal_reward: float = 1.0
    reward_mode: str = "competitive"
    attack_quality_weight: float = 0.0
    timeout_penalty: float = 0.20
    exhausted_penalty: float = 0.10
    observation_version: str = "v4"
    use_dynamic_launch_envelope: bool = True
    minimum_attack_distance_m: float = 500.0
    reset_pool_size: int = 8192


class TorchWVRVecEnv:
    """Full-tensor 1v1 environment with masked aircraft and missile states.

    The environment accepts normalized low-level aircraft controls with shape
    ``[E, 2, 4]`` and binary fire requests with shape ``[E, 2]``.  It keeps
    all per-environment state on one device and has no per-environment loop.
    """

    observation_size = 39

    def __init__(self, config: TorchWVRConfig | None = None,
                 device: str | torch.device = "cuda", seed: int = 85,
                 compile_dynamics: bool = True):
        self.config = config or TorchWVRConfig()
        self.device = torch.device(device)
        if self.config.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if self.config.sim_freq_hz != 60:
            raise ValueError("the validated Torch environment uses 60 Hz")
        if self.config.reward_mode not in {"competitive", "engagement"}:
            raise ValueError(
                "reward_mode must be 'competitive' or 'engagement'")
        if self.config.attack_quality_weight < 0.0:
            raise ValueError("attack_quality_weight must be non-negative")
        observation_sizes = {"v2": 24, "v3": 29, "v4": 39}
        if self.config.observation_version not in observation_sizes:
            raise ValueError(
                "observation_version must be 'v2', 'v3' or 'v4'")
        self.observation_size = observation_sizes[
            self.config.observation_version]
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.dynamics = TorchF16BatchDynamics(
            dt=1.0 / self.config.sim_freq_hz).to(self.device).eval()
        self.dynamics_runner = (
            CompiledTorchF16Runner(self.dynamics)
            if compile_dynamics else self.dynamics)
        self.missile = TorchWVRMissileBatch(
            dt=1.0 / self.config.sim_freq_hz).to(self.device).eval()
        self.missile_runner = self.missile
        if compile_dynamics and self.device.type == "cuda":
            self.missile_runner = torch.compile(
                self.missile, fullgraph=True, dynamic=False,
                options={"triton.cudagraphs": False})
        self.aircraft_state = torch.empty(
            self.config.num_envs, 2, self.dynamics.state_size,
            device=self.device)
        self.missile_state = torch.zeros(
            self.config.num_envs, 2, self.config.missiles_per_aircraft,
            self.missile.state_size, device=self.device)
        self.alive = torch.ones(
            self.config.num_envs, 2, dtype=torch.bool, device=self.device)
        self.remaining_missiles = torch.full(
            (self.config.num_envs, 2), self.config.missiles_per_aircraft,
            dtype=torch.long, device=self.device)
        self.last_launch_step = torch.full(
            (self.config.num_envs, 2),
            -self.config.min_launch_interval_steps,
            dtype=torch.long, device=self.device)
        self.episode_step = torch.zeros(
            self.config.num_envs, dtype=torch.long, device=self.device)
        self.episode_id = torch.full(
            (self.config.num_envs,), -1, dtype=torch.long, device=self.device)
        self.previous_potential = torch.zeros(
            self.config.num_envs, 2, device=self.device)
        # Generate reset states once. Selecting from this deterministic pool
        # avoids constructing a complete random batch on every device step.
        self._reset_pool = self._sample_initial_states(
            self.config.reset_pool_size)
        self.reset()

    def _uniform(self, *shape):
        return torch.rand(*shape, device=self.device, generator=self.generator)

    def _sample_initial_states(self, count: int | None = None):
        c = self.config
        count = c.num_envs if count is None else int(count)
        separation = c.initial_min_range_m + (
            c.initial_max_range_m - c.initial_min_range_m) * self._uniform(count)
        bearing = 2.0 * torch.pi * self._uniform(count)
        direction = torch.stack((torch.cos(bearing), torch.sin(bearing)), -1)
        midpoint = (2.0 * self._uniform(count, 2) - 1.0) * 2000.0
        positions = torch.zeros(count, 2, 3, device=self.device)
        positions[:, 0, :2] = midpoint - 0.5 * separation[:, None] * direction
        positions[:, 1, :2] = midpoint + 0.5 * separation[:, None] * direction
        center_altitude = c.initial_min_altitude_m + (
            c.initial_max_altitude_m - c.initial_min_altitude_m
        ) * self._uniform(count)
        altitude_delta = (2.0 * self._uniform(count) - 1.0) * (
            c.initial_max_altitude_delta_m)
        positions[:, 0, 2] = -(center_altitude - 0.5 * altitude_delta)
        positions[:, 1, 2] = -(center_altitude + 0.5 * altitude_delta)

        head_probability, crossing_probability, pursuit_probability = (
            c.initial_aspect_probabilities)
        if abs(head_probability + crossing_probability
               + pursuit_probability - 1.0) > 1.0e-6:
            raise ValueError("initial aspect probabilities must sum to one")
        aspect_draw = self._uniform(count)
        head_on = aspect_draw < head_probability
        crossing = ((aspect_draw >= head_probability)
                    & (aspect_draw < head_probability
                       + crossing_probability))
        headings = torch.empty(count, 2, device=self.device)
        headings[:, 0] = bearing
        headings[:, 1] = torch.where(
            head_on, bearing + torch.pi,
            torch.where(crossing, bearing + 0.5 * torch.pi, bearing))
        # Random colour-role exchange preserves the joint distribution while
        # preventing one colour from always receiving the pursuit geometry.
        swap = self._uniform(count) < 0.5
        positions = torch.where(
            swap[:, None, None], positions.flip(1), positions)
        headings = torch.where(swap[:, None], headings.flip(1), headings)
        speed = c.initial_min_speed_m_s + (
            c.initial_max_speed_m_s - c.initial_min_speed_m_s
        ) * self._uniform(count, 2)
        return self.dynamics.level_state(
            positions.reshape(-1, 3), headings.reshape(-1), speed.reshape(-1)
        ).reshape(count, 2, self.dynamics.state_size)

    def reset(self, mask: torch.Tensor | None = None):
        if mask is None:
            mask = torch.ones(
                self.config.num_envs, dtype=torch.bool, device=self.device)
        else:
            mask = mask.to(device=self.device, dtype=torch.bool)
        self.episode_id = torch.where(
            mask, self.episode_id + 1, self.episode_id)
        pool_index = (
            self.episode_id * self.config.num_envs
            + torch.arange(self.config.num_envs, device=self.device)
        ).remainder(self.config.reset_pool_size)
        sampled = self._reset_pool[pool_index]
        self.aircraft_state = torch.where(
            mask[:, None, None], sampled, self.aircraft_state)
        self.missile_state = torch.where(
            mask[:, None, None, None],
            torch.zeros_like(self.missile_state), self.missile_state)
        self.alive = torch.where(
            mask[:, None], torch.ones_like(self.alive), self.alive)
        self.remaining_missiles = torch.where(
            mask[:, None],
            torch.full_like(self.remaining_missiles,
                            self.config.missiles_per_aircraft),
            self.remaining_missiles)
        self.last_launch_step = torch.where(
            mask[:, None],
            torch.full_like(self.last_launch_step,
                            -self.config.min_launch_interval_steps),
            self.last_launch_step)
        self.episode_step = torch.where(
            mask, torch.zeros_like(self.episode_step), self.episode_step)
        potential = self._posture_potential(self.aircraft_state)
        self.previous_potential = torch.where(
            mask[:, None], potential, self.previous_potential)
        return self.observations()

    def _geometry(self, state: torch.Tensor | None = None):
        state = self.aircraft_state if state is None else state
        rotation = quaternion_to_matrix(state[..., 6:10])
        velocity_ned = torch.matmul(
            rotation, state[..., 3:6].unsqueeze(-1)).squeeze(-1)
        enemy_state = state.flip(1)
        enemy_velocity = velocity_ned.flip(1)
        relative = enemy_state[..., :3] - state[..., :3]
        distance = torch.linalg.vector_norm(relative, dim=-1).clamp_min(1.0)
        speed = torch.linalg.vector_norm(velocity_ned, dim=-1).clamp_min(1.0)
        enemy_speed = torch.linalg.vector_norm(
            enemy_velocity, dim=-1).clamp_min(1.0)
        ao = torch.acos(((relative * velocity_ned).sum(-1) /
                         (distance * speed)).clamp(-1.0, 1.0))
        ta = torch.acos(((-relative * enemy_velocity).sum(-1) /
                         (distance * enemy_speed)).clamp(-1.0, 1.0))
        side = torch.sign(
            velocity_ned[..., 0] * relative[..., 1]
            - velocity_ned[..., 1] * relative[..., 0])
        return rotation, velocity_ned, relative, distance, ao, ta, side

    def _launch_envelope_components(self, geometry):
        rotation, velocity_ned, relative, distance, ao, ta, _ = geometry
        angle_limit = torch.deg2rad(torch.tensor(
            self.config.max_attack_angle_deg, device=self.device))
        dynamic_range = torch.full_like(
            distance, self.config.max_attack_distance_m)
        _, _, los_rate, closure = self._los_features(
            rotation, velocity_ned, relative, distance)
        if self.config.use_dynamic_launch_envelope:
            # Conservative kinematic launch envelope fitted to the generic
            # missile grid scan. Positive closure increases available range;
            # high LOS rate and tail-chase geometry reduce it.
            closure_credit = 8.0 * closure.clamp(-150.0, 550.0)
            tail_penalty = 2200.0 * ((1.0 - torch.cos(ta)) * 0.5)
            rate_penalty = 5000.0 * (
                torch.linalg.vector_norm(los_rate, dim=-1) / 0.20
            ).clamp(0.0, 1.0)
            dynamic_range = (
                7500.0 + closure_credit - tail_penalty - rate_penalty
            ).clamp(2500.0, self.config.max_attack_distance_m)
        return angle_limit, dynamic_range, closure, los_rate

    def _attack_quality(self, geometry=None):
        """Smooth quality of the current kinematic launch solution.

        This is a bounded diagnostic/potential, not a launch or hit reward.
        It stays differentiable across the hard legal-fire boundary so PBRS
        can provide credit before the aircraft first enters the envelope.
        """
        geometry = self._geometry() if geometry is None else geometry
        _, _, _, distance, ao, _, _ = geometry
        angle_limit, dynamic_range, closure, los_rate = (
            self._launch_envelope_components(geometry))
        upper_range = torch.sigmoid((dynamic_range - distance) / 1800.0)
        lower_range = torch.sigmoid(
            (distance - self.config.minimum_attack_distance_m) / 250.0)
        angle_quality = torch.sigmoid((angle_limit - ao) / 0.20)
        closing_quality = 0.5 + 0.5 * torch.tanh(closure / 250.0)
        los_stability = torch.exp(-(
            torch.linalg.vector_norm(los_rate, dim=-1) / 0.20).square())
        support = 0.50 + 0.25 * closing_quality + 0.25 * los_stability
        return (upper_range * lower_range * angle_quality * support).clamp(
            0.0, 1.0)

    def legal_fire_mask(self, geometry=None):
        geometry = self._geometry() if geometry is None else geometry
        _, _, _, distance, ao, _, _ = geometry
        cooldown = ((self.episode_step[:, None] - self.last_launch_step)
                    >= self.config.min_launch_interval_steps)
        angle_limit, dynamic_range, _, _ = self._launch_envelope_components(
            geometry)
        return (
            self.alive & self.alive.flip(1)
            & (self.remaining_missiles > 0) & cooldown
            & (distance >= self.config.minimum_attack_distance_m)
            & (distance <= self.config.max_attack_distance_m)
            & (distance <= dynamic_range)
            & (ao <= angle_limit))

    def _los_features(self, rotation, velocity_ned, relative, distance):
        """Signed body-frame LOS angles/rates and inertial closure speed."""
        relative_velocity = velocity_ned.flip(1) - velocity_ned
        body_relative = torch.matmul(
            rotation.transpose(-1, -2), relative.unsqueeze(-1)).squeeze(-1)
        body_relative_velocity = torch.matmul(
            rotation.transpose(-1, -2),
            relative_velocity.unsqueeze(-1)).squeeze(-1)
        body_rates = self.aircraft_state[..., 10:13]
        body_relative_velocity = body_relative_velocity - torch.linalg.cross(
            body_rates, body_relative, dim=-1)
        x, y, z = body_relative.unbind(-1)
        vx, vy, vz = body_relative_velocity.unbind(-1)
        horizontal_sq = (x.square() + y.square()).clamp_min(1.0)
        horizontal = torch.sqrt(horizontal_sq)
        azimuth = torch.atan2(y, x)
        elevation = torch.atan2(-z, horizontal)
        azimuth_rate = (x * vy - y * vx) / horizontal_sq
        horizontal_rate = (x * vx + y * vy) / horizontal
        elevation_rate = (
            -vz * horizontal + z * horizontal_rate
        ) / distance.square().clamp_min(1.0)
        closure = -(
            relative * relative_velocity
        ).sum(-1) / distance.clamp_min(1.0)
        angles = torch.stack((azimuth, elevation), dim=-1)
        rates = torch.stack((azimuth_rate, elevation_rate), dim=-1)
        return angles, body_relative, rates, closure

    def _missile_observation_features(self, rotation, velocity_ned):
        """Nearest incoming threat and own-shot state for each aircraft."""
        slots = self.config.missiles_per_aircraft
        incoming = self.missile_state.flip(1)
        incoming_flying = incoming[..., 12] == float(self.missile.FLYING)
        own_flying = self.missile_state[..., 12] == float(self.missile.FLYING)
        aircraft_position_neu, aircraft_velocity_neu = self._aircraft_neu()
        relative_neu = incoming[..., :3] - aircraft_position_neu.unsqueeze(2)
        relative_velocity_neu = (
            incoming[..., 3:6] - aircraft_velocity_neu.unsqueeze(2))
        relative_ned = torch.stack((
            relative_neu[..., 0], relative_neu[..., 1],
            -relative_neu[..., 2]), dim=-1)
        relative_velocity_ned = torch.stack((
            relative_velocity_neu[..., 0], relative_velocity_neu[..., 1],
            -relative_velocity_neu[..., 2]), dim=-1)
        distance = torch.linalg.vector_norm(relative_ned, dim=-1).clamp_min(1.0)
        closure = -(
            relative_ned * relative_velocity_ned
        ).sum(-1) / distance
        tgo = distance / closure.clamp_min(1.0)
        priority = torch.where(
            incoming_flying & (closure > 1.0), tgo,
            torch.where(
                incoming_flying, 1000.0 + distance / 1000.0,
                torch.full_like(distance, torch.inf)))
        selected = priority.argmin(dim=-1)
        gather_vector = selected[..., None, None].expand(-1, -1, 1, 3)
        gather_scalar = selected[..., None]
        chosen_relative = torch.gather(
            relative_ned, 2, gather_vector).squeeze(2)
        chosen_relative_velocity = torch.gather(
            relative_velocity_ned, 2, gather_vector).squeeze(2)
        chosen_distance = torch.gather(
            distance, 2, gather_scalar).squeeze(2)
        chosen_closure = torch.gather(
            closure, 2, gather_scalar).squeeze(2)
        chosen_tgo = torch.gather(tgo, 2, gather_scalar).squeeze(2)
        chosen_missile_velocity = torch.gather(
            incoming[..., 3:6], 2, gather_vector).squeeze(2)
        body_relative = torch.matmul(
            rotation.transpose(-1, -2),
            chosen_relative.unsqueeze(-1)).squeeze(-1)
        body_relative_velocity = torch.matmul(
            rotation.transpose(-1, -2),
            chosen_relative_velocity.unsqueeze(-1)).squeeze(-1)
        body_relative_velocity = body_relative_velocity - torch.linalg.cross(
            self.aircraft_state[..., 10:13], body_relative, dim=-1)
        x, y, z = body_relative.unbind(-1)
        vx, vy, vz = body_relative_velocity.unbind(-1)
        horizontal_sq = (x.square() + y.square()).clamp_min(1.0)
        horizontal = torch.sqrt(horizontal_sq)
        azimuth = torch.atan2(y, x)
        elevation = torch.atan2(-z, horizontal)
        azimuth_rate = (x * vy - y * vx) / horizontal_sq
        horizontal_rate = (x * vx + y * vy) / horizontal
        elevation_rate = (
            -vz * horizontal + z * horizontal_rate
        ) / chosen_distance.square().clamp_min(1.0)
        has_incoming = incoming_flying.any(dim=-1)
        mask = has_incoming.float()
        features = torch.zeros(
            self.config.num_envs, 2, 10, device=self.device)
        features[..., 0] = incoming_flying.sum(-1).float() / slots
        features[..., 1] = own_flying.sum(-1).float() / slots
        features[..., 2] = mask * azimuth / torch.pi
        features[..., 3] = mask * elevation / (0.5 * torch.pi)
        features[..., 4] = mask * azimuth_rate
        features[..., 5] = mask * elevation_rate
        features[..., 6] = mask * chosen_closure / 1000.0
        features[..., 7] = mask * chosen_tgo.clamp(0.0, 20.0) / 20.0
        features[..., 8] = mask * chosen_distance / 10000.0
        features[..., 9] = mask * torch.linalg.vector_norm(
            chosen_missile_velocity, dim=-1) / 1000.0
        return features

    def observations(self):
        geometry = self._geometry()
        rotation, velocity_ned, relative, distance, ao, ta, side = geometry
        roll = torch.atan2(rotation[..., 2, 1], rotation[..., 2, 2])
        pitch = torch.asin((-rotation[..., 2, 0]).clamp(-1.0, 1.0))
        enemy = self.aircraft_state.flip(1)
        obs = torch.zeros(
            self.config.num_envs, 2, self.observation_size,
            device=self.device)
        obs[..., 0] = -self.aircraft_state[..., 2] / 5000.0
        obs[..., 1] = torch.sin(roll)
        obs[..., 2] = torch.cos(roll)
        obs[..., 3] = torch.sin(pitch)
        obs[..., 4] = torch.cos(pitch)
        obs[..., 5:8] = self.aircraft_state[..., 3:6] / 340.0
        obs[..., 8] = torch.linalg.vector_norm(
            self.aircraft_state[..., 3:6], dim=-1) / 340.0
        obs[..., 9] = (enemy[..., 3] - self.aircraft_state[..., 3]) / 340.0
        obs[..., 10] = (-enemy[..., 2] + self.aircraft_state[..., 2]) / 1000.0
        obs[..., 11] = ao
        obs[..., 12] = ta
        obs[..., 13] = distance / 10000.0
        obs[..., 14] = side
        obs[..., 15:18] = velocity_ned / 340.0
        obs[..., 18:21] = velocity_ned.flip(1) / 340.0
        obs[..., 21] = (self.remaining_missiles.float()
                        / self.config.missiles_per_aircraft)
        obs[..., 22] = (self.remaining_missiles.flip(1).float()
                        / self.config.missiles_per_aircraft)
        obs[..., 23] = self.legal_fire_mask(geometry).float()
        if self.config.observation_version in {"v3", "v4"}:
            angles, _, rates, closure = self._los_features(
                rotation, velocity_ned, relative, distance)
            obs[..., 24] = angles[..., 0] / torch.pi
            obs[..., 25] = angles[..., 1] / (0.5 * torch.pi)
            obs[..., 26:28] = rates
            obs[..., 28] = closure / 340.0
        if self.config.observation_version == "v4":
            obs[..., 29:39] = self._missile_observation_features(
                rotation, velocity_ned)
        return obs.clamp(-10.0, 10.0)

    def _posture_potential(self, state):
        geometry_data = self._geometry(state)
        _, _, _, distance, ao, ta, _ = geometry_data
        # +1 means the target is near own boresight while own aircraft is on
        # the target's tail; -1 describes the reverse disadvantage.  The
        # range term is shared by both sides.  It cancels in competitive mode,
        # but provides an explicit approach gradient in engagement curriculum.
        geometry = 0.5 * (torch.cos(ao) - torch.cos(ta))
        range_km = distance / 1000.0
        range_envelope = torch.exp(-0.5 * ((range_km - 5.0) / 4.0).square())
        base = 0.65 * geometry + 0.35 * range_envelope
        if self.config.attack_quality_weight <= 0.0:
            return base
        return base + self.config.attack_quality_weight * self._attack_quality(
            geometry_data)

    def _aircraft_neu(self):
        rotation = quaternion_to_matrix(self.aircraft_state[..., 6:10])
        velocity_ned = torch.matmul(
            rotation, self.aircraft_state[..., 3:6].unsqueeze(-1)).squeeze(-1)
        position_neu = torch.stack((
            self.aircraft_state[..., 0], self.aircraft_state[..., 1],
            -self.aircraft_state[..., 2]), dim=-1)
        velocity_neu = torch.stack((
            velocity_ned[..., 0], velocity_ned[..., 1],
            -velocity_ned[..., 2]), dim=-1)
        return position_neu, velocity_neu

    def _launch(self, fire_request: torch.Tensor,
                legal: torch.Tensor | None = None):
        legal = self.legal_fire_mask() if legal is None else legal
        accepted = fire_request.to(dtype=torch.bool, device=self.device) & legal
        position, velocity = self._aircraft_neu()
        launched = self.missile.launch(position, velocity)
        slot_index = self.config.missiles_per_aircraft - self.remaining_missiles
        slots = torch.arange(
            self.config.missiles_per_aircraft, device=self.device)
        slot_mask = accepted[..., None] & (slots == slot_index[..., None])
        self.missile_state = torch.where(
            slot_mask[..., None], launched[..., None, :], self.missile_state)
        self.remaining_missiles = self.remaining_missiles - accepted.long()
        self.last_launch_step = torch.where(
            accepted, self.episode_step[:, None], self.last_launch_step)
        return accepted

    def _missile_substep(self):
        position, velocity = self._aircraft_neu()
        target_position = position.flip(1).unsqueeze(2).expand(
            -1, -1, self.config.missiles_per_aircraft, -1)
        target_velocity = velocity.flip(1).unsqueeze(2).expand_as(
            target_position)
        target_alive = self.alive.flip(1).unsqueeze(2).expand(
            -1, -1, self.config.missiles_per_aircraft)
        flat = self.missile_state.reshape(-1, self.missile.state_size)
        previous_status = flat[:, 12].reshape(
            self.config.num_envs, 2, self.config.missiles_per_aircraft)
        next_flat = self.missile_runner(
            flat, target_position.reshape(-1, 3),
            target_velocity.reshape(-1, 3), target_alive.reshape(-1))
        self.missile_state = next_flat.reshape_as(self.missile_state)
        next_status = self.missile_state[..., 12]
        new_hit = ((previous_status != float(self.missile.HIT))
                   & (next_status == float(self.missile.HIT)))
        hit_by_side = new_hit.any(dim=-1)
        killed = hit_by_side.flip(1)
        self.alive = self.alive & ~killed
        return new_hit

    def step(self, controls: torch.Tensor, fire_request: torch.Tensor):
        controls = controls.to(device=self.device, dtype=torch.float32)
        if controls.shape != (self.config.num_envs, 2, 4):
            raise ValueError("controls must have shape [num_envs, 2, 4]")
        if fire_request.shape != (self.config.num_envs, 2):
            raise ValueError("fire_request must have shape [num_envs, 2]")
        requested = fire_request.to(dtype=torch.bool, device=self.device)
        prelaunch_geometry = self._geometry()
        legal = self.legal_fire_mask(prelaunch_geometry)
        prelaunch_attack_quality = self._attack_quality(prelaunch_geometry)
        accepted = self._launch(requested, legal)
        hit_count = torch.zeros(
            self.config.num_envs, 2, dtype=torch.long, device=self.device)
        for _ in range(self.config.substeps_per_decision):
            previous_state = self.aircraft_state
            next_state = self.dynamics_runner(
                previous_state.reshape(-1, self.dynamics.state_size),
                controls.reshape(-1, 4), substeps=1).reshape_as(previous_state)
            self.aircraft_state = torch.where(
                self.alive[..., None], next_state, previous_state)
            hit_count += self._missile_substep().sum(-1)

        altitude = -self.aircraft_state[..., 2]
        crashed = self.alive & (altitude < self.config.altitude_limit_m)
        self.alive = self.alive & ~crashed
        self.episode_step += 1
        timeout = self.episode_step >= self.config.max_episode_steps
        eliminated = ~self.alive.all(dim=-1)
        flying_missile = self.missile_state[..., 12] == float(
            self.missile.FLYING)
        weapons_exhausted = (
            (self.remaining_missiles == 0).all(dim=-1)
            & ~flying_missile.any(dim=(1, 2)))
        # With no gun model, an environment in which both aircraft have spent
        # every missile and every launched missile is resolved has no reachable
        # kill event. Terminate it immediately instead of adding hundreds of
        # reward-free transitions before the time limit.
        done = timeout | eliminated | weapons_exhausted

        potential = self._posture_potential(self.aircraft_state)
        # In episodic PBRS the absorbing terminal state has zero potential.
        # Retaining terminal posture here would bias the win/loss objective.
        raw = (self.config.potential_gamma
               * potential * (~done)[:, None] - self.previous_potential)
        if self.config.reward_mode == "competitive":
            competitive = 0.5 * (raw[:, 0] - raw[:, 1])
            reward = torch.stack((competitive, -competitive), dim=-1)
        else:
            # Curriculum-only individual PBRS retains the shared range signal;
            # mutual disengagement is therefore no longer reward-neutral.
            reward = raw
        red_only = self.alive[:, 0] & ~self.alive[:, 1]
        blue_only = self.alive[:, 1] & ~self.alive[:, 0]
        outcome = red_only.float() - blue_only.float()
        terminal = torch.stack((outcome, -outcome), dim=-1)
        reward = reward + done[:, None] * self.config.terminal_reward * terminal
        if self.config.reward_mode == "engagement":
            reward = reward - timeout[:, None] * self.config.timeout_penalty
            reward = reward - (
                weapons_exhausted & ~eliminated
            )[:, None] * self.config.exhausted_penalty
        zero_sum_error = torch.abs(reward.sum(-1))
        info = {
            "legal_fire_opportunities": legal,
            "fire_requests": requested,
            "accepted_launches": accepted,
            "rejected_fire_requests": requested & ~legal,
            "prelaunch_attack_quality": prelaunch_attack_quality,
            "hits": hit_count,
            "crashes": crashed,
            "timeout": timeout & ~eliminated,
            "weapons_exhausted": weapons_exhausted & ~eliminated,
            "outcome_red": outcome,
            "episode_length": self.episode_step.clone(),
            "zero_sum_error": zero_sum_error,
        }
        self.previous_potential = potential
        # Masked reset stays on the device and avoids a per-step GPU sync.
        self.reset(done)
        observation = self.observations()
        return observation, reward, done, info
