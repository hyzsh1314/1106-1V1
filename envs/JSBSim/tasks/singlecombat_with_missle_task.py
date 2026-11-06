import numpy as np
from gymnasium import spaces
from collections import deque

from .singlecombat_task import SingleCombatTask, HierarchicalSingleCombatTask
from ..reward_functions import (
    AltitudeReward, MissileEvasionReward, MissilePostureReward,
    ShootPenaltyReward,
)
from ..reward_functions.pbgr_reward_functions import PBGRDiscountedPostureReward
from ..reward_functions.pbgr_reward_functions import ThreatConditionedPotentialReward
from ..core.simulatior import MissileSimulator
from ..utils.utils import LLA2NEU, get_AO_TA_R
from .weapon_contract import WeaponEvent, evaluate_shot


class SingleCombatDodgeMissileTask(SingleCombatTask):
    """This task aims at training agent to dodge missile attacking
    """
    def __init__(self, config):
        super().__init__(config)

        self.max_attack_angle = getattr(self.config, 'max_attack_angle', 180)
        self.max_attack_distance = getattr(self.config, 'max_attack_distance', np.inf)
        self.min_attack_interval = getattr(self.config, 'min_attack_interval', 125)
        self.reward_functions = [
            MissilePostureReward(self.config),
            AltitudeReward(self.config),
            MissileEvasionReward(self.config),
        ]

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10, high=10., shape=(21,))

    def get_obs(self, env, agent_id):
        """
        Convert simulation states into the format of observation_space

        ------
        Returns: (np.ndarray)
        - ego info
            - [0] ego altitude           (unit: 5km)
            - [1] ego_roll_sin
            - [2] ego_roll_cos
            - [3] ego_pitch_sin
            - [4] ego_pitch_cos
            - [5] ego v_body_x           (unit: mh)
            - [6] ego v_body_y           (unit: mh)
            - [7] ego v_body_z           (unit: mh)
            - [8] ego_vc                 (unit: mh)
        - relative enm info
            - [9] delta_v_body_x         (unit: mh)
            - [10] delta_altitude        (unit: km)
            - [11] ego_AO                (unit: rad) [0, pi]
            - [12] ego_TA                (unit: rad) [0, pi]
            - [13] relative distance     (unit: 10km)
            - [14] side_flag             1 or 0 or -1
        - relative missile info
            - [15] delta_v_body_x
            - [16] delta altitude
            - [17] ego_AO
            - [18] ego_TA
            - [19] relative distance
            - [20] side flag
        """
        norm_obs = np.zeros(21)
        ego_obs_list = np.array(env.agents[agent_id].get_property_values(self.state_var))
        enm_obs_list = np.array(env.agents[agent_id].enemies[0].get_property_values(self.state_var))
        # (0) extract feature: [north(km), east(km), down(km), v_n(mh), v_e(mh), v_d(mh)]
        ego_cur_ned = LLA2NEU(*ego_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        enm_cur_ned = LLA2NEU(*enm_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *ego_obs_list[6:9]])
        enm_feature = np.array([*enm_cur_ned, *enm_obs_list[6:9]])
        # (1) ego info normalization
        norm_obs[0] = ego_obs_list[2] / 5000
        norm_obs[1] = np.sin(ego_obs_list[3])
        norm_obs[2] = np.cos(ego_obs_list[3])
        norm_obs[3] = np.sin(ego_obs_list[4])
        norm_obs[4] = np.cos(ego_obs_list[4])
        norm_obs[5] = ego_obs_list[9] / 340
        norm_obs[6] = ego_obs_list[10] / 340
        norm_obs[7] = ego_obs_list[11] / 340
        norm_obs[8] = ego_obs_list[12] / 340
        # (2) relative enm info
        ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, enm_feature, return_side=True)
        norm_obs[9] = (enm_obs_list[9] - ego_obs_list[9]) / 340
        norm_obs[10] = (enm_obs_list[2] - ego_obs_list[2]) / 1000
        norm_obs[11] = ego_AO
        norm_obs[12] = ego_TA
        norm_obs[13] = R / 10000
        norm_obs[14] = side_flag
        # (3) relative missile info
        missile_sim = env.agents[agent_id].check_missile_warning()
        if missile_sim is not None:
            missile_feature = np.concatenate((missile_sim.get_position(), missile_sim.get_velocity()))
            ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, missile_feature, return_side=True)
            norm_obs[15] = (np.linalg.norm(missile_sim.get_velocity()) - ego_obs_list[9]) / 340
            norm_obs[16] = (missile_feature[2] - ego_obs_list[2]) / 1000
            norm_obs[17] = ego_AO
            norm_obs[18] = ego_TA
            norm_obs[19] = R / 10000
            norm_obs[20] = side_flag
        return norm_obs

    def reset(self, env):
        """Reset fighter blood & missile status
        """
        self._last_shoot_time = {agent_id: -self.min_attack_interval for agent_id in env.agents.keys()}
        self.remaining_missiles = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        self.lock_duration = {agent_id: deque(maxlen=int(1 / env.time_interval)) for agent_id in env.agents.keys()}
        return super().reset(env)

    def prepare_weapon_actions(self, env, decision_step):
        """Apply rule-based launch decisions at the actor-observation state."""

        for agent_id, agent in env.agents.items():
            # [Rule-based missile launch]
            target_aircraft = agent.enemies[0]
            target = target_aircraft.get_position() - agent.get_position()
            heading = agent.get_velocity()
            distance = np.linalg.norm(target)
            attack_angle = np.rad2deg(np.arccos(np.clip(np.sum(target * heading) / (distance * np.linalg.norm(heading) + 1e-8), -1, 1)))
            valid_track = bool(
                agent.is_alive and target_aircraft.is_alive and
                np.isfinite(distance) and distance > 1e-8)
            self.lock_duration[agent_id].append(
                valid_track and attack_angle < self.max_attack_angle)
            shoot_interval = int(decision_step) - self._last_shoot_time[agent_id]

            shoot_flag = valid_track and np.sum(self.lock_duration[agent_id]) >= self.lock_duration[agent_id].maxlen \
                and distance <= self.max_attack_distance and self.remaining_missiles[agent_id] > 0 and shoot_interval >= self.min_attack_interval
            if shoot_flag:
                new_missile_uid = agent_id + str(self.remaining_missiles[agent_id])
                env.add_temp_simulator(
                    MissileSimulator.create(parent=agent, target=target_aircraft, uid=new_missile_uid))
                self.remaining_missiles[agent_id] -= 1
                agent.num_left_missiles = self.remaining_missiles[agent_id]
                self._last_shoot_time[agent_id] = int(decision_step)

    def step(self, env):
        SingleCombatTask.step(self, env)


class HierarchicalSingleCombatDodgeMissileTask(HierarchicalSingleCombatTask, SingleCombatDodgeMissileTask):

    def __init__(self, config: str):
        HierarchicalSingleCombatTask.__init__(self, config)

        self.reward_functions = [
            MissilePostureReward(self.config),
            AltitudeReward(self.config),
            MissileEvasionReward(self.config),
        ]

    def load_observation_space(self):
        return SingleCombatDodgeMissileTask.load_observation_space(self)

    def load_action_space(self):
        return HierarchicalSingleCombatTask.load_action_space(self)

    def get_obs(self, env, agent_id):
        return SingleCombatDodgeMissileTask.get_obs(self, env, agent_id)

    def normalize_action(self, env, agent_id, action):
        return HierarchicalSingleCombatTask.normalize_action(self, env, agent_id, action)

    def reset(self, env):
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        return SingleCombatDodgeMissileTask.reset(self, env)

    def step(self, env):
        return SingleCombatDodgeMissileTask.step(self, env)


class SingleCombatShootMissileTask(SingleCombatDodgeMissileTask):
    def __init__(self, config):
        super().__init__(config)

        self.reward_functions = self._build_reward_functions()

    def _build_reward_functions(self):
        variant = str(getattr(self.config, "RewardVariant", "legacy"))
        if variant == "R0_terminal":
            return []
        if variant in {"R1_attack_pbrs", "R1_timeout025"}:
            return [PBGRDiscountedPostureReward(self.config)]
        if variant == "R2_threat_pbrs":
            return [ThreatConditionedPotentialReward(self.config)]
        if variant == "R3_constraints":
            return [
                ThreatConditionedPotentialReward(self.config),
                AltitudeReward(self.config),
                ShootPenaltyReward(self.config),
            ]
        if variant != "legacy":
            raise ValueError(f"unsupported RewardVariant: {variant}")
        return [
            PBGRDiscountedPostureReward(self.config),
            AltitudeReward(self.config),
            ShootPenaltyReward(self.config),
        ]

    def load_observation_space(self):
        # The first 21 features retain the historical layout.  The final
        # three expose ammunition, cooldown readiness and pre-action fire
        # legality so the policy does not have to infer hidden weapon state.
        threat_v2 = bool(getattr(
            self.config, "ThreatObservation_v2_enabled", False))
        self.observation_space = spaces.Box(
            low=-10, high=10., shape=(26 if threat_v2 else 24,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle, shoot control
        self.action_space = spaces.Tuple([spaces.MultiDiscrete([41, 41, 41, 30]), spaces.Discrete(2)])

    def get_reward(self, env, agent_id, info=None):
        """Close discounted shaping exactly once when an aircraft dies."""

        agent = env.agents[agent_id]
        was_dead = bool(self._agent_die_flag.get(agent_id, False))
        first_death = (not agent.is_alive) and (not was_dead)
        if agent.is_alive:
            return SingleCombatTask.get_reward(
                self, env, agent_id, info=info)
        self._agent_die_flag[agent_id] = True
        if not first_death:
            return 0.0, ({} if info is None else dict(info))
        info = {} if info is None else dict(info)
        components = info.setdefault(
            "reward_components", {}).setdefault(str(agent_id), {})
        reward = 0.0
        for reward_function in self.reward_functions:
            settle = getattr(reward_function, "settle_terminal", None)
            if callable(settle):
                value = float(settle(agent_id))
                reward += value
                components[reward_function.__class__.__name__] = value
        return float(reward), info
    
    def get_obs(self, env, agent_id):
        kinematic_obs = super().get_obs(env, agent_id)
        agent = env.agents[agent_id]
        remaining = int(self.remaining_missiles.get(
            agent_id, getattr(agent, "num_missiles", 0)))
        initial = max(1, int(self._initial_missiles.get(
            agent_id, getattr(agent, "num_missiles", 1))))
        elapsed = int(env.current_step) - int(self._last_shoot_time[agent_id])
        cooldown_fraction = float(np.clip(
            elapsed / max(1, self.min_attack_interval), 0.0, 1.0))
        legality, _ = evaluate_shot(
            agent=agent,
            requested=False,
            remaining_missiles=remaining,
            current_step=int(env.current_step),
            last_shot_step=int(self._last_shoot_time[agent_id]),
            max_attack_angle=self.max_attack_angle,
            max_attack_distance=self.max_attack_distance,
            min_attack_interval=self.min_attack_interval,
        )
        weapon_obs = np.asarray([
            remaining / initial,
            cooldown_fraction,
            float(legality.can_fire),
        ], dtype=np.float64)
        observation = np.concatenate((kinematic_obs, weapon_obs))
        if not bool(getattr(
                self.config, "ThreatObservation_v2_enabled", False)):
            return observation
        missile = agent.check_missile_warning()
        warning = float(missile is not None)
        urgency = 0.0
        if missile is not None:
            relative_position = missile.get_position() - agent.get_position()
            distance = float(np.linalg.norm(relative_position))
            if distance > 1.0:
                relative_velocity = (
                    missile.get_velocity() - agent.get_velocity())
                closing_speed = float(-np.dot(
                    relative_position, relative_velocity) / distance)
                if closing_speed > 1.0:
                    time_to_go = distance / closing_speed
                    horizon = float(getattr(
                        self.config,
                        "ThreatObservation_time_to_go_horizon_s", 30.0))
                    urgency = float(np.clip(
                        1.0 - time_to_go / max(horizon, 1.0), 0.0, 1.0))
        return np.concatenate((observation, np.asarray(
            [warning, urgency], dtype=np.float64)))
    
    def normalize_action(self, env, agent_id, action):
        self._shoot_action[agent_id] = action[-1]
        return super().normalize_action(env, agent_id, action[:-1].astype(np.int32))
    
    def reset(self, env):
        self._shoot_action = {agent_id: 0 for agent_id in env.agents.keys()}
        self.remaining_missiles = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        self._initial_missiles = dict(self.remaining_missiles)
        self._weapon_events = []
        self._step_weapon_events = []
        self._weapon_legality = {}
        self._known_missile_outcomes = set()
        super().reset(env)

    def _append_weapon_event(self, event):
        payload = event.to_dict() if isinstance(event, WeaponEvent) else dict(event)
        self._step_weapon_events.append(payload)
        self._weapon_events.append(payload)

    def _record_resolved_missiles(self, env):
        for agent_id, agent in env.agents.items():
            for missile in agent.launch_missiles:
                missile_id = str(missile.uid)
                if (not missile.is_done or
                        missile_id in self._known_missile_outcomes):
                    continue
                target = getattr(missile, "target_aircraft", None)
                reason = str(getattr(
                    missile, "resolution_reason",
                    "capture" if missile.is_success else "expired"))
                role = str(getattr(missile, "resolution_role", ""))
                if missile.is_success and role == "redundant_hit":
                    event_type = "redundant_hit"
                elif missile.is_success:
                    event_type = "hit"
                elif reason == "target_already_resolved":
                    event_type = "invalidated"
                else:
                    event_type = "miss"
                self._append_weapon_event(WeaponEvent(
                    event_type=event_type,
                    step=int(env.current_step),
                    shooter_id=str(agent_id),
                    target_id=(None if target is None else str(target.uid)),
                    missile_id=missile_id,
                    reason=reason,
                ))
                self._known_missile_outcomes.add(missile_id)
    
    def prepare_weapon_actions(self, env, decision_step):
        """Apply requested launches before aircraft dynamics advance."""

        self._step_weapon_events = []
        self._weapon_legality = {}
        for agent_id, agent in env.agents.items():
            legality, target_aircraft = evaluate_shot(
                agent=agent,
                requested=bool(self._shoot_action[agent_id]),
                remaining_missiles=self.remaining_missiles[agent_id],
                current_step=int(decision_step),
                last_shot_step=int(self._last_shoot_time[agent_id]),
                max_attack_angle=self.max_attack_angle,
                max_attack_distance=self.max_attack_distance,
                min_attack_interval=self.min_attack_interval,
            )
            self._weapon_legality[str(agent_id)] = legality.to_dict()
            if legality.accepted:
                new_missile_uid = agent_id + str(self.remaining_missiles[agent_id])
                env.add_temp_simulator(
                    MissileSimulator.create(parent=agent, target=target_aircraft, uid=new_missile_uid))
                self.remaining_missiles[agent_id] -= 1
                agent.num_left_missiles = self.remaining_missiles[agent_id]
                self._last_shoot_time[agent_id] = int(decision_step)
                self._append_weapon_event(WeaponEvent(
                    event_type="launch",
                    step=int(env.current_step),
                    shooter_id=str(agent_id),
                    target_id=str(target_aircraft.uid),
                    missile_id=str(new_missile_uid),
                    reason="accepted",
                ))
            elif legality.requested:
                self._append_weapon_event(WeaponEvent(
                    event_type="rejected_request",
                    step=int(env.current_step),
                    shooter_id=str(agent_id),
                    target_id=legality.target_id,
                    missile_id=None,
                    reason=legality.reason,
                ))

    def get_weapon_action_mask(self, env, agent_id):
        """Return the pre-action [hold, fire] mask for one aircraft."""

        agent = env.agents[agent_id]
        legality, _ = evaluate_shot(
            agent=agent,
            requested=False,
            remaining_missiles=self.remaining_missiles[agent_id],
            current_step=int(env.current_step),
            last_shot_step=int(self._last_shoot_time[agent_id]),
            max_attack_angle=self.max_attack_angle,
            max_attack_distance=self.max_attack_distance,
            min_attack_interval=self.min_attack_interval,
        )
        return np.asarray([True, legality.can_fire], dtype=np.bool_)

    def get_weapon_step_info(self):
        requests = sum(
            int(item.get("requested", False))
            for item in self._weapon_legality.values())
        accepted = sum(
            int(item.get("accepted", False))
            for item in self._weapon_legality.values())
        opportunities = sum(
            int(item.get("can_fire", False))
            for item in self._weapon_legality.values())
        event_types = (
            "launch", "rejected_request", "hit", "redundant_hit",
            "miss", "invalidated",
        )
        counts = {
            event_type: sum(
                event.get("event_type") == event_type
                for event in self._weapon_events)
            for event_type in event_types
        }
        unresolved = max(
            0, counts["launch"] - counts["hit"] -
            counts["redundant_hit"] - counts["miss"] -
            counts["invalidated"])
        return {
            "weapon_schema_version": "simple-missile-v2",
            "weapon_events": list(self._step_weapon_events),
            "weapon_event_counts": counts,
            "weapon_launch_count": counts["launch"],
            "weapon_hit_count": counts["hit"],
            "weapon_redundant_hit_count": counts["redundant_hit"],
            "weapon_miss_count": counts["miss"],
            "weapon_invalidated_count": counts["invalidated"],
            "weapon_unresolved_count": unresolved,
            "remaining_missiles": dict(self.remaining_missiles),
            "weapon_legality": dict(self._weapon_legality),
            "weapon_fire_requests": requests,
            "weapon_accepted_launches": accepted,
            "weapon_rejected_requests": requests - accepted,
            "weapon_legal_opportunities": opportunities,
        }

    def step(self, env):
        SingleCombatTask.step(self, env)
        self._record_resolved_missiles(env)


class HierarchicalSingleCombatShootTask(HierarchicalSingleCombatTask, SingleCombatShootMissileTask):
    def __init__(self, config: str):
        HierarchicalSingleCombatTask.__init__(self, config)
        self.reward_functions = self._build_reward_functions()

    def load_observation_space(self):
        return SingleCombatShootMissileTask.load_observation_space(self)

    def load_action_space(self):
        # altitude control + heading control + velocity control + shoot control
        self.action_space = spaces.Tuple([spaces.MultiDiscrete([3, 5, 3]), spaces.Discrete(2)])

    def get_obs(self, env, agent_id):
        return SingleCombatShootMissileTask.get_obs(self, env, agent_id)

    def prepare_lowlevel_action(self, env, agent_id, action):
        """Validate the weapon command and return the flight command."""
        if self.use_baseline and agent_id in env.enm_ids:
            return action
        action = np.asarray(action)
        limits = np.asarray([3, 5, 3, 2], dtype=np.int64)
        if (action.shape != (4,) or not np.all(np.isfinite(action)) or
                not np.all(action == np.floor(action))):
            raise ValueError(
                "hierarchical shoot action must contain four finite integer commands")
        action = action.astype(np.int64)
        if np.any(action < 0) or np.any(action >= limits):
            raise ValueError("hierarchical shoot action is outside the action space")
        self._shoot_action[agent_id] = action[-1]
        return HierarchicalSingleCombatTask.prepare_lowlevel_action(
            self, env, agent_id, action[:-1])

    def normalize_action(self, env, agent_id, action):
        """Convert one high-level action through the compatibility path."""
        return HierarchicalSingleCombatTask.normalize_action(
            self, env, agent_id, action)

    def reset(self, env):
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        SingleCombatShootMissileTask.reset(self, env)

    def step(self, env):
        SingleCombatShootMissileTask.step(self, env)
