import torch
import numpy as np
from gymnasium import spaces
from typing import Literal
from .task_base import BaseTask
from ..core.simulatior import AircraftSimulator
from ..core.catalog import Catalog as c
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, SafeReturn
from ..reward_functions import AltitudeReward, PostureReward, EventDrivenReward
from ..utils.utils import get_AO_TA_R, get2d_AO_TA_R, in_range_rad, LLA2NEU, get_root_dir
from ..model.baseline_actor import BaselineActor


class SingleCombatTask(BaseTask):
    def __init__(self, config):
        super().__init__(config)
        self.use_baseline = getattr(self.config, 'use_baseline', False)
        self.use_artillery = getattr(self.config, 'use_artillery', False)
        if self.use_baseline:
            self.baseline_agent = self.load_agent(self.config.baseline_type)

        self.reward_functions = [
            AltitudeReward(self.config),
            PostureReward(self.config),
            EventDrivenReward(self.config)
        ]

        self.termination_conditions = [
            LowAltitude(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            SafeReturn(self.config),
            Timeout(self.config),
        ]

    @property
    def num_agents(self) -> int:
        return 2 if not self.use_baseline else 1

    def load_variables(self):
        self.state_var = [
            c.position_long_gc_deg,             # 0. lontitude  (unit: °)
            c.position_lat_geod_deg,            # 1. latitude   (unit: °)
            c.position_h_sl_m,                  # 2. altitude   (unit: m)
            c.attitude_roll_rad,                # 3. roll       (unit: rad)
            c.attitude_pitch_rad,               # 4. pitch      (unit: rad)
            c.attitude_heading_true_rad,        # 5. yaw        (unit: rad)
            c.velocities_v_north_mps,           # 6. v_north    (unit: m/s)
            c.velocities_v_east_mps,            # 7. v_east     (unit: m/s)
            c.velocities_v_down_mps,            # 8. v_down     (unit: m/s)
            c.velocities_u_mps,                 # 9. v_body_x   (unit: m/s)
            c.velocities_v_mps,                 # 10. v_body_y  (unit: m/s)
            c.velocities_w_mps,                 # 11. v_body_z  (unit: m/s)
            c.velocities_vc_mps,                # 12. vc        (unit: m/s)
            c.accelerations_n_pilot_x_norm,     # 13. a_north   (unit: G)
            c.accelerations_n_pilot_y_norm,     # 14. a_east    (unit: G)
            c.accelerations_n_pilot_z_norm,     # 15. a_down    (unit: G)
        ]
        self.action_var = [
            c.fcs_aileron_cmd_norm,             # [-1., 1.]
            c.fcs_elevator_cmd_norm,            # [-1., 1.]
            c.fcs_rudder_cmd_norm,              # [-1., 1.]
            c.fcs_throttle_cmd_norm,            # [0.4, 0.9]
        ]
        self.render_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_m,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.attitude_heading_true_rad,
        ]

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10, high=10., shape=(15,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = spaces.MultiDiscrete([41, 41, 41, 30])

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
        """
        norm_obs = np.zeros(15)
        ego_obs_list = np.array(env.agents[agent_id].get_property_values(self.state_var))
        enm_obs_list = np.array(env.agents[agent_id].enemies[0].get_property_values(self.state_var))
        # (0) extract feature: [north(km), east(km), down(km), v_n(mh), v_e(mh), v_d(mh)]
        ego_cur_ned = LLA2NEU(*ego_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        enm_cur_ned = LLA2NEU(*enm_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_obs_list[6:9])])
        enm_feature = np.array([*enm_cur_ned, *(enm_obs_list[6:9])])
        # (1) ego info normalization
        norm_obs[0] = ego_obs_list[2] / 5000            # 0. ego altitude   (unit: 5km)
        norm_obs[1] = np.sin(ego_obs_list[3])           # 1. ego_roll_sin
        norm_obs[2] = np.cos(ego_obs_list[3])           # 2. ego_roll_cos
        norm_obs[3] = np.sin(ego_obs_list[4])           # 3. ego_pitch_sin
        norm_obs[4] = np.cos(ego_obs_list[4])           # 4. ego_pitch_cos
        norm_obs[5] = ego_obs_list[9] / 340             # 5. ego v_body_x   (unit: mh)
        norm_obs[6] = ego_obs_list[10] / 340            # 6. ego v_body_y   (unit: mh)
        norm_obs[7] = ego_obs_list[11] / 340            # 7. ego v_body_z   (unit: mh)
        norm_obs[8] = ego_obs_list[12] / 340            # 8. ego vc   (unit: mh)
        # (2) relative info w.r.t enm state
        ego_AO, ego_TA, R, side_flag = get2d_AO_TA_R(ego_feature, enm_feature, return_side=True)
        norm_obs[9] = (enm_obs_list[9] - ego_obs_list[9]) / 340
        norm_obs[10] = (enm_obs_list[2] - ego_obs_list[2]) / 1000
        norm_obs[11] = ego_AO
        norm_obs[12] = ego_TA
        norm_obs[13] = R / 10000
        norm_obs[14] = side_flag
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs

    def normalize_action(self, env, agent_id, action):
        """Convert discrete action index into continuous value.
        """
        if self.use_baseline and agent_id in env.enm_ids:
            action = self.baseline_agent.get_action(env.agents[agent_id])
            return action
        else:
            norm_act = np.zeros(4)
            norm_act[0] = action[0] / 20  - 1.
            norm_act[1] = action[1] / 20 - 1.
            norm_act[2] = action[2] / 20 - 1.
            norm_act[3] = action[3] / 58 + 0.4
            return norm_act

    def reset(self, env):
        """Task-specific reset, include reward function reset.
        """
        self._agent_die_flag = {}
        if self.use_baseline:
            self.baseline_agent.reset()
        return super().reset(env)

    def step(self, env):
        def _orientation_fn(AO):
            if AO >= 0 and AO <= 0.5236:  # [0, pi/6]
                return 1 - AO / 0.5236
            elif AO >= -0.5236 and AO <= 0: # [-pi/6, 0]
                return 1 + AO / 0.5236
            return 0
        def _distance_fn(R):
            if R <=1: # [0, 1km]
                return 1
            elif R > 1 and R <= 3: # [1km, 3km]
                return (3 - R) / 2.
            else:
                return 0
        if self.use_artillery:
            for agent_id in env.agents.keys():
                ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                        env.agents[agent_id].get_velocity()])
                for enm in env.agents[agent_id].enemies:
                    if enm.is_alive:
                        enm_feature = np.hstack([enm.get_position(),
                                                enm.get_velocity()])
                        AO, _, R = get_AO_TA_R(ego_feature, enm_feature)
                        enm.bloods -= _orientation_fn(AO) * _distance_fn(R/1000)
                        # if agent_id == 'A0100' and enm.uid == 'B0100':
                        #     print(f"AO: {AO * 180 / np.pi}, {_orientation_fn(AO)}, dis:{R/1000}, {_distance_fn(R/1000)}")

    def get_reward(self, env, agent_id, info=None):
        if self._agent_die_flag.get(agent_id, False):
            return 0.0, info
        else:
            self._agent_die_flag[agent_id] = not env.agents[agent_id].is_alive
            return super().get_reward(env, agent_id, info=info)

    def load_agent(self, name):
        if name == 'pursue':
            return PursueAgent()
        elif name == 'maneuver':
            return ManeuverAgent(maneuver='n')
        elif name == 'dodge':
            return DodgeMissileAgent()
        elif name == 'straight':
            return StraightFlyAgent()
        else:
            raise NotImplementedError

class HierarchicalSingleCombatTask(SingleCombatTask):

    def __init__(self, config: str):
        super().__init__(config)
        self.lowlevel_policy = BaselineActor()
        self.lowlevel_policy.load_state_dict(torch.load(get_root_dir() + '/model/baseline_model.pt', map_location=torch.device('cpu'), weights_only=True))
        self.lowlevel_policy.eval()
        self.lowlevel_policy.requires_grad_(False)
        self.norm_delta_altitude = np.array([0.1, 0, -0.1])
        heading_span_deg = float(getattr(
            self.config, 'Hierarchical_heading_command_span_deg', 30.0))
        if not np.isfinite(heading_span_deg) or not 15.0 <= heading_span_deg <= 90.0:
            raise ValueError(
                "Hierarchical_heading_command_span_deg must be in [15, 90]")
        self.heading_command_span_deg = heading_span_deg
        self.norm_delta_heading = np.deg2rad(np.linspace(
            -heading_span_deg, heading_span_deg, num=5, dtype=np.float64))
        self.threat_relative_break_skill = bool(getattr(
            self.config, 'ThreatRelativeBreakSkill_enabled', False))
        self.threat_break_option = bool(getattr(
            self.config, 'ThreatRelativeBreakOption_enabled', False))
        self.threat_break_pulse = bool(getattr(
            self.config, 'ThreatRelativeBreakPulse_enabled', False))
        self.threat_break_pulse_steps = int(getattr(
            self.config, 'ThreatRelativeBreakPulse_steps', 4))
        self.threat_break_pulse_urgency = float(getattr(
            self.config, 'ThreatRelativeBreakPulse_urgency_threshold', 0.5))
        self.threat_break_min_steps = int(getattr(
            self.config, 'ThreatRelativeBreakOption_min_steps', 10))
        self.threat_break_clear_steps = int(getattr(
            self.config, 'ThreatRelativeBreakOption_clear_steps', 5))
        self.threat_break_recovery_steps = int(getattr(
            self.config, 'ThreatRelativeBreakOption_recovery_steps', 10))
        self.threat_break_recovery_ta_max_deg = float(getattr(
            self.config,
            'ThreatRelativeBreakOption_recovery_ta_max_deg', 180.0))
        if self.threat_break_option and not self.threat_relative_break_skill:
            raise ValueError(
                "ThreatRelativeBreakOption requires ThreatRelativeBreakSkill")
        if self.threat_break_pulse and not self.threat_relative_break_skill:
            raise ValueError(
                "ThreatRelativeBreakPulse requires ThreatRelativeBreakSkill")
        if self.threat_break_option and self.threat_break_pulse:
            raise ValueError(
                "threat break option and pulse are mutually exclusive")
        if self.threat_break_pulse_steps <= 0:
            raise ValueError("threat break pulse duration must be positive")
        if (not np.isfinite(self.threat_break_pulse_urgency) or
                not 0.0 <= self.threat_break_pulse_urgency <= 1.0):
            raise ValueError("threat break pulse urgency must be in [0, 1]")
        if min(self.threat_break_min_steps, self.threat_break_clear_steps,
               self.threat_break_recovery_steps) <= 0:
            raise ValueError("threat break option durations must be positive")
        if (not np.isfinite(self.threat_break_recovery_ta_max_deg) or
                not 0.0 <= self.threat_break_recovery_ta_max_deg <= 180.0):
            raise ValueError(
                "threat break recovery TA limit must be in [0, 180] degrees")
        self.threat_break_recovery_ta_max = np.deg2rad(
            self.threat_break_recovery_ta_max_deg)
        self._threat_break_state = {}
        self._threat_break_pulse_state = {}
        self.norm_delta_velocity = np.array([0.05, 0, -0.05])

    def load_action_space(self):
        self.action_space = spaces.MultiDiscrete([3, 5, 3])

    @staticmethod
    def _physical_action(low_action):
        return np.asarray([
            low_action[0] / 20.0 - 1.0,
            low_action[1] / 20.0 - 1.0,
            low_action[2] / 20.0 - 1.0,
            low_action[3] / 58.0 + 0.4,
        ], dtype=np.float64)

    def prepare_lowlevel_action(self, env, agent_id, action):
        """Validate one tactical command before fixed-policy inference."""
        action = np.asarray(action)
        if (action.shape != (3,) or not np.all(np.isfinite(action)) or
                not np.all(action == np.floor(action))):
            raise ValueError(
                "hierarchical action must contain three finite integer commands")
        action = action.astype(np.int64)
        command_limits = np.asarray([3, 5, 3], dtype=np.int64)
        if np.any(action < 0) or np.any(action >= command_limits):
            raise ValueError("hierarchical action is outside the action space")
        return action

    def _build_lowlevel_input(self, env, agent_id, action):
        cached_observations = getattr(
            env, "_decision_observations", None)
        raw_obs = (
            cached_observations.get(agent_id)
            if isinstance(cached_observations, dict)
            else None
        )
        if raw_obs is None:
            raw_obs = self.get_obs(env, agent_id)
        input_obs = np.zeros(12, dtype=np.float32)
        input_obs[0] = self.norm_delta_altitude[action[0]]
        input_obs[1] = self._resolve_heading_command(
            raw_obs, int(action[1]), agent_id=agent_id)
        input_obs[2] = self.norm_delta_velocity[action[2]]
        input_obs[3:12] = raw_obs[:9]
        return input_obs

    def normalize_actions(self, env, actions):
        """Run the frozen recurrent low-level controller as one agent batch."""
        normalized = {}
        agent_ids = []
        prepared = {}
        for agent_id in env.agents.keys():
            if self.use_baseline and agent_id in env.enm_ids:
                normalized[agent_id] = self.baseline_agent.get_action(
                    env.agents[agent_id])
            else:
                prepared[agent_id] = self.prepare_lowlevel_action(
                    env, agent_id, actions[agent_id])
                agent_ids.append(agent_id)
        if not agent_ids:
            return normalized

        input_batch = np.stack([
            self._build_lowlevel_input(env, agent_id, prepared[agent_id])
            for agent_id in agent_ids
        ], axis=0)
        rnn_batch = np.concatenate([
            self._inner_rnn_states[agent_id] for agent_id in agent_ids
        ], axis=0)
        with torch.inference_mode():
            low_actions, next_states = self.lowlevel_policy(
                input_batch, rnn_batch)
        low_actions = low_actions.cpu().numpy()
        next_states = next_states.cpu().numpy()
        for index, agent_id in enumerate(agent_ids):
            self._inner_rnn_states[agent_id] = next_states[index:index + 1]
            normalized[agent_id] = self._physical_action(low_actions[index])
        return normalized

    def normalize_action(self, env, agent_id, action):
        """Compatibility path for callers that submit one agent at a time."""
        if self.use_baseline and agent_id in env.enm_ids:
            return self.baseline_agent.get_action(env.agents[agent_id])
        prepared = self.prepare_lowlevel_action(
            env, agent_id, action)
        input_obs = np.expand_dims(
            self._build_lowlevel_input(env, agent_id, prepared), axis=0)
        with torch.inference_mode():
            low_action, next_state = self.lowlevel_policy(
                input_obs, self._inner_rnn_states[agent_id])
        self._inner_rnn_states[agent_id] = next_state.cpu().numpy()
        return self._physical_action(low_action.cpu().numpy().squeeze(0))

    def _resolve_heading_command(self, raw_obs, heading_index, agent_id=None):
        """Resolve a physical or threat-relative heading command.

        During a missile warning, action 0 becomes break-away and action 4
        becomes cut-toward. The middle three bins keep physical semantics.
        """
        heading_index = int(heading_index)
        command = float(self.norm_delta_heading[heading_index])
        if not self.threat_relative_break_skill:
            return command
        observation = np.asarray(raw_obs, dtype=np.float64)
        warning = bool(
            observation[24] >= 0.5 if observation.shape[0] > 24
            else observation[19] > 0.0)
        if self.threat_break_option and agent_id is not None:
            return self._resolve_break_option_command(
                observation, heading_index, str(agent_id), warning, command)
        if self.threat_break_pulse and agent_id is not None:
            return self._resolve_break_pulse_command(
                observation, heading_index, str(agent_id), warning, command)
        if not warning or heading_index not in (0, 4):
            return command
        missile_side = float(observation[20])
        maximum = float(self.norm_delta_heading[-1])
        away = -maximum if missile_side >= 0.0 else maximum
        return away if heading_index == 0 else -away

    def _resolve_break_option_command(self, observation, heading_index,
                                      agent_id, warning, physical_command):
        """Run a short stateful break/reacquisition option.

        Action 0 under warning starts the option.  Break direction is held
        through a minimum duration and until the warning has stayed clear for
        several consecutive decisions.  A short recovery phase then points
        toward the opponent before returning control to the policy.  A new
        warning always interrupts recovery and restarts the break phase.
        """
        state = self._threat_break_state.setdefault(agent_id, {
            "mode": "idle",
            "away": 0.0,
            "minimum_remaining": 0,
            "clear_count": 0,
            "recovery_remaining": 0,
        })
        maximum = float(self.norm_delta_heading[-1])

        def start_break():
            missile_side = float(observation[20])
            state.update({
                "mode": "break",
                "away": -maximum if missile_side >= 0.0 else maximum,
                "minimum_remaining": self.threat_break_min_steps,
                "clear_count": 0,
                "recovery_remaining": 0,
            })

        if state["mode"] == "recovery" and warning:
            start_break()
        elif state["mode"] == "idle" and warning and heading_index == 0:
            start_break()

        if state["mode"] == "break":
            state["minimum_remaining"] = max(
                0, int(state["minimum_remaining"]) - 1)
            if warning:
                state["clear_count"] = 0
            elif state["minimum_remaining"] == 0:
                state["clear_count"] = int(state["clear_count"]) + 1
            if (state["minimum_remaining"] == 0 and
                    state["clear_count"] >= self.threat_break_clear_steps):
                opponent_ta = float(observation[12])
                if (np.isfinite(opponent_ta) and
                        opponent_ta <= self.threat_break_recovery_ta_max):
                    state["mode"] = "recovery"
                    state["recovery_remaining"] = (
                        self.threat_break_recovery_steps)
                else:
                    # The opponent is still pointing toward us.  Do not
                    # convert successful missile defeat into a forced
                    # head-on re-entry; return authority to the PPO policy.
                    state["mode"] = "idle"
                    state["recovery_remaining"] = 0
            else:
                return float(state["away"])

        if state["mode"] == "recovery":
            opponent_side = float(observation[14])
            opponent_angle = float(observation[11])
            if not np.isfinite(opponent_side) or not np.isfinite(opponent_angle):
                recovery_command = 0.0
            else:
                recovery_command = float(np.clip(
                    opponent_side * opponent_angle, -maximum, maximum))
            state["recovery_remaining"] = max(
                0, int(state["recovery_remaining"]) - 1)
            if state["recovery_remaining"] == 0:
                state["mode"] = "idle"
            return recovery_command

        # Preserve the stateless skill semantics when the option is idle.
        if warning and heading_index in (0, 4):
            missile_side = float(observation[20])
            away = -maximum if missile_side >= 0.0 else maximum
            return away if heading_index == 0 else -away
        return physical_command

    def _resolve_break_pulse_command(self, observation, heading_index,
                                     agent_id, warning, physical_command):
        """Hold a high-urgency break briefly, then return control to PPO.

        Unlike the legacy stateful option this pulse never waits for warning
        clearance and never forces opponent reacquisition.  It can retrigger
        only after the previous pulse has ended and the policy selects the
        break action again under a sufficiently urgent warning.
        """
        state = self._threat_break_pulse_state.setdefault(agent_id, {
            "remaining": 0,
            "away": 0.0,
        })
        maximum = float(self.norm_delta_heading[-1])
        if int(state["remaining"]) > 0:
            state["remaining"] = int(state["remaining"]) - 1
            return float(state["away"])

        urgency = float(observation[25]) if observation.shape[0] > 25 else 0.0
        if (warning and urgency >= self.threat_break_pulse_urgency and
                heading_index == 0):
            missile_side = float(observation[20])
            state["away"] = -maximum if missile_side >= 0.0 else maximum
            state["remaining"] = self.threat_break_pulse_steps - 1
            return float(state["away"])

        # Outside an active pulse retain the stateless threat-relative action
        # semantics; importantly, PPO authority is restored immediately.
        if warning and heading_index in (0, 4):
            missile_side = float(observation[20])
            away = -maximum if missile_side >= 0.0 else maximum
            return away if heading_index == 0 else -away
        return physical_command

    def reset(self, env):
        """Task-specific reset, include reward function reset.
        """
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        self._threat_break_state = {}
        self._threat_break_pulse_state = {}
        return super().reset(env)


class StraightFlyAgent:

    def normalize_action(self, action):
        norm_act = np.zeros(4)
        norm_act[0] = action[0] / 20 - 1.   # 0~40 => -1~1
        norm_act[1] = action[1] / 20 - 1.   # 0~40 => -1~1
        norm_act[2] = action[2] / 20 - 1.   # 0~40 => -1~1
        norm_act[3] = action[3] / 58 + 0.4  # 0~29 => 0.4~0.9
        return norm_act

    def get_action(self, sim: AircraftSimulator):
        action = np.array([20, 18.6, 20, 0])
        return self.normalize_action(action)

    def reset(self):
        pass


class BaselineAgent:
    def __init__(self) -> None:
        self.model_path = get_root_dir() + '/model/baseline_model.pt'
        self.actor = BaselineActor()
        self.actor.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu'), weights_only=True))
        self.actor.eval()
        self.state_var = [
            c.delta_altitude,                   #  0. delta_h   (unit: m)
            c.delta_heading,                    #  1. delta_heading  (unit: °)
            c.delta_velocities_u,               #  2. delta_v   (unit: m/s)
            c.attitude_roll_rad,                #  3. roll      (unit: rad)
            c.attitude_pitch_rad,               #  4. pitch     (unit: rad)
            c.velocities_u_mps,                 #  5. v_body_x   (unit: m/s)
            c.velocities_v_mps,                 #  6. v_body_y   (unit: m/s)
            c.velocities_w_mps,                 #  7. v_body_z   (unit: m/s)
            c.velocities_vc_mps,                #  8. vc        (unit: m/s)
            c.position_h_sl_m                   #  9. altitude  (unit: m)
        ]
        self.reset()

    def normalize_action(self, action):
        norm_act = np.zeros(4)
        norm_act[0] = action[0] / 20 - 1.   # 0~40 => -1~1
        norm_act[1] = action[1] / 20 - 1.   # 0~40 => -1~1
        norm_act[2] = action[2] / 20 - 1.   # 0~40 => -1~1
        norm_act[3] = action[3] / 58 + 0.4  # 0~29 => 0.4~0.9
        return norm_act

    def reset(self):
        self.rnn_states = np.zeros((1, 1, 128))

    def set_delta_value(self, sim: AircraftSimulator):
        raise NotImplementedError

    def get_observation(self, sim: AircraftSimulator, delta_value):
        obs = sim.get_property_values(self.state_var)
        norm_obs = np.zeros(12)
        norm_obs[0] = delta_value[0] / 1000          #  0. ego delta altitude  (unit: 1km)
        norm_obs[1] = in_range_rad(delta_value[1])   #  1. ego delta heading   (unit rad)
        norm_obs[2] = delta_value[2] / 340           #  2. ego delta velocities_u  (unit: mh)
        norm_obs[3] = obs[9] / 5000                  #  3. ego_altitude (unit: km)
        norm_obs[4] = np.sin(obs[3])                 #  4. ego_roll_sin
        norm_obs[5] = np.cos(obs[3])                 #  5. ego_roll_cos
        norm_obs[6] = np.sin(obs[4])                 #  6. ego_pitch_sin
        norm_obs[7] = np.cos(obs[4])                 #  7. ego_pitch_cos
        norm_obs[8] = obs[5] / 340                   #  8. ego_v_x   (unit: mh)
        norm_obs[9] = obs[6] / 340                   #  9. ego_v_y    (unit: mh)
        norm_obs[10] = obs[7] / 340                  #  10. ego_v_z    (unit: mh)
        norm_obs[11] = obs[8] / 340                  #  11. ego_vc        (unit: mh)
        norm_obs = np.expand_dims(norm_obs, axis=0)  # dim: (1,12)
        return norm_obs

    def get_action(self, sim: AircraftSimulator):
        delta_value = self.set_delta_value(sim)
        observation = self.get_observation(sim, delta_value)
        _action, self.rnn_states = self.actor(observation, self.rnn_states)
        action = _action.detach().cpu().numpy().squeeze()
        return self.normalize_action(action)


class PursueAgent(BaselineAgent):
    def __init__(self) -> None:
        super().__init__()

    def set_delta_value(self, sim: AircraftSimulator):
        # NOTE: only adapt for 1v1
        ego_x, ego_y, ego_z = sim.get_position()
        ego_vx, ego_vy, ego_vz = sim.get_velocity()
        enm_x, enm_y, enm_z = sim.enemies[0].get_position()
        # delta altitude
        delta_altitude = enm_z - ego_z
        # delta heading
        ego_v = np.linalg.norm([ego_vx, ego_vy])
        delta_x, delta_y = enm_x - ego_x, enm_y - ego_y
        R = np.linalg.norm([delta_x, delta_y])
        proj_dist = delta_x * ego_vx + delta_y * ego_vy
        ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))
        # Scalar 2-D determinant. NumPy 2.x rejects two-component vectors in
        # ``np.cross``; the determinant preserves the historical sign exactly.
        side_flag = np.sign(
            ego_vx * delta_y - ego_vy * delta_x)
        delta_heading = ego_AO * side_flag
        # delta velocity
        delta_velocity = sim.enemies[0].get_property_value(c.velocities_u_mps) - \
                         sim.get_property_value(c.velocities_u_mps)
        return np.array([delta_altitude, delta_heading, delta_velocity])


class ManeuverAgent(BaselineAgent):
    def __init__(self, maneuver: Literal['l', 'r', 'n']) -> None:
        super().__init__()
        self.turn_interval = 30
        self.dodge_missile = True # if set true, start turn when missile is detected
        if maneuver == 'l':
            self.target_heading_list = [0]
        elif maneuver == 'r':
            self.target_heading_list = [np.pi/2, np.pi/2, np.pi/2, np.pi/2]
        elif maneuver == 'n':
            self.target_heading_list = [np.pi, np.pi, np.pi, np.pi]
        # self.target_altitude_list = [8000, 7000, 7500, 5500, 6000, 6000]
        # self.target_velocity_list = [340, 340, 340, 340, 243, 243]
        self.target_altitude_list = [6096] * 4
        self.target_velocity_list = [243] * 4

    def reset(self):
        self.step = 0
        self.rnn_states = np.zeros((1, 1, 128))
        self.init_heading = None

    def set_delta_value(self, sim: AircraftSimulator):
        step_list = np.arange(1, len(self.target_heading_list)+1) * self.turn_interval / 0.2
        cur_heading = sim.get_property_value(c.attitude_heading_true_rad)
        if self.init_heading is None:
            self.init_heading = cur_heading
        if not self.dodge_missile or len(sim.under_missiles) != 0:
            for i, interval in enumerate(step_list):
                if self.step <= interval:
                    break
            delta_heading = self.init_heading + self.target_heading_list[i] - cur_heading
            delta_altitude = self.target_altitude_list[i] - sim.get_property_value(c.position_h_sl_m)
            delta_velocity = self.target_velocity_list[i] - sim.get_property_value(c.velocities_u_mps)
            self.step += 1
        else:
            delta_heading = self.init_heading  - cur_heading
            delta_altitude = 6096 - sim.get_property_value(c.position_h_sl_m)
            delta_velocity = 243 - sim.get_property_value(c.velocities_u_mps)

        return np.array([delta_altitude, delta_heading, delta_velocity])


class DodgeMissileAgent:
    def __init__(self) -> None:
        self.model_path = get_root_dir() + '/model/dodge_missile_model.pt'
        self.actor = BaselineActor(input_dim=21, use_mlp_actlayer=True)
        self.actor.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu'), weights_only=True))
        self.state_var = [
            c.position_long_gc_deg,             # 0. lontitude  (unit: °)
            c.position_lat_geod_deg,            # 1. latitude   (unit: °)
            c.position_h_sl_m,                  # 2. altitude   (unit: m)
            c.attitude_roll_rad,                # 3. roll       (unit: rad)
            c.attitude_pitch_rad,               # 4. pitch      (unit: rad)
            c.attitude_heading_true_rad,        # 5. yaw        (unit: rad)
            c.velocities_v_north_mps,           # 6. v_north    (unit: m/s)
            c.velocities_v_east_mps,            # 7. v_east     (unit: m/s)
            c.velocities_v_down_mps,            # 8. v_down     (unit: m/s)
            c.velocities_u_mps,                 # 9. v_body_x   (unit: m/s)
            c.velocities_v_mps,                 # 10. v_body_y  (unit: m/s)
            c.velocities_w_mps,                 # 11. v_body_z  (unit: m/s)
            c.velocities_vc_mps,                # 12. vc        (unit: m/s)
        ]
        self.reset()

    def get_observation(self, sim: AircraftSimulator):
        norm_obs = np.zeros(21)
        ego_obs_list = np.array(sim.get_property_values(self.state_var))
        enm_obs_list = np.array(sim.enemies[0].get_property_values(self.state_var))
        # (0) extract feature: [north(km), east(km), down(km), v_n(mh), v_e(mh), v_d(mh)]
        ego_cur_ned = LLA2NEU(*ego_obs_list[:3], 120.0, 60.0, 0.0)
        enm_cur_ned = LLA2NEU(*enm_obs_list[:3], 120.0, 60.0, 0.0)
        ego_feature = np.array([*ego_cur_ned, *(ego_obs_list[6:9])])
        enm_feature = np.array([*enm_cur_ned, *(enm_obs_list[6:9])])
        # (1) ego info normalization
        norm_obs[0] = ego_obs_list[2] / 5000            # 0. ego altitude   (unit: 5km)
        norm_obs[1] = np.sin(ego_obs_list[3])           # 1. ego_roll_sin
        norm_obs[2] = np.cos(ego_obs_list[3])           # 2. ego_roll_cos
        norm_obs[3] = np.sin(ego_obs_list[4])           # 3. ego_pitch_sin
        norm_obs[4] = np.cos(ego_obs_list[4])           # 4. ego_pitch_cos
        norm_obs[5] = ego_obs_list[9] / 340             # 5. ego v_body_x   (unit: mh)
        norm_obs[6] = ego_obs_list[10] / 340            # 6. ego v_body_y   (unit: mh)
        norm_obs[7] = ego_obs_list[11] / 340            # 7. ego v_body_z   (unit: mh)
        norm_obs[8] = ego_obs_list[12] / 340            # 8. ego vc   (unit: mh)
        # (2) relative info w.r.t enm state
        ego_AO, ego_TA, R, side_flag = get2d_AO_TA_R(ego_feature, enm_feature, return_side=True)
        norm_obs[9] = (enm_obs_list[9] - ego_obs_list[9]) / 340
        norm_obs[10] = (enm_obs_list[2] - ego_obs_list[2]) / 1000
        norm_obs[11] = ego_AO
        norm_obs[12] = ego_TA
        norm_obs[13] = R / 10000
        norm_obs[14] = side_flag
        # (3) relative missile info
        if len(sim.under_missiles) != 0 and sim.under_missiles[0].is_alive:
            missile_sim = sim.under_missiles[0]
        else:
            missile_sim = None
        if missile_sim is not None:
            missile_feature = np.concatenate((missile_sim.get_position(), missile_sim.get_velocity()))
            ego_AO, ego_TA, R, side_flag = get2d_AO_TA_R(ego_feature, missile_feature, return_side=True)
            norm_obs[15] = (np.linalg.norm(missile_sim.get_velocity()) - ego_obs_list[9]) / 340
            norm_obs[16] = (missile_feature[2] - ego_obs_list[2]) / 1000
            norm_obs[17] = ego_AO
            norm_obs[18] = ego_TA
            norm_obs[19] = R / 10000
            norm_obs[20] = side_flag
        norm_obs = np.expand_dims(norm_obs, axis=0)
        return norm_obs

    def get_action(self, sim: AircraftSimulator):
        obs = self.get_observation(sim)
        _action, self.rnn_states = self.actor(obs, self.rnn_states)
        action = _action.squeeze().detach().cpu().numpy().squeeze()
        return action

    def reset(self):
        self.rnn_states = np.zeros((1, 1, 128))
