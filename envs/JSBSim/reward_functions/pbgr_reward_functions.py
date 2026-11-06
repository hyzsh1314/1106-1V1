"""PBGR-only reward repairs that must not change historical baselines."""

import numpy as np

from .event_driven_reward import EventDrivenReward
from .missile_posture_reward import MissilePostureReward
from .posture_reward import PostureReward
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R


class PBGREventDrivenReward(EventDrivenReward):
    """Emit each crash, shoot-down and successful-hit event exactly once."""

    def __init__(self, config):
        super().__init__(config)
        # BaseRewardFunction keys scales by the concrete class name.  PBGR
        # intentionally reuses the frozen public EventDrivenReward keys.
        self.reward_scale = getattr(config, "EventDrivenReward_scale", 1.0)
        self.is_potential = getattr(config, "EventDrivenReward_potential", False)
        self.reward_item_names = ["EventDrivenReward"]
        self._processed_agent_events = set()
        self._processed_missiles = set()
        self._event_totals = {}

    def reset(self, task, env):
        self._processed_agent_events.clear()
        self._processed_missiles.clear()
        self._event_totals.clear()
        return BaseRewardFunction.reset(self, task, env)

    def get_reward(self, task, env, agent_id):
        reward = 0.0
        agent_event = None
        if env.agents[agent_id].is_shotdown:
            agent_event = (agent_id, "shotdown")
        elif env.agents[agent_id].is_crash:
            agent_event = (agent_id, "crash")
        if (agent_event is not None and
                agent_event not in self._processed_agent_events):
            reward -= 200.0
            self._processed_agent_events.add(agent_event)
        for missile in env.agents[agent_id].launch_missiles:
            missile_key = (agent_id, missile.uid)
            is_kill = (
                missile.is_success and
                str(getattr(missile, "resolution_role", "kill")) == "kill"
            )
            if (is_kill and
                    missile_key not in self._processed_missiles):
                reward += 200.0
                self._processed_missiles.add(missile_key)
        if self.is_potential:
            self._event_totals[agent_id] = (
                self._event_totals.get(agent_id, 0.0) + reward)
            reward = self._event_totals[agent_id]
        return self._process(reward, agent_id)


class PBGRMissilePostureReward(MissilePostureReward):
    """Keep immutable, per-agent missile-velocity history for PBGR."""

    def __init__(self, config):
        super().__init__(config)
        self.reward_scale = getattr(config, "MissilePostureReward_scale", 1.0)
        self.is_potential = getattr(
            config, "MissilePostureReward_potential", False)
        self.reward_item_names = ["MissilePostureReward"]
        self.previous_missile_v = {}

    def reset(self, task, env):
        self.previous_missile_v.clear()
        return BaseRewardFunction.reset(self, task, env)

    def get_reward(self, task, env, agent_id):
        missile_sim = env.agents[agent_id].check_missile_warning()
        if missile_sim is None:
            self.previous_missile_v.pop(agent_id, None)
            reward = 0.0
        else:
            missile_v = missile_sim.get_velocity()
            aircraft_v = env.agents[agent_id].get_velocity()
            missile_uid = str(missile_sim.uid)
            previous_entry = self.previous_missile_v.get(agent_id)
            if (previous_entry is None or
                    previous_entry[0] != missile_uid):
                # A newly observed threat has no valid temporal predecessor.
                # Its first frame is a neutral initialization, not a velocity
                # jump from whichever missile previously occupied this agent.
                reward = 0.0
            else:
                previous = previous_entry[1]
                v_decrease = (
                    (np.linalg.norm(previous) - np.linalg.norm(missile_v)) /
                    340.0 * self.reward_scale
                )
                denominator = (
                    np.linalg.norm(missile_v) * np.linalg.norm(aircraft_v)
                )
                angle = (0.0 if denominator <= 1e-8 else
                         np.dot(missile_v, aircraft_v) / denominator)
                if angle < 0:
                    reward = angle / (max(v_decrease, 0.0) + 1.0)
                else:
                    reward = angle * max(v_decrease, 0.0)
            self.previous_missile_v[agent_id] = (
                missile_uid,
                np.asarray(missile_v, dtype=np.float64).copy(),
            )
        self.reward_trajectory[agent_id].append([reward])
        return reward


class PBGRDiscountedPostureReward(PostureReward):
    """Opt-in discounted potential shaping for the V22 platform baseline."""

    def __init__(self, config):
        super().__init__(config)
        self.reward_scale = float(getattr(config, "PostureReward_scale", 1.0))
        self.orientation_version = getattr(config, "PostureReward_orientation_version", "v2")
        self.range_version = getattr(config, "PostureReward_range_version", "v3")
        self.target_dist = float(getattr(config, "PostureReward_target_dist", 3.0))
        self.enemy_aggregation = getattr(config, "PostureReward_enemy_aggregation", "mean_alive")
        if self.enemy_aggregation not in {"sum", "mean_alive"}:
            raise ValueError("unsupported posture enemy aggregation")
        self.orientation_fn = self.get_orientation_function(self.orientation_version)
        self.range_fn = self.get_range_funtion(self.range_version)
        self.gamma = float(getattr(config, "PotentialReward_gamma", 0.99))
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("PotentialReward_gamma must be in (0,1]")
        self.is_potential = True
        self.reward_item_names = [
            "DiscountedPostureReward", "DiscountedPostureReward_orn",
            "DiscountedPostureReward_range",
        ]

    def _potential(self, env, agent_id):
        pair_rewards, orientation_rewards, range_rewards = [], [], []
        ego = env.agents[agent_id]
        ego_feature = np.hstack((ego.get_position(), ego.get_velocity()))
        for enemy in ego.enemies:
            if self.enemy_aggregation == "mean_alive" and not enemy.is_alive:
                continue
            enemy_feature = np.hstack((enemy.get_position(), enemy.get_velocity()))
            orientation, target_aspect, distance = get_AO_TA_R(ego_feature, enemy_feature)
            orientation_reward = self.orientation_fn(orientation, target_aspect)
            range_reward = self.range_fn(distance / 1000.0)
            orientation_rewards.append(orientation_reward)
            range_rewards.append(range_reward)
            pair_rewards.append(orientation_reward * range_reward)
        if not pair_rewards:
            return 0.0, 0.0, 0.0
        if self.enemy_aggregation == "mean_alive":
            raw = float(np.mean(pair_rewards))
            orientation = float(np.mean(orientation_rewards))
            range_reward = float(np.mean(range_rewards))
        else:
            raw = float(np.sum(pair_rewards))
            orientation = float(orientation_rewards[-1])
            range_reward = float(range_rewards[-1])
        return self.reward_scale * raw, orientation, range_reward

    @staticmethod
    def _terminal(env):
        # Import lazily to avoid a module cycle: tasks import reward classes.
        from ..tasks.weapon_contract import resolve_team_outcome

        return bool(resolve_team_outcome(
            env, timeout=int(env.current_step) >= int(env.max_steps)
        ).terminated)

    def reset(self, task, env):
        self.pre_rewards.clear()
        for agent_id in env.agents:
            potential, _, _ = self._potential(env, agent_id)
            self.pre_rewards[agent_id] = potential
        self.reward_trajectory.clear()

    def get_reward(self, task, env, agent_id):
        potential, orientation, range_reward = self._potential(env, agent_id)
        successor = 0.0 if self._terminal(env) else potential
        previous = float(self.pre_rewards[agent_id])
        reward = self.gamma * successor - previous
        self.pre_rewards[agent_id] = successor
        self.reward_trajectory[agent_id].append([reward, orientation, range_reward])
        return float(reward)

    def settle_terminal(self, agent_id):
        """Settle one agent's cached potential when that agent dies first."""

        previous = float(self.pre_rewards[agent_id])
        reward = -previous
        self.pre_rewards[agent_id] = 0.0
        self.reward_trajectory[agent_id].append([reward, 0.0, 0.0])
        return reward


class ThreatConditionedPotentialReward(PBGRDiscountedPostureReward):
    """Bounded PBRS that switches from attack geometry to missile survival.

    With no incoming missile the potential is the clipped attack posture.
    Under warning it becomes a bounded combination of missile distance and
    estimated time-to-go, so a rational defensive manoeuvre is not punished
    merely for surrendering nose-on geometry.
    """

    def __init__(self, config):
        super().__init__(config)
        self.reward_scale = float(getattr(
            config, "ThreatConditionedPotentialReward_scale", 0.1))
        self.tgo_scale = float(getattr(
            config, "ThreatConditionedPotentialReward_tgo_scale_s", 15.0))
        self.distance_scale = float(getattr(
            config, "ThreatConditionedPotentialReward_distance_scale_m",
            getattr(config, "max_attack_distance", 14000.0)))
        if self.reward_scale < 0.0 or self.tgo_scale <= 0.0 or self.distance_scale <= 0.0:
            raise ValueError("invalid threat-conditioned potential configuration")
        self.reward_item_names = [
            "ThreatConditionedPotentialReward",
            "ThreatConditionedPotentialReward_attack",
            "ThreatConditionedPotentialReward_escape",
            "ThreatConditionedPotentialReward_threat",
        ]

    def _potential(self, env, agent_id):
        attack, _, _ = super()._potential(env, agent_id)
        attack = float(np.clip(attack, -self.reward_scale, self.reward_scale))
        agent = env.agents[agent_id]
        missile = agent.check_missile_warning()
        if missile is None:
            return attack, attack, 0.0, 0.0

        relative = (np.asarray(missile.get_position(), dtype=np.float64) -
                    np.asarray(agent.get_position(), dtype=np.float64))
        relative_velocity = (
            np.asarray(missile.get_velocity(), dtype=np.float64) -
            np.asarray(agent.get_velocity(), dtype=np.float64))
        distance = float(np.linalg.norm(relative))
        if distance <= 1.0e-8:
            time_to_go = 0.0
        else:
            closing_speed = max(
                0.0, -float(np.dot(relative, relative_velocity)) / distance)
            time_to_go = distance / max(closing_speed, 1.0)
        distance_score = float(np.clip(
            distance / self.distance_scale, 0.0, 1.0))
        tgo_score = float(np.clip(
            time_to_go / self.tgo_scale, 0.0, 1.0))
        escape = self.reward_scale * 0.5 * (distance_score + tgo_score)
        return float(escape), attack, float(escape), 1.0

    def reset(self, task, env):
        self.pre_rewards.clear()
        for agent_id in env.agents:
            potential, *_ = self._potential(env, agent_id)
            self.pre_rewards[agent_id] = potential
        self.reward_trajectory.clear()

    def get_reward(self, task, env, agent_id):
        potential, attack, escape, threat = self._potential(env, agent_id)
        successor = 0.0 if self._terminal(env) else potential
        previous = float(self.pre_rewards[agent_id])
        reward = self.gamma * successor - previous
        self.pre_rewards[agent_id] = successor
        self.reward_trajectory[agent_id].append(
            [reward, attack, escape, threat])
        return float(reward)
