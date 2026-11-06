import numpy as np
from .reward_function_base import BaseRewardFunction


class MissilePostureReward(BaseRewardFunction):
    """
    MissilePostureReward
    Use the velocity attenuation
    """
    def __init__(self, config):
        super().__init__(config)
        self.previous_missile_v = {}

    def reset(self, task, env):
        self.previous_missile_v.clear()
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        """
        Reward is velocity attenuation of the missile

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        reward = 0
        missile_sim = env.agents[agent_id].check_missile_warning()
        if missile_sim is not None:
            missile_v = np.asarray(
                missile_sim.get_velocity(), dtype=np.float64).copy()
            aircraft_v = env.agents[agent_id].get_velocity()
            missile_uid = str(missile_sim.uid)
            previous_entry = self.previous_missile_v.get(agent_id)
            if previous_entry is None or previous_entry[0] != missile_uid:
                reward = 0.0
            else:
                previous = previous_entry[1]
                speed_decrease = (
                    np.linalg.norm(previous) - np.linalg.norm(missile_v)
                ) / 340.0
                denominator = (
                    np.linalg.norm(missile_v) * np.linalg.norm(aircraft_v)
                )
                angle = (0.0 if denominator <= 1e-8 else
                         float(np.dot(missile_v, aircraft_v) / denominator))
                if angle < 0.0:
                    reward = angle / (max(speed_decrease, 0.0) + 1.0)
                else:
                    reward = angle * max(speed_decrease, 0.0)
                reward *= self.reward_scale
            self.previous_missile_v[agent_id] = (missile_uid, missile_v)
        else:
            self.previous_missile_v.pop(agent_id, None)
            reward = 0
        self.reward_trajectory[agent_id].append([reward])
        return reward
