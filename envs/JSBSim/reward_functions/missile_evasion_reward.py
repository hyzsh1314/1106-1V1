"""Event reward for missile-evasion tasks without offensive kill credit."""

from .reward_function_base import BaseRewardFunction


class MissileEvasionReward(BaseRewardFunction):
    """Reward survival and defeated incoming missiles exactly once."""

    def __init__(self, config):
        super().__init__(config)
        self.survival_reward = float(getattr(
            config, "MissileEvasionReward_survival", 200.0))
        self.failure_penalty = float(getattr(
            config, "MissileEvasionReward_failure", 200.0))
        self.defeat_reward = float(getattr(
            config, "MissileEvasionReward_defeat", 20.0))
        self._processed_agents = set()
        self._processed_missiles = set()

    def reset(self, task, env):
        self._processed_agents.clear()
        self._processed_missiles.clear()
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        agent = env.agents[agent_id]
        reward = 0.0
        if (not agent.is_alive and agent_id not in self._processed_agents):
            reward -= self.failure_penalty
            self._processed_agents.add(agent_id)

        for missile in agent.under_missiles:
            key = (agent_id, str(missile.uid))
            if (missile.is_done and not missile.is_success and
                    key not in self._processed_missiles):
                reward += self.defeat_reward
                self._processed_missiles.add(key)

        incoming_alive = any(
            missile.is_alive for missile in agent.under_missiles)
        mission_survived = (
            agent.is_alive and
            (int(env.current_step) >= int(env.max_steps) or
             (all(not enemy.is_alive for enemy in agent.enemies) and
              not incoming_alive))
        )
        survival_key = (agent_id, "survived")
        if mission_survived and survival_key not in self._processed_agents:
            reward += self.survival_reward
            self._processed_agents.add(survival_key)
        return self._process(reward, agent_id)
