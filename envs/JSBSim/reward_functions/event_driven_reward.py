from .reward_function_base import BaseRewardFunction


class EventDrivenReward(BaseRewardFunction):
    """
    EventDrivenReward
    Achieve reward when the following event happens:
    - Shot down by missile: -200
    - Crash accidentally: -200
    - Shoot down other aircraft: +200
    """
    def __init__(self, config):
        super().__init__(config)
        self._processed_agent_events = set()
        self._processed_missiles = set()
        self._event_totals = {}

    def reset(self, task, env):
        self._processed_agent_events.clear()
        self._processed_missiles.clear()
        self._event_totals.clear()
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        """
        Reward is the sum of all the events.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        reward = 0
        agent_event = None
        if env.agents[agent_id].is_shotdown:
            agent_event = (agent_id, "shotdown")
        elif env.agents[agent_id].is_crash:
            agent_event = (agent_id, "crash")
        if (agent_event is not None and
                agent_event not in self._processed_agent_events):
            reward -= 200
            self._processed_agent_events.add(agent_event)
        for missile in env.agents[agent_id].launch_missiles:
            missile_key = (agent_id, str(missile.uid))
            is_kill = (
                missile.is_success and
                str(getattr(missile, "resolution_role", "kill")) == "kill"
            )
            if is_kill and missile_key not in self._processed_missiles:
                reward += 200
                self._processed_missiles.add(missile_key)
        if self.is_potential:
            self._event_totals[agent_id] = (
                self._event_totals.get(agent_id, 0.0) + reward)
            reward = self._event_totals[agent_id]
        return self._process(reward, agent_id)
