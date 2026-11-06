import logging
import time
import gymnasium
from gymnasium.utils import seeding
import numpy as np
from typing import Dict, Any, Tuple
from ..core.simulatior import AircraftSimulator, BaseSimulator, MissileSimulator
from ..tasks.task_base import BaseTask
from ..utils.utils import parse_config


class NonFinitePackedDataError(FloatingPointError):
    """Raised when an environment output contains NaN or infinity."""


class BaseEnv(gymnasium.Env):
    """
    A class wrapping the JSBSim flight dynamics module (FDM) for simulating
    aircraft as an RL environment conforming to the OpenAI Gym Env
    interface.

    An BaseEnv is instantiated with a Task that implements a specific
    aircraft control task with its own specific observation/action space and
    variables and agent_reward calculation.
    """
    metadata = {"render.modes": ["human", "txt"]}

    def __init__(self, config_name: str, reuse_fdm: bool = False,
                 config_overrides=None):
        # basic args
        self.config = parse_config(config_name)
        for key, value in dict(config_overrides or {}).items():
            setattr(self.config, str(key), value)
        self.max_steps = getattr(self.config, 'max_steps', 100)  # type: int
        self.sim_freq = getattr(self.config, 'sim_freq', 60)  # type: int
        self.agent_interaction_steps = getattr(self.config, 'agent_interaction_steps', 12)  # type: int
        self.center_lon, self.center_lat, self.center_alt = \
            getattr(self.config, 'battle_field_center', (120.0, 60.0, 0.0))
        self.reuse_fdm = bool(reuse_fdm)
        self._create_records = False
        self.load()

    @property
    def num_agents(self) -> int:
        return self.task.num_agents

    @property
    def observation_space(self) -> gymnasium.Space:
        return self.task.observation_space

    @property
    def action_space(self) -> gymnasium.Space:
        return self.task.action_space

    @property
    def agents(self) -> Dict[str, AircraftSimulator]:
        return self._jsbsims

    @property
    def time_interval(self) -> int:
        return self.agent_interaction_steps / self.sim_freq

    def load(self):
        self.load_task()
        self.load_simulator()
        self.seed()

    def load_task(self):
        self.task = BaseTask(self.config)

    def load_simulator(self):
        self._jsbsims = {}     # type: Dict[str, AircraftSimulator]
        for uid, config in self.config.aircraft_configs.items():
            self._jsbsims[uid] = AircraftSimulator(
                uid=uid,
                color=config.get("color", "Red"),
                model=config.get("model", "f16"),
                init_state=config.get("init_state"),
                origin=getattr(self.config, 'battle_field_center', (120.0, 60.0, 0.0)),
                sim_freq=self.sim_freq,
                reuse_fdm=self.reuse_fdm,
                num_missiles=config.get("missile", 0))
        # Different teams have different uid[0]
        _default_team_uid = list(self._jsbsims.keys())[0][0]
        self.ego_ids = [uid for uid in self._jsbsims.keys() if uid[0] == _default_team_uid]
        self.enm_ids = [uid for uid in self._jsbsims.keys() if uid[0] != _default_team_uid]

        # Link jsbsims
        for key, sim in self._jsbsims.items():
            for k, s in self._jsbsims.items():
                if k == key:
                    pass
                elif k[0] == key[0]:
                    sim.partners.append(s)
                else:
                    sim.enemies.append(s)

        self._tempsims = {}    # type: Dict[str, BaseSimulator]

    def add_temp_simulator(self, sim: BaseSimulator):
        self._tempsims[sim.uid] = sim

    def _run_simulation_substep(self):
        """Advance aircraft and missiles with synchronized target effects."""

        # Most decision steps have no missile in flight.  Avoid two native
        # JSBSim property reads, NumPy allocations, and an empty resolver call
        # on every 60 Hz substep in that common case.
        if not self._tempsims:
            for sim in self._jsbsims.values():
                sim.run()
            return

        temporary_simulators = tuple(self._tempsims.values())
        deferred_missiles = [
            sim for sim in temporary_simulators
            if isinstance(sim, MissileSimulator)
        ]
        target_aircraft = {
            id(missile.target_aircraft): missile.target_aircraft
            for missile in deferred_missiles
        }
        target_segment_starts = {
            target_id: np.asarray(
                aircraft.get_position(), dtype=np.float64).copy()
            for target_id, aircraft in target_aircraft.items()
        }
        for sim in self._jsbsims.values():
            sim.run()
        for sim in temporary_simulators:
            if isinstance(sim, MissileSimulator):
                sim.run(
                    defer_target_effect=True,
                    target_segment_start=target_segment_starts.get(
                        id(sim.target_aircraft)),
                )
            else:
                sim.run()
        if deferred_missiles:
            MissileSimulator.resolve_deferred_target_effects(
                deferred_missiles)

    def reset(self) -> np.ndarray:
        """Resets the state of the environment and returns an initial observation.

        Returns:
            obs (np.ndarray): initial observation
        """
        # reset sim
        self.current_step = 0
        for sim in self._jsbsims.values():
            sim.reload()
        self._tempsims.clear()
        # reset task
        self.task.reset(self)
        obs = self.get_obs()
        return self._pack(obs)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's observation. Accepts an action and
        returns a tuple (observation, reward_visualize, done, info).

        Args:
            action (np.ndarray): the agents' actions, allow opponent's action input

        Returns:
            (tuple):
                obs: agents' observation of the current environment
                rewards: amount of rewards returned after previous actions
                dones: whether the episode has ended, in which case further step() calls are undefined
                info: auxiliary information
        """
        self.current_step += 1
        info = {"current_step": self.current_step}
        # apply actions
        action = self._unpack(action)
        normalize_actions = getattr(self.task, "normalize_actions", None)
        if callable(normalize_actions):
            normalized_actions = normalize_actions(self, action)
        else:
            normalized_actions = {
                agent_id: self.task.normalize_action(
                    self, agent_id, action[agent_id])
                for agent_id in self.agents.keys()
            }
        for agent_id in self.agents.keys():
            a_action = normalized_actions[agent_id]
            self.agents[agent_id].set_property_values(self.task.action_var, a_action)
        prepare_weapon_actions = getattr(
            self.task, "prepare_weapon_actions", None)
        if callable(prepare_weapon_actions):
            prepare_weapon_actions(
                self, decision_step=self.current_step - 1)
        # run simulation
        for _ in range(self.agent_interaction_steps):
            self._run_simulation_substep()
        self.task.step(self)
        weapon_info = getattr(self.task, "get_weapon_step_info", None)
        if callable(weapon_info):
            info.update(weapon_info())

        obs = self.get_obs()

        dones = {}
        for agent_id in self.agents.keys():
            done, info = self.task.get_termination(self, agent_id, info)
            dones[agent_id] = [done]
        if callable(weapon_info):
            # Some safety termination conditions mutate aircraft status, so
            # lifecycle telemetry must be captured after all checks run.
            info["aircraft_status"] = {
                str(agent_id): {
                    "alive": bool(agent.is_alive),
                    "crash": bool(agent.is_crash),
                    "shotdown": bool(agent.is_shotdown),
                }
                for agent_id, agent in self.agents.items()
            }

        rewards = {}
        for agent_id in self.agents.keys():
            reward, info = self.task.get_reward(self, agent_id, info)
            rewards[agent_id] = [reward]

        if bool(getattr(
                self.config, "FormalRewardContract_enabled", False)):
            ego_raw = float(np.mean([
                rewards[agent_id][0] for agent_id in self.ego_ids
            ]))
            enemy_raw = float(np.mean([
                rewards[agent_id][0] for agent_id in self.enm_ids
            ]))
            competitive = 0.5 * (ego_raw - enemy_raw)
            for agent_id in self.ego_ids:
                rewards[agent_id] = [competitive]
            for agent_id in self.enm_ids:
                rewards[agent_id] = [-competitive]

            from ..tasks.weapon_contract import (
                resolve_team_outcome, team_outcome_terminal_rewards,
            )

            outcome = resolve_team_outcome(
                self, timeout=self.current_step >= self.max_steps)
            terminal_rewards = team_outcome_terminal_rewards(
                self.config, outcome)
            if outcome.terminated:
                for agent_id in self.ego_ids:
                    rewards[agent_id][0] += terminal_rewards[0]
                    dones[agent_id] = [True]
                for agent_id in self.enm_ids:
                    rewards[agent_id][0] += terminal_rewards[1]
                    dones[agent_id] = [True]
            info["formal_reward_contract"] = {
                "version": str(getattr(
                    self.config,
                    "FormalRewardContract_version",
                    "unified-weapon-reward-v1",
                )),
                "zero_sum_shaping": True,
            }
            info["raw_team_shaping_reward"] = [ego_raw, enemy_raw]
            info["team_shaping_reward"] = [competitive, -competitive]
            raw_components = info.get("reward_components", {})
            component_names = sorted({
                name
                for values in raw_components.values()
                for name in values
            })
            team_components = {}
            for name in component_names:
                ego_component = float(np.mean([
                    raw_components.get(agent_id, {}).get(name, 0.0)
                    for agent_id in self.ego_ids
                ]))
                enemy_component = float(np.mean([
                    raw_components.get(agent_id, {}).get(name, 0.0)
                    for agent_id in self.enm_ids
                ]))
                zero_sum_component = 0.5 * (
                    ego_component - enemy_component)
                team_components[name] = [
                    zero_sum_component, -zero_sum_component]
            info["team_terminal_reward"] = terminal_rewards
            info["team_evaluation_utility"] = [
                outcome.ego_result, outcome.enemy_result]
            terminal_component = (
                info["team_terminal_reward"]
                if outcome.terminated else [0.0, 0.0]
            )
            team_components["TeamOutcomeReward"] = terminal_component
            info["team_reward_components"] = team_components
            info["team_outcome"] = outcome.to_dict()
            info["episode_terminated"] = bool(outcome.terminated)
            info["episode_truncated"] = bool(outcome.truncated)
            info["bootstrap_terminal"] = bool(outcome.terminated)

        return self._pack(obs), self._pack(rewards), self._pack(dones), info

    def get_obs(self):
        """Returns all agent observations in a list.

        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        observations = dict([
            (agent_id, self.task.get_obs(self, agent_id))
            for agent_id in self.agents.keys()
        ])
        # The next action is, by definition, conditioned on these returned
        # observations.  The fixed low-level controller only consumes fields
        # from the same decision observation, so retain them instead of
        # rebuilding identical geometry before the next dynamics advance.
        self._decision_observations = observations
        return observations

    def get_state(self):
        """Returns the global state.

        NOTE: This functon should not be used during decentralised execution.
        """
        state = np.hstack([self.task.get_obs(self, agent_id) for agent_id in self.agents.keys()])
        return dict([(agent_id, state.copy()) for agent_id in self.agents.keys()])

    def close(self):
        """Cleans up this environment's objects

        NOTE: Environments automatically close() when garbage collected or when the
        program exits.
        """
        for sim in self._jsbsims.values():
            sim.close()
        for sim in self._tempsims.values():
            sim.close()
        self._jsbsims.clear()
        self._tempsims.clear()

    def render(self, mode="txt", filepath='./JSBSimRecording.txt.acmi', tacview=None):
        """Renders the environment.

        The set of supported modes varies per environment. (And some

        environments do not support rendering at all.) By convention,

        if mode is:

        - human: print on the terminal
        - txt: output to txt.acmi files
        - real_time: realtime render with tacview by socket comm

        Note:

            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        :param mode: str, the mode to render with
        """
        if mode == "txt":
            if not self._create_records:
                with open(filepath, mode='w', encoding='utf-8-sig') as f:
                    f.write("FileType=text/acmi/tacview\n")
                    f.write("FileVersion=2.1\n")
                    f.write("0,ReferenceTime=2020-04-01T00:00:00Z\n")
                self._create_records = True
            with open(filepath, mode='a', encoding='utf-8-sig') as f:
                timestamp = self.current_step * self.time_interval
                f.write(f"#{timestamp:.2f}\n")
                for sim in self._jsbsims.values():
                    log_msg = sim.log()
                    if log_msg is not None:
                        f.write(log_msg + "\n")
                for sim in self._tempsims.values():
                    log_msg = sim.log()
                    if log_msg is not None:
                        f.write(log_msg + "\n")
        elif mode == "real_time":
            timestamp = self.current_step * self.time_interval
            data = [f"#{timestamp:.2f}\n"]
            for sim in self._jsbsims.values():
                log_msg = sim.log()
                if log_msg is not None:
                    data.append(log_msg + "\n")
        
            for sim in self._tempsims.values():
                log_msg = sim.log()
                if log_msg is not None:
                    data.append(log_msg + "\n")

            data_str = "".join(data)
            # send data to tacview
            tacview.send_data_to_client(data_str)
        else:
            raise NotImplementedError

    def seed(self, seed=None):
        """
        Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _pack(self, data: Dict[str, Any]) -> np.ndarray:
        """Pack seperated key-value dict into grouped np.ndarray"""
        ego_data = np.array([data[uid] for uid in self.ego_ids])
        enm_data = np.array([data[uid] for uid in self.enm_ids])
        if enm_data.shape[0] > 0:
            data = np.concatenate((ego_data, enm_data))  # type: np.ndarray
        else:
            data = ego_data  # type: np.ndarray
        invalid_indices = np.argwhere(~np.isfinite(data))
        if invalid_indices.size:
            preview = invalid_indices[:8].tolist()
            suffix = "..." if len(invalid_indices) > len(preview) else ""
            raise NonFinitePackedDataError(
                "BaseEnv._pack received "
                f"{len(invalid_indices)} non-finite value(s); "
                f"indices={preview}{suffix}"
            )
        # only return data that belongs to RL agents
        return data[:self.num_agents, ...]

    def _unpack(self, data: np.ndarray) -> Dict[str, Any]:
        """Unpack grouped np.ndarray into seperated key-value dict"""
        assert isinstance(data, (np.ndarray, list, tuple)) and len(data) == self.num_agents
        # unpack data in the same order to packing process
        unpack_data = dict(zip((self.ego_ids + self.enm_ids)[:self.num_agents], data))
        # fill in None for other not-RL agents
        for agent_id in (self.ego_ids + self.enm_ids)[self.num_agents:]:
            unpack_data[agent_id] = None
        return unpack_data
