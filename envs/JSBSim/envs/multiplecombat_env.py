import hashlib
import json
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from ..tasks.multiplecombat_task import (
    HierarchicalMultipleCombatShootTask,
    HierarchicalMultipleCombatTask,
    MultipleCombatTask,
)
from .env_base import BaseEnv


class MultipleCombatEnv(BaseEnv):
    """
    MultipleCombatEnv is an multi-player competitive environment.
    """

    def __init__(self, config_name: str, reuse_fdm: bool = False):
        # ``load_task`` sets this protocol flag from the explicit task name.
        # Keep PBGR imports out of the legacy module-import/construction path.
        self._pbgr_enabled = False
        self._pbgr_entity_enabled = False
        self._pbgr_flat_entity_observations = False
        self._pbgr_v22_platform_correct = False
        self._pbgr_v22_platform_correct = False
        self._initial_geometry_protocol = None
        self._pending_initial_geometry = None
        self._initial_geometry_environment_seed = None
        self._initial_geometry_reset_index = 0
        self._initial_geometry_side_policy = "uniform_native_swapped"
        self.last_initial_geometry = None
        self._latest_entity_observations = {}
        self._decision_property_cache = {}
        self._decision_vector_cache = {}
        self._decision_snapshot_token = None
        super().__init__(config_name, reuse_fdm=reuse_fdm)
        self._initialize_initial_geometry_protocol()
        # Env-Specific initialization here!
        self._create_records = False
        if self._pbgr_entity_enabled:
            try:
                self._initialize_pbgr_entity_adapter()
            except Exception:
                # BaseEnv has already created native simulators.  Do not leak
                # them when an invalid PBGR capacity aborts construction.
                self.close()
                raise

    def seed(self, seed=None):
        """Seed the environment and restart its indexed R4 geometry stream."""

        seeds = super().seed(seed)
        self._initial_geometry_environment_seed = int(seeds[0])
        self._initial_geometry_reset_index = 0
        return seeds

    def _initialize_initial_geometry_protocol(self):
        enabled = bool(
            getattr(self.config, "InitialGeometryRandomization_enabled", False)
        )
        if not enabled:
            return
        from evaluation.frozen_geometry import FrozenGeometryProtocol

        raw_path = str(
            getattr(
                self.config,
                "InitialGeometryRandomization_protocol_file",
                "",
            )
        ).strip()
        expected_hash = str(
            getattr(
                self.config,
                "InitialGeometryRandomization_protocol_sha256",
                "",
            )
        ).strip().lower()
        if not raw_path or len(expected_hash) != 64:
            raise ValueError(
                "randomized initial geometry requires a protocol file and SHA256"
            )
        path = Path(raw_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[3] / path
        path = path.resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_hash:
            raise ValueError("initial geometry protocol SHA256 mismatch")
        protocol = FrozenGeometryProtocol.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
        side_policy = str(
            getattr(
                self.config,
                "InitialGeometryRandomization_training_side_policy",
                "uniform_native_swapped",
            )
        ).strip().lower()
        if side_policy not in (
            "uniform_native_swapped",
            "paired_native_swapped",
        ):
            raise ValueError("unsupported initial geometry side policy")
        self._initial_geometry_protocol = protocol
        self._initial_geometry_protocol_path = path
        self._initial_geometry_side_policy = side_policy

    def set_frozen_initial_geometry(self, sample):
        """Queue an explicit fixed-v geometry for the next reset only."""

        if not isinstance(sample, dict) or not isinstance(
            sample.get("aircraft_states"), dict
        ):
            raise ValueError("invalid frozen initial geometry sample")
        if set(sample["aircraft_states"]) != set(self.agents):
            raise ValueError("geometry sample aircraft IDs do not match environment")
        self._pending_initial_geometry = sample

    def get_initial_geometry_state(self):
        """Return the minimal state needed to continue the next R4 reset.

        Legacy scenarios deliberately return ``None`` and retain their fixed
        initial conditions.  A randomized R4 checkpoint stores this payload
        for each vector slot before the process is torn down.
        """

        if self._initial_geometry_protocol is None:
            return None
        return {
            "schema_version": "PBGRInitialGeometryStateV1",
            "protocol_sha256": self._initial_geometry_protocol.protocol_sha256,
            "environment_seed": int(self._initial_geometry_environment_seed),
            "next_reset_index": int(self._initial_geometry_reset_index),
        }

    def set_initial_geometry_state(self, state):
        """Restore the next unused R4 geometry index, failing closed."""

        if self._initial_geometry_protocol is None:
            if state is not None:
                raise ValueError(
                    "fixed-geometry scenarios cannot restore randomized geometry state"
                )
            return
        if not isinstance(state, Mapping):
            raise ValueError("randomized initial geometry state must be a mapping")
        expected = {
            "schema_version",
            "protocol_sha256",
            "environment_seed",
            "next_reset_index",
        }
        if set(state) != expected:
            raise ValueError("initial geometry state fields do not match the schema")
        if state["schema_version"] != "PBGRInitialGeometryStateV1":
            raise ValueError("unsupported initial geometry state schema")
        if (
            str(state["protocol_sha256"]).lower()
            != self._initial_geometry_protocol.protocol_sha256
        ):
            raise ValueError("initial geometry protocol changed across resume")
        environment_seed = state["environment_seed"]
        reset_index = state["next_reset_index"]
        for value, name in (
            (environment_seed, "environment_seed"),
            (reset_index, "next_reset_index"),
        ):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be a non-negative integer")
            if int(value) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if int(environment_seed) != int(self._initial_geometry_environment_seed):
            raise ValueError("environment seed changed across geometry resume")
        self._initial_geometry_reset_index = int(reset_index)

    def set_initial_geometry_reset_index(self, reset_index):
        """Set only the next reset index for a frozen matched-evaluation round."""

        state = self.get_initial_geometry_state()
        if state is None:
            raise RuntimeError(
                "initial geometry reset indices require randomized R4 geometry"
            )
        state["next_reset_index"] = reset_index
        self.set_initial_geometry_state(state)

    def _install_initial_geometry(self, sample, indexed_metadata=None):
        states = sample["aircraft_states"]
        for aircraft_id, state in states.items():
            merged = dict(self.agents[aircraft_id].init_state)
            merged.update({str(key): float(value) for key, value in state.items()})
            self.agents[aircraft_id].init_state = merged
        self.last_initial_geometry = {
            key: sample[key]
            for key in (
                "protocol_id",
                "protocol_sha256",
                "geometry_seed",
                "side_assignment",
                "sample_sha256",
                "engagement_range_m",
                "formation_spacing_m",
                "bearing_deg",
            )
        }
        if indexed_metadata is not None:
            self.last_initial_geometry.update(indexed_metadata)

    def _initialize_pbgr_entity_adapter(self):
        from algorithms.pbgr.entities import JSBSimEntityAdapter

        agents = tuple(self.agents.values())
        default_teammates = max(
            (len(getattr(agent, "partners", ())) for agent in agents),
            default=0,
        )
        default_enemies = max(
            (len(getattr(agent, "enemies", ())) for agent in agents),
            default=0,
        )
        # For each observer, every missile initially carried by its opponents
        # can become a distinct incoming UID.  A shared adapter width must
        # cover every observer's opponent-total requirement; it is not the
        # combined ammunition of both teams.
        opponent_ammunition = {
            str(agent.uid): sum(
                JSBSimEntityAdapter._initial_ammunition(enemy)
                for enemy in getattr(agent, "enemies", ())
            )
            for agent in agents
        }
        default_missiles = max(opponent_ammunition.values(), default=0)
        adapter = JSBSimEntityAdapter(
            max_teammates=getattr(self.config, "pbgr_max_teammates", default_teammates),
            max_enemies=getattr(self.config, "pbgr_max_enemies", default_enemies),
            max_missiles=getattr(
                self.config, "pbgr_max_incoming_missiles", default_missiles
            ),
            visibility_range_m=getattr(self.config, "sensor_range_m", np.inf),
            missile_age_scale_s=getattr(self.config, "missile_age_scale_s", 60.0),
        )
        adapter.validate_environment_capacity(self)
        self.entity_adapter = adapter

    @property
    def share_observation_space(self):
        return self.task.share_observation_space

    def load_task(self):
        taskname = getattr(self.config, "task", None)
        self._pbgr_enabled = False
        self._pbgr_entity_enabled = False
        self._pbgr_flat_entity_observations = False
        if taskname == "multiplecombat":
            self.task = MultipleCombatTask(self.config)
        elif taskname == "hierarchical_multiplecombat":
            self.task = HierarchicalMultipleCombatTask(self.config)
        elif taskname == "hierarchical_multiplecombat_shoot":
            self.task = HierarchicalMultipleCombatShootTask(self.config)
        elif taskname == "pbgr_hierarchical_multiplecombat_shoot":
            from ..tasks.pbgr_multiplecombat_task import (
                PBGRHierarchicalMultipleCombatShootTask,
            )

            self.task = PBGRHierarchicalMultipleCombatShootTask(self.config)
            self._pbgr_enabled = True
            self._pbgr_entity_enabled = True
        elif taskname == "pbgr_bric_hierarchical_multiplecombat_shoot":
            from ..tasks.pbgr_multiplecombat_task import (
                PBGRBRICHierarchicalMultipleCombatShootTask,
            )

            self.task = PBGRBRICHierarchicalMultipleCombatShootTask(self.config)
            self._pbgr_enabled = True
            self._pbgr_entity_enabled = True
        elif taskname == "pbgr_v22_platform_correct_hierarchical_multiplecombat_shoot":
            from ..tasks.pbgr_multiplecombat_task import (
                PBGRV22PlatformCorrectHierarchicalMultipleCombatShootTask,
            )
            self.task = PBGRV22PlatformCorrectHierarchicalMultipleCombatShootTask(self.config)
            self._pbgr_enabled = True
            self._pbgr_entity_enabled = True
            self._pbgr_flat_entity_observations = True
            self._pbgr_v22_platform_correct = True
        elif taskname == "pbgr_multiplecombat":
            from ..tasks.pbgr_multiplecombat_task import PBGRMultipleCombatTask

            self.task = PBGRMultipleCombatTask(self.config)
            self._pbgr_entity_enabled = True
            self._pbgr_flat_entity_observations = True
        else:
            raise NotImplementedError(f"Unknown taskname: {taskname}")

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Resets the state of the environment and returns an initial observation.

        Returns:
            obs (dict): {agent_id: initial observation}
            share_obs (dict): {agent_id: initial state}
        """
        self.current_step = 0
        self.reset_simulators()
        self.task.reset(self)
        self._start_decision_snapshot(0)
        if bool(getattr(self, "_pbgr_entity_enabled", False)) and hasattr(
            self, "entity_adapter"
        ):
            self.entity_adapter.reset()
            self._latest_entity_observations.clear()
        if bool(getattr(self, "_pbgr_flat_entity_observations", False)):
            obs, share_obs = self._get_flat_entity_obs_and_state()
        else:
            obs = self.get_obs()
            share_obs = self.get_state()
        return self._pack(obs), self._pack(share_obs)

    def _start_decision_snapshot(self, decision_id):
        """Open a cache for one decision-boundary state.

        Physics consumers continue to read the simulator's substep-updated
        position/velocity arrays. This cache is only for decision-boundary
        observation, reward and termination property reads.
        """
        self._decision_property_cache.clear()
        self._decision_vector_cache.clear()
        self._decision_snapshot_token = int(decision_id)

    def get_agent_property_values(self, agent_id, props):
        """Read each JSBSim property at most once per decision snapshot."""
        agent_id = str(agent_id)
        if agent_id not in self.agents:
            raise KeyError("unknown agent_id: {0}".format(agent_id))
        if not hasattr(self, "_decision_property_cache"):
            self._decision_property_cache = {}
        cache = self._decision_property_cache.setdefault(agent_id, {})
        missing = [prop for prop in props if prop.name_jsbsim not in cache]
        if missing:
            values = self.agents[agent_id].get_property_values(missing)
            cache.update({
                prop.name_jsbsim: value
                for prop, value in zip(missing, values)
            })
        return [cache[prop.name_jsbsim] for prop in props]

    def get_simulator_vector(self, simulator, method):
        """Return one validated kinematic vector per decision snapshot."""
        if not hasattr(self, "_decision_vector_cache"):
            self._decision_vector_cache = {}
        key = (id(simulator), str(method))
        value = self._decision_vector_cache.get(key)
        if value is None:
            value = np.asarray(
                getattr(simulator, method)(), dtype=np.float64
            )
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(
                    "{0} must return a finite 3-vector".format(method)
                )
            # Simulator arrays mutate in place on the next substep. Copy so
            # one decision snapshot cannot change underneath its consumers.
            value = value.copy()
            self._decision_vector_cache[key] = value
        return value

    def _get_flat_entity_obs_and_state(self):
        """Build every local view exactly once so slot_reset stays coherent."""
        obs = {
            agent_id: self.task.get_obs(self, agent_id)
            for agent_id in self.agents.keys()
        }
        state = np.hstack([obs[agent_id] for agent_id in self.agents.keys()])
        share_obs = {agent_id: state.copy() for agent_id in self.agents.keys()}
        return obs, share_obs

    def get_entity_observation(self, agent_id):
        """Return the mask-explicit PBGR entity view for one aircraft."""
        if not bool(getattr(self, "_pbgr_entity_enabled", False)) or not hasattr(
            self, "entity_adapter"
        ):
            raise RuntimeError("entity observations require the PBGR opt-in task")
        observation = self.entity_adapter.build(self, agent_id)
        self._latest_entity_observations[str(agent_id)] = observation
        return observation

    def get_cached_entity_observation(self, agent_id):
        key = str(agent_id)
        if key not in self._latest_entity_observations:
            return self.get_entity_observation(agent_id)
        return self._latest_entity_observations[key]

    def get_entity_observations(self):
        return {
            agent_id: self.get_entity_observation(agent_id)
            for agent_id in self.agents.keys()
        }

    def get_weapon_action_state(self, agent_id):
        """Return a copy of one agent's state before an action is applied."""
        if not bool(getattr(self, "_pbgr_enabled", False)):
            raise RuntimeError("weapon action state requires the PBGR opt-in task")
        getter = getattr(self.task, "get_weapon_action_state", None)
        if getter is None:
            raise RuntimeError("the active task does not expose a weapon action state")
        return dict(getter(self, agent_id))

    def get_weapon_action_mask(self, agent_id):
        """Return the per-agent ``[hold, fire]`` pre-action legality mask."""
        if not bool(getattr(self, "_pbgr_enabled", False)):
            raise RuntimeError("weapon action mask requires the PBGR opt-in task")
        getter = getattr(self.task, "get_weapon_action_mask", None)
        if getter is None:
            raise RuntimeError("the active task does not expose a weapon action mask")
        return np.asarray(getter(self, agent_id), dtype=np.bool_).copy()

    def get_weapon_action_masks(self):
        """Return pre-action masks for every agent in deterministic env order."""
        return {
            agent_id: self.get_weapon_action_mask(agent_id)
            for agent_id in self.agents.keys()
        }

    def get_bric_weapon_action_state(self, agent_id):
        """Return explicit target slots and fire masks for the V20 task."""
        getter = getattr(self.task, "get_bric_weapon_action_state", None)
        if getter is None:
            raise RuntimeError("the active task is not the BRIC weapon task")
        return dict(getter(self, agent_id))

    def get_bric_weapon_action_masks(self):
        """Return ``[agent,target,hold/fire]`` masks in environment order."""
        rows = []
        for agent_id in self.agents.keys():
            state = self.get_bric_weapon_action_state(agent_id)
            masks = np.asarray(state["fire_masks"], dtype=np.bool_)
            if masks.ndim != 2 or masks.shape[1] != 2:
                raise RuntimeError("BRIC per-target fire masks are malformed")
            rows.append(masks)
        result = np.stack(rows, axis=0)
        if not np.all(result[..., 0]):
            raise RuntimeError("BRIC fire masks must always permit hold")
        return result

    def get_v22_weapon_action_masks(self):
        getter = getattr(self.task, "get_v22_weapon_action_state", None)
        if getter is None:
            raise RuntimeError("the active task is not the V22 weapon task")
        target_rows, fire_rows = [], []
        for agent_id in self.agents:
            state = getter(self, agent_id)
            target_rows.append(np.asarray(state["target_mask"], dtype=np.bool_))
            fire_rows.append(np.asarray(state["fire_masks"], dtype=np.bool_))
        targets = np.stack(target_rows, axis=0)
        fires = np.stack(fire_rows, axis=0)
        if fires.ndim != 3 or fires.shape[-1] != 2 or targets.shape != fires.shape[:-1]:
            raise RuntimeError("V22 target/fire support is malformed")
        if not np.all(fires[..., 0]):
            raise RuntimeError("V22 hold action must always be legal")
        return {"target_masks": targets, "fire_masks": fires}

    def get_v22_platform_contract(self):
        if not bool(getattr(self, "_pbgr_v22_platform_correct", False)):
            raise RuntimeError("the active environment is not the V22 platform task")
        return {
            "potential_reward_gamma": float(
                getattr(self.config, "PotentialReward_gamma")
            ),
            "formal_reward_contract": bool(
                getattr(self.config, "FormalRewardContract_enabled", False)
            ),
            "high_level_commitment_decisions": int(
                getattr(self.config, "high_level_commitment_decisions")
            ),
        }

    def reset_simulators(self):
        sample = self._pending_initial_geometry
        self._pending_initial_geometry = None
        indexed_metadata = None
        if sample is None and self._initial_geometry_protocol is not None:
            # Every vector slot owns an indexed geometry stream.  Checkpoints
            # persist ``next_reset_index`` so process recreation cannot replay
            # the beginning of a mutable RNG stream.
            from evaluation.frozen_geometry import indexed_geometry_assignment

            reset_index = int(self._initial_geometry_reset_index)
            geometry_seed, side_assignment = indexed_geometry_assignment(
                int(self._initial_geometry_environment_seed),
                reset_index,
                self._initial_geometry_side_policy,
            )
            sample = self._initial_geometry_protocol.sample(
                geometry_seed,
                side_assignment,
                self.ego_ids,
                self.enm_ids,
            )
            self._initial_geometry_reset_index = reset_index + 1
            indexed_metadata = {
                "environment_seed": int(self._initial_geometry_environment_seed),
                "reset_index": reset_index,
            }
        if sample is not None:
            self._install_initial_geometry(sample, indexed_metadata=indexed_metadata)
        else:
            self.last_initial_geometry = None
        for sim in self._jsbsims.values():
            sim.reload()
        self._tempsims.clear()

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's observation. Accepts an action and
        returns a tuple (observation, reward_visualize, done, info).

        Args:
            action (dict): the agents' actions, each key corresponds to an agent_id

        Returns:
            (tuple):
                obs: agents' observation of the current environment
                share_obs: agents' share observation of the current environment
                rewards: amount of rewards returned after previous actions
                dones: whether the episode has ended, in which case further
                    step() calls are undefined
                info: auxiliary information
        """
        self.current_step += 1
        info = {"current_step": self.current_step}
        weapon_task = bool(getattr(self, "_pbgr_enabled", False))
        formal_reward_contract = bool(
            getattr(
                getattr(self, "config", None),
                "FormalRewardContract_enabled",
                False,
            )
        )
        pbgr_protocol = bool(
            weapon_task
            or getattr(self, "_pbgr_flat_entity_observations", False)
            or formal_reward_contract
        )

        # apply actions
        action = self._unpack(action)
        normalized_actions = self.task.normalize_actions(self, action)
        for agent_id in self.agents.keys():
            a_action = normalized_actions[agent_id]
            self.agents[agent_id].set_property_values(self.task.action_var, a_action)
        prepare_weapon_actions = getattr(
            self.task, "prepare_weapon_actions", None)
        if callable(prepare_weapon_actions):
            # The actor observed state ``current_step - 1``. Freeze and apply
            # weapon legality before any dynamics substep changes geometry.
            prepare_weapon_actions(
                self, decision_step=self.current_step - 1)
        # run simulation
        for _ in range(self.agent_interaction_steps):
            self._run_simulation_substep()
        self.task.step(self)
        self._start_decision_snapshot(self.current_step)
        if weapon_task:
            info.update(self.task.get_weapon_step_info())
        if bool(getattr(self, "_pbgr_flat_entity_observations", False)):
            obs, share_obs = self._get_flat_entity_obs_and_state()
        else:
            obs = self.get_obs()
            share_obs = self.get_state()
        # Weapon and formal-comparison protocols resolve state-mutating crash
        # conditions before reward settlement. Historical configurations keep
        # their original reward-then-termination order.
        dones = None
        if weapon_task or formal_reward_contract:
            dones = {}
            for agent_id in self.agents.keys():
                done, info = self.task.get_termination(self, agent_id, info)
                dones[agent_id] = [done]

        rewards = {}
        for agent_id in self.agents.keys():
            reward, info = self.task.get_reward(self, agent_id, info)
            rewards[agent_id] = [reward]
        ego_reward = np.mean([rewards[ego_id] for ego_id in self.ego_ids])
        enm_reward = np.mean([rewards[enm_id] for enm_id in self.enm_ids])
        if formal_reward_contract:
            raw_team_rewards = [float(ego_reward), float(enm_reward)]
            # Orthogonal projection of the two raw team means onto the
            # zero-sum subspace. This removes common-mode shaping reward and
            # keeps its scale symmetric for the two sides.
            competitive_reward = 0.5 * float(ego_reward - enm_reward)
            ego_reward = competitive_reward
            enm_reward = -competitive_reward
            for ego_id in self.ego_ids:
                rewards[ego_id] = [ego_reward]
            for enm_id in self.enm_ids:
                rewards[enm_id] = [enm_reward]
            info["formal_reward_contract"] = {
                "version": str(getattr(
                    self.config, "FormalRewardContract_version", "unspecified"
                )),
                "zero_sum_shaping": True,
                "posture_enemy_aggregation": str(getattr(
                    self.config,
                    "PostureReward_enemy_aggregation",
                    "sum",
                )),
            }
            info["raw_team_shaping_reward"] = raw_team_rewards
            info["team_shaping_reward"] = [ego_reward, enm_reward]
        elif weapon_task:
            # Preserve a first one-shot terminal event for the PBGR team while
            # keeping an already-settled dead aircraft inert thereafter.
            def receives_team_reward(agent_id):
                if self.agents[agent_id].is_alive:
                    return True
                return bool(self.task._reward_transition_active.get(agent_id, False))

            for ego_id in self.ego_ids:
                rewards[ego_id] = [ego_reward if receives_team_reward(ego_id) else 0.0]
            for enm_id in self.enm_ids:
                rewards[enm_id] = [enm_reward if receives_team_reward(enm_id) else 0.0]
        else:
            # Exact historical team averaging, including its dead-agent
            # assignment semantics, is retained for baseline reproduction.
            for ego_id in self.ego_ids:
                rewards[ego_id] = [ego_reward]
            for enm_id in self.enm_ids:
                rewards[enm_id] = [enm_reward]

        if dones is None:
            dones = {}
            for agent_id in self.agents.keys():
                done, info = self.task.get_termination(self, agent_id, info)
                dones[agent_id] = [done]

        if pbgr_protocol:
            from ..tasks.weapon_contract import (
                resolve_team_outcome, team_outcome_terminal_rewards,
            )

            agent_local_terminated = {
                agent_id: bool(dones[agent_id][0]) for agent_id in self.agents.keys()
            }
            outcome = resolve_team_outcome(
                self, timeout=self.current_step >= self.max_steps
            )
            if outcome.terminated:
                for agent_id in dones:
                    dones[agent_id] = [True]
            # ``agent_terminated`` is the exact terminal mask returned from
            # this step. ``agent_local_terminated`` retains diagnostic causes
            # that occurred before team-level episode settlement. PBGR_R1 has
            # no environment-side truncation; an infrastructure cutoff must be
            # injected explicitly by the collector.
            info["agent_local_terminated"] = agent_local_terminated
            info["agent_terminated"] = {
                agent_id: bool(dones[agent_id][0]) for agent_id in self.agents.keys()
            }
            info["agent_truncated"] = {
                agent_id: False for agent_id in self.agents.keys()
            }
            info["episode_terminated"] = bool(outcome.terminated)
            info["episode_truncated"] = bool(outcome.truncated)
            info["team_outcome"] = outcome.to_dict()
            info["evaluation_utility"] = [outcome.ego_result, outcome.enemy_result]
            info["terminated"] = bool(outcome.terminated)
            info["truncated"] = bool(outcome.truncated)
            info["bootstrap_terminal"] = bool(outcome.terminated)
            info["termination_reason"] = (
                "timeout_draw" if outcome.reason == "timeout" else outcome.reason
            )
            if formal_reward_contract:
                terminal_rewards = team_outcome_terminal_rewards(
                    self.config, outcome)
                if outcome.terminated:
                    for ego_id in self.ego_ids:
                        rewards[ego_id][0] += terminal_rewards[0]
                    for enm_id in self.enm_ids:
                        rewards[enm_id][0] += terminal_rewards[1]
                info["team_terminal_reward"] = terminal_rewards

        return (
            self._pack(obs),
            self._pack(share_obs),
            self._pack(rewards),
            self._pack(dones),
            info,
        )
