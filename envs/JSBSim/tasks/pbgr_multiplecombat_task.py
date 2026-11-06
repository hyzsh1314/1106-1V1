"""PBGR-only extensions for the multi-aircraft simple-weapon task.

The historical task remains untouched.  A configuration must explicitly use
``pbgr_hierarchical_multiplecombat_shoot`` to opt into these actor-facing
resource observations and pre-action masks.
"""

import numpy as np
from typing import Tuple

from ..core.pbgr_missile_simulator import PBGRMissileSimulator
from ..reward_functions import AltitudeReward, PostureReward
from ..reward_functions.pbgr_reward_functions import (
    PBGRDiscountedPostureReward,
    PBGREventDrivenReward,
    PBGRMissilePostureReward,
)
from .multiplecombat_task import (
    HierarchicalMultipleCombatTask,
    HierarchicalMultipleCombatShootTask,
    MultipleCombatTask,
)
from .singlecombat_task import SingleCombatTask
from .weapon_contract import (
    WeaponEvent,
    evaluate_shot,
    explicit_enemy_slots,
)


class PBGRMultipleCombatTask(MultipleCombatTask):
    """No-weapon 2v2 task exposing only mask-explicit entity observations."""

    def load_observation_space(self):
        from algorithms.pbgr.entities import ENTITY_FEATURE_NAMES, FlatEntityCodec
        from gymnasium import spaces

        teammates = int(getattr(self.config, "pbgr_max_teammates", 1))
        enemies = int(getattr(self.config, "pbgr_max_enemies", 2))
        missiles = int(getattr(self.config, "pbgr_max_incoming_missiles", 0))
        self.entity_codec = FlatEntityCodec(
            1 + teammates + enemies + missiles,
            len(ENTITY_FEATURE_NAMES),
        )
        self.obs_length = self.entity_codec.flat_dim
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.obs_length,),
            dtype=np.float32,
        )
        self.share_observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.num_agents * self.obs_length,),
            dtype=np.float32,
        )

    def get_obs(self, env, agent_id):
        return self.entity_codec.pack(env.get_entity_observation(agent_id))


class PBGRHierarchicalMultipleCombatShootTask(
        HierarchicalMultipleCombatShootTask):
    """Opt-in PBGR simple-weapon protocol and actor-facing state contract."""

    def __init__(self, config: str):
        super().__init__(config)
        self.missile_model = getattr(
            self.config, "missile_model", "simple_kinematic_PN_R1")
        if bool(getattr(
                self.config, "UnifiedWeaponReward_enabled", False)):
            self.reward_functions = [
                PBGRDiscountedPostureReward(self.config),
                AltitudeReward(self.config),
            ]
        else:
            self.reward_functions = [
                PostureReward(self.config),
                PBGRMissilePostureReward(self.config),
                AltitudeReward(self.config),
                PBGREventDrivenReward(self.config),
            ]

    def get_reward(self, env, agent_id, info: dict = ...) -> Tuple[float, dict]:
        """Settle a death once without evaluating invalid shaping terms."""
        if not hasattr(self, "_reward_transition_active"):
            self._reward_transition_active = {}
        agent = env.agents[agent_id]
        was_dead = bool(self._agent_die_flag.get(agent_id, False))
        first_death = (not agent.is_alive) and (not was_dead)
        self._reward_transition_active[agent_id] = bool(
            agent.is_alive or first_death)
        if agent.is_alive:
            return SingleCombatTask.get_reward(
                self, env, agent_id, info=info)
        self._agent_die_flag[agent_id] = True
        if not first_death:
            return 0.0, info
        # Geometry and altitude shaping are undefined after death.  Only the
        # auditable terminal event may contribute on this transition.
        reward = 0.0
        for reward_function in self.reward_functions:
            settle = getattr(reward_function, "settle_terminal", None)
            if callable(settle):
                reward += settle(agent_id)
            elif isinstance(reward_function, PBGREventDrivenReward):
                reward += reward_function.get_reward(self, env, agent_id)
        return reward, info

    def reset(self, env):
        result = super().reset(env)
        for agent_id, remaining in self._remaining_missiles.items():
            env.agents[agent_id].num_left_missiles = remaining
        self._weapon_events = []
        self._step_weapon_events = []
        self._weapon_legality = {}
        self._known_missile_outcomes = set()
        self._reward_transition_active = {}
        self._pending_weapon_decisions = None
        self._weapon_decision_step = None
        self._weapon_transition_step = None
        return result

    def prepare_weapon_actions(self, env, decision_step):
        """Evaluate and apply fire requests at the actor-observation instant.

        ``MultipleCombatEnv`` increments ``current_step`` before applying an
        action, so it passes the preceding state index explicitly.  Geometry,
        ammunition and cooldown are therefore identical to the values used by
        :meth:`get_weapon_action_mask`; a target cannot cross the range/cone
        boundary during the dynamics substeps and retroactively change whether
        the submitted request was accepted.
        """
        self._step_weapon_events = []
        self._weapon_decision_step = int(decision_step)
        self._weapon_transition_step = int(env.current_step)
        self._pending_weapon_decisions = {}
        for agent_id, agent in env.agents.items():
            target_id = self._requested_weapon_target_id(agent_id)
            legality, target = evaluate_shot(
                agent=agent,
                requested=self._shoot_action[agent_id],
                remaining_missiles=self._remaining_missiles[agent_id],
                current_step=decision_step,
                last_shot_step=self._last_shoot_time[agent_id],
                max_attack_angle=self.max_attack_angle,
                max_attack_distance=self.max_attack_distance,
                min_attack_interval=self.min_attack_interval,
                target_id=target_id,
            )
            self._weapon_legality[agent_id] = legality
            self._pending_weapon_decisions[agent_id] = (legality, target)
            if legality.accepted:
                new_missile_uid = (
                    agent_id + str(self._remaining_missiles[agent_id])
                )
                missile = PBGRMissileSimulator.create(
                    parent=agent,
                    target=target,
                    uid=new_missile_uid,
                    missile_model=self.missile_model,
                )
                env.add_temp_simulator(missile)
                self._remaining_missiles[agent_id] -= 1
                agent.num_left_missiles = self._remaining_missiles[agent_id]
                self._last_shoot_time[agent_id] = int(decision_step)
                self._append_event(WeaponEvent(
                    "launch", env.current_step, agent_id, target.uid,
                    new_missile_uid, "accepted"))
            elif legality.requested:
                self._append_event(WeaponEvent(
                    "rejected", env.current_step, agent_id,
                    legality.target_id, None, legality.reason))

    def _requested_weapon_target_id(self, agent_id):
        """Return ``None`` for the historical nearest-enemy contract."""
        return None

    def step(self, env):
        """Advance gun logic and record outcomes of the prepared transition."""
        if self._pending_weapon_decisions is None:
            raise RuntimeError(
                "prepare_weapon_actions must run before each PBGR task step")
        SingleCombatTask.step(self, env)
        self._record_resolved_missiles(env)
        self._pending_weapon_decisions = None

    def _append_event(self, event):
        self._step_weapon_events.append(event)
        self._weapon_events.append(event)

    def _record_resolved_missiles(self, env):
        for agent_id, agent in env.agents.items():
            for missile in agent.launch_missiles:
                if (not missile.is_done or
                        missile.uid in self._known_missile_outcomes):
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
                self._append_event(WeaponEvent(
                    event_type,
                    env.current_step,
                    agent_id,
                    None if target is None else target.uid,
                    missile.uid,
                    reason,
                ))
                self._known_missile_outcomes.add(missile.uid)

    def get_weapon_step_info(self):
        event_types = (
            "launch", "rejected", "hit", "redundant_hit", "miss",
            "invalidated",
        )
        counts = {
            event_type: sum(
                event.event_type == event_type for event in self._weapon_events
            )
            for event_type in event_types
        }
        return {
            "weapon_schema_version": "simple-missile-v1",
            "weapon_decision_step": self._weapon_decision_step,
            "weapon_transition_step": self._weapon_transition_step,
            "weapon_events": [
                event.to_dict() for event in self._step_weapon_events
            ],
            "weapon_event_count": len(self._weapon_events),
            "weapon_event_counts": counts,
            "weapon_launch_count": counts["launch"],
            "weapon_rejected_count": counts["rejected"],
            "weapon_hit_count": counts["hit"],
            "weapon_redundant_hit_count": counts["redundant_hit"],
            "weapon_kill_count": counts["hit"],
            "weapon_geometric_hit_count": (
                counts["hit"] + counts["redundant_hit"]
            ),
            "weapon_miss_count": counts["miss"],
            "weapon_invalidated_count": counts["invalidated"],
            "weapon_legality": {
                agent_id: legality.to_dict()
                for agent_id, legality in self._weapon_legality.items()
            },
        }

    def get_weapon_action_state(self, env, agent_id):
        """Return the pre-action weapon state without mutating the task.

        The returned mask is computed by the same ``evaluate_shot`` function
        used when the action is executed.  This keeps data collection and
        action sampling on one legality contract instead of inferring
        legality from the previous step's event receipt.
        """
        if agent_id not in env.agents:
            raise KeyError("unknown agent_id: {0}".format(agent_id))
        if (not hasattr(self, '_remaining_missiles') or
                not hasattr(self, '_last_shoot_time')):
            raise RuntimeError("weapon task must be reset before querying masks")
        if (agent_id not in self._remaining_missiles or
                agent_id not in self._last_shoot_time):
            raise KeyError("weapon state missing for agent_id: {0}".format(agent_id))

        agent = env.agents[agent_id]
        remaining = int(self._remaining_missiles[agent_id])
        target_id = self._requested_weapon_target_id(agent_id)
        legality, _ = evaluate_shot(
            agent=agent, requested=False,
            remaining_missiles=remaining,
            current_step=env.current_step,
            last_shot_step=self._last_shoot_time[agent_id],
            max_attack_angle=self.max_attack_angle,
            max_attack_distance=self.max_attack_distance,
            min_attack_interval=self.min_attack_interval,
            target_id=target_id)
        capacity = max(int(getattr(agent, 'num_missiles', 0)), 0)
        ammo_fraction = (float(remaining) / float(capacity)
                         if capacity > 0 else 0.0)
        interval = int(self.min_attack_interval)
        cooldown_fraction = (1.0 if interval <= 0 else
                             float(np.clip(
                                 float(legality.cooldown_steps) / interval,
                                 0.0, 1.0)))
        state = legality.to_dict()
        state.update({
            "weapon_schema_version": "simple-missile-v1",
            "ammo_capacity": capacity,
            "ammo_fraction": float(np.clip(ammo_fraction, 0.0, 1.0)),
            # Zero means just fired; one means the interval is fully served.
            "cooldown_fraction": cooldown_fraction,
            "shoot_legal": bool(legality.can_fire),
        })
        return state

    def get_weapon_action_mask(self, env, agent_id):
        """Return ``[hold, fire]`` legality for the next submitted action."""
        state = self.get_weapon_action_state(env, agent_id)
        return np.asarray(state["shoot_action_mask"], dtype=np.bool_)


class PBGRBRICHierarchicalMultipleCombatShootTask(
        PBGRHierarchicalMultipleCombatShootTask):
    """V20 opt-in hierarchy with explicit target and fire decisions.

    The first three components retain the historical altitude, heading and
    speed subgoals.  Component three selects one immutable enemy uid slot and
    component four authorizes fire.  No historical PBGR action is reinterpreted.
    """

    def load_action_space(self):
        from gymnasium import spaces

        max_enemies = int(getattr(self.config, "pbgr_max_enemies", 2))
        if max_enemies <= 0:
            raise ValueError("BRIC requires at least one explicit enemy slot")
        self.action_space = spaces.MultiDiscrete([3, 5, 3, max_enemies, 2])

    def reset(self, env):
        result = super().reset(env)
        self._selected_target_slot = {agent_id: 0 for agent_id in env.agents}
        self._selected_target_id = {
            agent_id: self._target_id_for_slot(env.agents[agent_id], 0)
            for agent_id in env.agents
        }
        return result

    @staticmethod
    def _target_id_for_slot(agent, slot):
        slots = explicit_enemy_slots(agent)
        index = int(slot)
        if index < 0 or index >= len(slots):
            return None
        return str(slots[index].uid)

    def prepare_lowlevel_action(self, env, agent_id, action):
        action = np.asarray(action)
        if action.shape != (5,):
            raise ValueError("BRIC high-level action must have five components")
        target_slot = int(action[3])
        target_id = self._target_id_for_slot(env.agents[agent_id], target_slot)
        if target_id is None:
            raise ValueError("BRIC target slot does not exist for this aircraft")
        self._selected_target_slot[agent_id] = target_slot
        self._selected_target_id[agent_id] = target_id
        self._shoot_action[agent_id] = bool(int(action[4]) > 0)
        return HierarchicalMultipleCombatTask.prepare_lowlevel_action(
            self, env, agent_id, action[:3])

    def _requested_weapon_target_id(self, agent_id):
        if not hasattr(self, "_selected_target_id"):
            raise RuntimeError("BRIC weapon task must be reset before use")
        return self._selected_target_id[agent_id]

    def get_bric_weapon_action_state(self, env, agent_id):
        """Return immutable target slots and per-target fire masks."""
        if agent_id not in env.agents:
            raise KeyError("unknown agent_id: {0}".format(agent_id))
        agent = env.agents[agent_id]
        remaining = int(self._remaining_missiles[agent_id])
        target_rows = []
        for slot, target in enumerate(explicit_enemy_slots(agent)):
            legality, _ = evaluate_shot(
                agent=agent,
                requested=False,
                remaining_missiles=remaining,
                current_step=env.current_step,
                last_shot_step=self._last_shoot_time[agent_id],
                max_attack_angle=self.max_attack_angle,
                max_attack_distance=self.max_attack_distance,
                min_attack_interval=self.min_attack_interval,
                target_id=target.uid,
            )
            row = legality.to_dict()
            row.update({"target_slot": int(slot), "target_id": str(target.uid)})
            target_rows.append(row)
        return {
            "weapon_schema_version": "bric-explicit-target-v1",
            "target_ids": [row["target_id"] for row in target_rows],
            "target_mask": [
                "unknown_target" not in row["violations"] for row in target_rows
            ],
            "fire_masks": [row["shoot_action_mask"] for row in target_rows],
            "targets": target_rows,
            "selected_target_slot": int(self._selected_target_slot[agent_id]),
            "selected_target_id": str(self._selected_target_id[agent_id]),
        }


class PBGRV22PlatformCorrectHierarchicalMultipleCombatShootTask(
        PBGRBRICHierarchicalMultipleCombatShootTask):
    """V22 weapon task with deployable entity observations."""

    def __init__(self, config: str):
        super().__init__(config)
        gamma = float(getattr(self.config, "PotentialReward_gamma", 0.99))
        ppo_gamma = float(getattr(self.config, "PPO_gamma", gamma))
        if not np.isclose(gamma, ppo_gamma, atol=0.0, rtol=0.0):
            raise ValueError("V22 PBRS gamma must equal PPO gamma")
        self.reward_functions = [
            PBGRDiscountedPostureReward(self.config),
            AltitudeReward(self.config),
        ]

    def reset(self, env):
        result = super().reset(env)
        self._v22_known_dead_targets = {
            agent_id: set() for agent_id in env.agents
        }
        return result

    def get_reward(self, env, agent_id, info=None):
        """Settle discounted potential exactly once on an individual death."""

        if not hasattr(self, "_reward_transition_active"):
            self._reward_transition_active = {}
        agent = env.agents[agent_id]
        was_dead = bool(self._agent_die_flag.get(agent_id, False))
        first_death = (not agent.is_alive) and (not was_dead)
        self._reward_transition_active[agent_id] = bool(
            agent.is_alive or first_death
        )
        if agent.is_alive:
            return SingleCombatTask.get_reward(self, env, agent_id, info=info)
        self._agent_die_flag[agent_id] = True
        if not first_death:
            return 0.0, ({} if info is None else dict(info))
        reward = 0.0
        for reward_function in self.reward_functions:
            if isinstance(reward_function, PBGRDiscountedPostureReward):
                reward += reward_function.settle_terminal(agent_id)
        return float(reward), ({} if info is None else dict(info))

    def load_observation_space(self):
        PBGRMultipleCombatTask.load_observation_space(self)

    def get_obs(self, env, agent_id):
        return PBGRMultipleCombatTask.get_obs(self, env, agent_id)

    def prepare_lowlevel_action(self, env, agent_id, action):
        action = np.asarray(action)
        if action.shape != (5,):
            raise ValueError("V22 high-level action must have five components")
        target_slot = int(action[3])
        target_id = self._target_id_for_slot(env.agents[agent_id], target_slot)
        if target_id is None:
            raise ValueError("V22 target slot does not exist for this aircraft")
        self._selected_target_slot[agent_id] = target_slot
        self._selected_target_id[agent_id] = target_id
        self._shoot_action[agent_id] = bool(int(action[4]) > 0)
        return HierarchicalMultipleCombatTask.prepare_lowlevel_action(
            self, env, agent_id, action[:3]
        )

    def _build_lowlevel_input(self, env, agent_id, action):
        raw_obs = MultipleCombatTask.get_obs(self, env, agent_id)
        input_obs = np.zeros(12, dtype=np.float32)
        input_obs[0] = self.norm_delta_altitude[int(action[0])]
        input_obs[1] = self.norm_delta_heading[int(action[1])]
        input_obs[2] = self.norm_delta_velocity[int(action[2])]
        input_obs[3:12] = raw_obs[:9]
        return input_obs

    def get_v22_weapon_action_state(self, env, agent_id):
        """Return non-leaking target support and visible exact fire support."""
        observation = env.get_cached_entity_observation(agent_id)
        entity_index = {uid: index for index, uid in enumerate(observation.entity_ids) if uid}
        agent = env.agents[agent_id]
        remaining = int(self._remaining_missiles[agent_id])
        if not hasattr(self, "_v22_known_dead_targets"):
            self._v22_known_dead_targets = {}
        known_dead = self._v22_known_dead_targets.setdefault(agent_id, set())
        target_rows = []
        for slot, target in enumerate(explicit_enemy_slots(agent)):
            index = entity_index.get(str(target.uid))
            exists = index is not None and bool(observation.entity_mask[index])
            visible = exists and bool(observation.visible_mask[index])
            alive_observed = visible and bool(observation.alive_observed_mask[index])
            last_observed_alive = exists and bool(observation.alive_mask[index])
            target_uid = str(target.uid)
            if alive_observed:
                if last_observed_alive:
                    known_dead.discard(target_uid)
                else:
                    known_dead.add(target_uid)
            target_allowed = bool(exists and target_uid not in known_dead)
            legality, _ = evaluate_shot(
                agent=agent, requested=False, remaining_missiles=remaining,
                current_step=env.current_step,
                last_shot_step=self._last_shoot_time[agent_id],
                max_attack_angle=self.max_attack_angle,
                max_attack_distance=self.max_attack_distance,
                min_attack_interval=self.min_attack_interval,
                target_id=target_uid,
            )
            fire_allowed = bool(visible and last_observed_alive and legality.can_fire)
            target_rows.append({
                "target_slot": int(slot), "target_id": target_uid,
                "target_allowed": target_allowed, "visible": visible,
                "alive_observed": alive_observed,
                "last_observed_alive": last_observed_alive,
                "fire_mask": [True, fire_allowed],
                "legality": legality.to_dict(),
            })
        if not target_rows:
            raise RuntimeError("V22 weapon task requires an enemy target slot")
        return {
            "weapon_schema_version": "v22-platform-correct-weapon-v1",
            "target_ids": [row["target_id"] for row in target_rows],
            "target_mask": [row["target_allowed"] for row in target_rows],
            "fire_masks": [row["fire_mask"] for row in target_rows],
            "targets": target_rows,
        }
