"""PBGR-only missile terminal-state and numerical-safety regressions."""

from types import SimpleNamespace

import numpy as np

from envs.JSBSim.core.pbgr_missile_simulator import (
    PBGRGenericWVRMissileSimulator,
    PBGRMissileSimulator,
)
from envs.JSBSim.core.simulatior import MissileSimulator
from envs.JSBSim.reward_functions.event_driven_reward import EventDrivenReward
from envs.JSBSim.tasks.pbgr_multiplecombat_task import (
    PBGRHierarchicalMultipleCombatShootTask,
)


class StubAircraft(object):
    def __init__(self, uid, position, velocity, color="Red", dt=1.0 / 60.0):
        self.uid = uid
        self.color = color
        self.dt = float(dt)
        self.lon0, self.lat0, self.alt0 = 120.0, 60.0, 0.0
        self._position = np.asarray(position, dtype=np.float64)
        self._velocity = np.asarray(velocity, dtype=np.float64)
        self._geodetic = np.asarray((120.0, 60.0, 6000.0), dtype=np.float64)
        self._rpy = np.zeros(3, dtype=np.float64)
        self.launch_missiles = []
        self.under_missiles = []
        self.is_alive = True
        self.is_shotdown = False
        self.is_crash = False
        self.shotdown_calls = 0

    def get_position(self):
        return self._position

    def get_velocity(self):
        return self._velocity

    def get_geodetic(self):
        return self._geodetic

    def get_rpy(self):
        return self._rpy

    def shotdown(self):
        self.shotdown_calls += 1
        self.is_alive = False
        self.is_shotdown = True


def _create_pair(target_position=(100.0, 0.0, 0.0),
                 parent_velocity=(300.0, 0.0, 0.0)):
    parent = StubAircraft("A1", (0.0, 0.0, 0.0), parent_velocity)
    target = StubAircraft("B1", target_position, (0.0, 0.0, 0.0),
                          color="Blue")
    return parent, target


def test_hit_is_absorbing_across_twelve_substeps_and_event_is_once():
    parent, target = _create_pair()
    missile = PBGRMissileSimulator.create(
        parent, target, "A1M1", "simple_kinematic_PN_R1")

    for _ in range(12):
        missile.run()

    assert missile.is_done
    assert missile.is_success
    assert not missile.is_alive
    assert target.shotdown_calls == 1
    # Only the first substep advances time; terminal calls are no-ops.
    assert np.isclose(missile._t, missile.dt)

    task = PBGRHierarchicalMultipleCombatShootTask.__new__(
        PBGRHierarchicalMultipleCombatShootTask)
    task._known_missile_outcomes = set()
    task._step_weapon_events = []
    task._weapon_events = []
    env = SimpleNamespace(
        agents={"A1": parent, "B1": target}, current_step=1)
    task._record_resolved_missiles(env)
    task._record_resolved_missiles(env)

    hit_events = [
        event for event in task._weapon_events
        if event.event_type == "hit"
    ]
    assert len(hit_events) == 1
    assert hit_events[0].missile_id == "A1M1"


def test_degenerate_guidance_and_low_speed_fail_closed_with_finite_state():
    # Vertical line of sight makes Rxy=0 in the historical guidance formula.
    parent, target = _create_pair(target_position=(0.0, 0.0, 1000.0))
    vertical = PBGRMissileSimulator.create(parent, target, "A1M2")
    action, distance = vertical._guidance()
    assert np.all(np.isfinite(action))
    assert np.isfinite(distance)
    np.testing.assert_array_equal(action, np.zeros(2))
    vertical.run()
    assert np.all(np.isfinite(vertical.get_position()))
    assert np.all(np.isfinite(vertical.get_velocity()))
    assert np.all(np.isfinite(vertical.get_rpy()))

    # A zero-speed missile must terminate as MISS without dividing by speed.
    parent, target = _create_pair(
        target_position=(1000.0, 0.0, 0.0),
        parent_velocity=(0.0, 0.0, 0.0),
    )
    stopped = PBGRMissileSimulator.create(parent, target, "A1M3")
    action, distance = stopped._guidance()
    assert np.all(np.isfinite(action))
    assert np.isfinite(distance)
    stopped.run()
    assert stopped.is_done
    assert not stopped.is_success
    assert np.all(np.isfinite(stopped.get_position()))
    assert np.all(np.isfinite(stopped.get_velocity()))


def test_pbgr_factory_is_opt_in_and_legacy_factory_type_is_unchanged():
    parent, target = _create_pair(target_position=(1000.0, 0.0, 0.0))
    legacy = MissileSimulator.create(parent, target, "LEGACY")
    parent2, target2 = _create_pair(target_position=(1000.0, 0.0, 0.0))
    pbgr = PBGRMissileSimulator.create(parent2, target2, "PBGR")

    assert type(legacy) is MissileSimulator
    assert type(pbgr) is PBGRMissileSimulator

    target._position[:] = (100.0, 0.0, 0.0)
    for _ in range(12):
        legacy.run()
    assert legacy.is_success
    assert legacy.resolution_role == "kill"
    assert target.shotdown_calls == 1


def _synchronized_capture_roles(reverse=False):
    target = StubAircraft("B1", (10.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                          color="Blue")
    parents = [
        StubAircraft("A1", (0.0, 0.0, 0.0), (300.0, 0.0, 0.0)),
        StubAircraft("A2", (0.0, 0.0, 0.0), (300.0, 0.0, 0.0)),
    ]
    missiles = [
        PBGRGenericWVRMissileSimulator.create(parent, target, parent.uid + "M")
        for parent in parents
    ]
    propagation_order = list(reversed(missiles)) if reverse else missiles
    for missile in propagation_order:
        missile.run(defer_target_effect=True)
    PBGRGenericWVRMissileSimulator.resolve_deferred_target_effects(
        propagation_order)
    return target, {
        missile.uid: (missile.is_success, missile.resolution_role,
                      missile.resolution_reason)
        for missile in missiles
    }


def test_synchronized_capture_is_invariant_to_missile_iteration_order():
    forward_target, forward = _synchronized_capture_roles(reverse=False)
    reverse_target, reverse = _synchronized_capture_roles(reverse=True)

    assert forward_target.shotdown_calls == 1
    assert reverse_target.shotdown_calls == 1
    assert forward == reverse
    assert sum(role == "kill" for _, role, _ in forward.values()) == 1
    assert sum(role == "redundant_hit" for _, role, _ in forward.values()) == 1
    assert all(success for success, _, _ in forward.values())


def test_already_resolved_target_is_invalidated_not_counted_as_miss():
    parent, target = _create_pair(target_position=(1000.0, 0.0, 0.0))
    missile = PBGRGenericWVRMissileSimulator.create(parent, target, "A1M4")
    target.shotdown()
    missile.run(defer_target_effect=True)

    task = PBGRHierarchicalMultipleCombatShootTask.__new__(
        PBGRHierarchicalMultipleCombatShootTask)
    task._known_missile_outcomes = set()
    task._step_weapon_events = []
    task._weapon_events = []
    task._weapon_legality = {}
    task._weapon_decision_step = 3
    task._weapon_transition_step = 4
    env = SimpleNamespace(
        agents={"A1": parent, "B1": target}, current_step=4)

    task._record_resolved_missiles(env)
    info = PBGRHierarchicalMultipleCombatShootTask.get_weapon_step_info(task)

    assert task._weapon_events[0].event_type == "invalidated"
    assert task._weapon_events[0].reason == "target_already_resolved"
    assert info["weapon_invalidated_count"] == 1
    assert info["weapon_miss_count"] == 0
    assert info["weapon_decision_step"] == 3
    assert info["weapon_transition_step"] == 4


def test_legacy_event_reward_is_exactly_once_for_potential_and_dense_modes():
    for potential in (False, True):
        parent, target = _create_pair()
        env = SimpleNamespace(agents={"A1": parent, "B1": target})
        task = SimpleNamespace()
        config = SimpleNamespace(
            EventDrivenReward_scale=1.0,
            EventDrivenReward_potential=potential,
        )
        reward = EventDrivenReward(config)
        reward.reset(task, env)
        missile = MissileSimulator.create(parent, target, "A1REWARD")
        missile.run()

        assert reward.get_reward(task, env, "A1") == 200.0
        assert reward.get_reward(task, env, "A1") == 0.0
        assert reward.get_reward(task, env, "B1") == -200.0
        assert reward.get_reward(task, env, "B1") == 0.0
