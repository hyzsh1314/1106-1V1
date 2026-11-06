"""Pre-action weapon masks and PBGR resource-observation boundaries."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from algorithms.pbgr.entities import (
    ENTITY_FEATURE_NAMES,
    EntityCapacityError,
    JSBSimEntityAdapter,
)
from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
from envs.JSBSim.tasks.multiplecombat_task import HierarchicalMultipleCombatShootTask
from envs.JSBSim.tasks.pbgr_multiplecombat_task import (
    PBGRHierarchicalMultipleCombatShootTask,
)


class FakeAircraft(object):
    def __init__(self, uid, position, velocity=(200.0, 0.0, 0.0),
                 health=100.0, missiles=2):
        self.uid = uid
        self._position = np.asarray(position, dtype=np.float64)
        self._velocity = np.asarray(velocity, dtype=np.float64)
        self.is_alive = True
        self.bloods = float(health)
        self.num_missiles = int(missiles)
        self.num_left_missiles = int(missiles)
        self.partners = []
        self.enemies = []
        self.under_missiles = []

    def get_position(self):
        return self._position

    def get_velocity(self):
        return self._velocity


class FakeIncomingMissile(object):
    def __init__(self, uid, position):
        self.uid = uid
        self._position = np.asarray(position, dtype=np.float64)
        self._velocity = np.asarray((-300.0, 0.0, 0.0), dtype=np.float64)
        self.is_alive = True

    def get_position(self):
        return self._position

    def get_velocity(self):
        return self._velocity


class ForbiddenPrivateScalar(object):
    """Fail a test if an actor-side adapter reads an exact private value."""

    def __float__(self):
        raise AssertionError("non-ego private state was read")

    def __int__(self):
        raise AssertionError("non-ego private state was read")


def _task(agents, remaining=None, last_shot=None):
    task = PBGRHierarchicalMultipleCombatShootTask.__new__(
        PBGRHierarchicalMultipleCombatShootTask)
    task.max_attack_angle = 45.0
    task.max_attack_distance = 14000.0
    task.min_attack_interval = 100
    task._remaining_missiles = {
        uid: int(agent.num_missiles) for uid, agent in agents.items()
    }
    task._last_shoot_time = {uid: -100 for uid in agents}
    if remaining is not None:
        task._remaining_missiles.update(remaining)
    if last_shot is not None:
        task._last_shoot_time.update(last_shot)
    return task


def _environment(ally_health=100.0, ally_ammo=2,
                 enemy_health=100.0, enemy_ammo=2,
                 current_step=50, ego_ammo=1, ego_last_shot=0):
    ego = FakeAircraft("A1", (0.0, 0.0, 0.0), health=75.0, missiles=2)
    ally = FakeAircraft("A2", (0.0, 1000.0, 0.0),
                        health=ally_health, missiles=4)
    enemy = FakeAircraft("B1", (1000.0, 0.0, 0.0),
                         health=enemy_health, missiles=6)
    ego.partners = [ally]
    ego.enemies = [enemy]
    ally.partners = [ego]
    ally.enemies = [enemy]
    enemy.enemies = [ego, ally]
    agents = {agent.uid: agent for agent in (ego, ally, enemy)}

    env = MultipleCombatEnv.__new__(MultipleCombatEnv)
    env._pbgr_enabled = True
    env._jsbsims = agents
    env.current_step = int(current_step)
    env.sim_freq = 60
    env.agent_interaction_steps = 12
    env.task = _task(
        agents,
        remaining={"A1": ego_ammo, "A2": ally_ammo, "B1": enemy_ammo},
        last_shot={"A1": ego_last_shot})
    return env


def _capacity_environment(max_teammates=1, max_enemies=2,
                          max_incoming_missiles=4):
    agents = {
        uid: FakeAircraft(uid, position, missiles=2)
        for uid, position in (
            ("A1", (0.0, 0.0, 0.0)),
            ("A2", (0.0, 1000.0, 0.0)),
            ("B1", (1000.0, 0.0, 0.0)),
            ("B2", (1000.0, 1000.0, 0.0)),
        )
    }
    for uid, agent in agents.items():
        agent.partners = [
            other for other_uid, other in agents.items()
            if other_uid != uid and other_uid[0] == uid[0]
        ]
        agent.enemies = [
            other for other_uid, other in agents.items()
            if other_uid[0] != uid[0]
        ]
    env = MultipleCombatEnv.__new__(MultipleCombatEnv)
    env._pbgr_enabled = True
    env._jsbsims = agents
    env.config = SimpleNamespace(
        pbgr_max_teammates=max_teammates,
        pbgr_max_enemies=max_enemies,
        pbgr_max_incoming_missiles=max_incoming_missiles,
        sensor_range_m=30000.0,
        missile_age_scale_s=60.0,
    )
    return env


def test_pre_action_mask_tracks_ammo_cooldown_and_geometry_without_mutation():
    env = _environment(current_step=200, ego_ammo=1, ego_last_shot=0)
    task = env.task
    remaining_before = deepcopy(task._remaining_missiles)
    last_shot_before = deepcopy(task._last_shoot_time)

    np.testing.assert_array_equal(
        task.get_weapon_action_mask(env, "A1"), [True, True])
    state = task.get_weapon_action_state(env, "A1")
    assert state["shoot_legal"] is True
    assert state["ammo_fraction"] == 0.5
    assert state["cooldown_fraction"] == 1.0
    assert task._remaining_missiles == remaining_before
    assert task._last_shoot_time == last_shot_before

    task._remaining_missiles["A1"] = 0
    np.testing.assert_array_equal(
        task.get_weapon_action_mask(env, "A1"), [True, False])
    assert "no_ammunition" in task.get_weapon_action_state(
        env, "A1")["violations"]

    task._remaining_missiles["A1"] = 1
    task._last_shoot_time["A1"] = 190
    cooldown = task.get_weapon_action_state(env, "A1")
    np.testing.assert_array_equal(cooldown["shoot_action_mask"], [True, False])
    assert "cooldown" in cooldown["violations"]
    assert np.isclose(cooldown["cooldown_fraction"], 0.1)

    task._last_shoot_time["A1"] = 0
    env.agents["B1"]._position[:] = (20000.0, 0.0, 0.0)
    assert "out_of_range" in task.get_weapon_action_state(
        env, "A1")["violations"]
    env.agents["B1"]._position[:] = (-1000.0, 0.0, 0.0)
    assert "outside_firing_cone" in task.get_weapon_action_state(
        env, "A1")["violations"]


def test_environment_exposes_independent_per_agent_pre_action_masks():
    env = _environment(current_step=200, ego_ammo=1, ego_last_shot=0)
    env.task._remaining_missiles["A2"] = 0
    masks = env.get_weapon_action_masks()
    np.testing.assert_array_equal(masks["A1"], [True, True])
    np.testing.assert_array_equal(masks["A2"], [True, False])
    masks["A1"][1] = False
    # Accessors return fresh values; caller mutation cannot alter task state.
    np.testing.assert_array_equal(
        env.get_weapon_action_mask("A1"), [True, True])


def test_pbgr_capable_task_cannot_bypass_explicit_environment_opt_in():
    env = _environment(current_step=200, ego_ammo=1, ego_last_shot=0)
    env._pbgr_enabled = False

    with pytest.raises(RuntimeError, match="PBGR opt-in"):
        env.get_weapon_action_mask("A1")


def test_entity_capacity_uses_each_observers_opponent_total_ammunition():
    env = _environment()
    # A observers face one B aircraft with one initial missile; the B
    # observer faces two A aircraft carrying two and four missiles.
    env.agents["B1"].num_missiles = 1
    adapter = JSBSimEntityAdapter(1, 2, 6)

    requirements = adapter.validate_environment_capacity(env)

    assert requirements["A1"]["incoming_missiles"] == 1
    assert requirements["A2"]["incoming_missiles"] == 1
    assert requirements["B1"]["incoming_missiles"] == 6


@pytest.mark.parametrize("value", (True, 1.0, "1", -1, np.float64(1.0)))
def test_entity_slot_capacities_reject_non_integral_or_negative_values(value):
    with pytest.raises(ValueError, match="max_teammates"):
        JSBSimEntityAdapter(value, 2, 4)


@pytest.mark.parametrize("value", (True, 2.0, "2", -1, np.float64(2.0)))
def test_initial_ammunition_rejects_non_integral_or_negative_values(value):
    env = _capacity_environment()
    env.agents["B1"].num_missiles = value

    with pytest.raises(EntityCapacityError, match="num_missiles"):
        env._initialize_pbgr_entity_adapter()
    assert not hasattr(env, "entity_adapter")


def test_numpy_integer_capacity_and_ammunition_are_accepted():
    env = _capacity_environment(
        max_teammates=np.int64(1),
        max_enemies=np.int64(2),
        max_incoming_missiles=np.int64(4),
    )
    for agent in env.agents.values():
        agent.num_missiles = np.int64(2)

    env._initialize_pbgr_entity_adapter()

    assert env.entity_adapter.max_missiles == 4
    assert env.entity_adapter.validate_environment_capacity(env)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("pbgr_max_teammates", 0, "required_teammates=1"),
        ("pbgr_max_enemies", 1, "required_enemies=2"),
        (
            "pbgr_max_incoming_missiles",
            3,
            "required_incoming_missiles=4",
        ),
    ),
)
def test_pbgr_environment_initialization_rejects_low_entity_capacity(
    field, value, message
):
    kwargs = {
        "max_teammates": 1,
        "max_enemies": 2,
        "max_incoming_missiles": 4,
    }
    key = {
        "pbgr_max_teammates": "max_teammates",
        "pbgr_max_enemies": "max_enemies",
        "pbgr_max_incoming_missiles": "max_incoming_missiles",
    }[field]
    kwargs[key] = value
    env = _capacity_environment(**kwargs)

    with pytest.raises(EntityCapacityError, match=message):
        env._initialize_pbgr_entity_adapter()
    assert not hasattr(env, "entity_adapter")


def test_entity_build_rejects_aircraft_slot_truncation_before_slicing():
    env = _environment()

    with pytest.raises(EntityCapacityError, match="teammate slots exhausted"):
        JSBSimEntityAdapter(0, 1, 0).build(env, "A1")
    with pytest.raises(EntityCapacityError, match="enemy slots exhausted"):
        JSBSimEntityAdapter(1, 0, 0).build(env, "A1")


def test_entity_build_rejects_new_missile_when_stable_slots_are_full():
    env = _environment()
    ego = env.agents["A1"]
    ego.under_missiles = [FakeIncomingMissile("M1", (500.0, 0.0, 0.0))]
    adapter = JSBSimEntityAdapter(
        1, 1, 1, visibility_range_m=5000.0)
    first = adapter.build(env, "A1")
    assert "M1" in first.entity_ids

    ego.under_missiles.append(
        FakeIncomingMissile("M2", (600.0, 0.0, 0.0)))
    with pytest.raises(EntityCapacityError, match="slots exhausted"):
        adapter.build(env, "A1")


def test_non_ego_exact_health_and_ammunition_never_enter_actor_features():
    nominal = _environment(ally_health=100.0, ally_ammo=4,
                           enemy_health=100.0, enemy_ammo=6)
    changed = _environment(ally_health=1.0, ally_ammo=0,
                           enemy_health=17.0, enemy_ammo=1)
    for uid in ("A2", "B1"):
        nominal.agents[uid].bloods = ForbiddenPrivateScalar()
        nominal.agents[uid].num_missiles = ForbiddenPrivateScalar()
        nominal.agents[uid].num_left_missiles = ForbiddenPrivateScalar()
    adapter_a = JSBSimEntityAdapter(1, 1, 0, visibility_range_m=5000.0)
    adapter_b = JSBSimEntityAdapter(1, 1, 0, visibility_range_m=5000.0)
    obs_a = adapter_a.build(nominal, "A1")
    obs_b = adapter_b.build(changed, "A1")
    np.testing.assert_array_equal(obs_a.features, obs_b.features)

    index = {name: ENTITY_FEATURE_NAMES.index(name) for name in (
        "health_fraction", "health_observed", "ammo_fraction",
        "ammo_observed", "cooldown_observed", "shoot_legal_observed")}
    for row in (1, 2):
        assert obs_a.features[row, index["health_fraction"]] == 0.0
        assert obs_a.features[row, index["health_observed"]] == 0.0
        assert obs_a.features[row, index["ammo_fraction"]] == 0.0
        assert obs_a.features[row, index["ammo_observed"]] == 0.0
        assert obs_a.features[row, index["cooldown_observed"]] == 0.0
        assert obs_a.features[row, index["shoot_legal_observed"]] == 0.0


def test_ego_observes_own_ammo_cooldown_and_shoot_legality():
    env = _environment(current_step=50, ego_ammo=1, ego_last_shot=0)
    obs = JSBSimEntityAdapter(
        1, 1, 0, visibility_range_m=5000.0).build(env, "A1")
    index = {name: ENTITY_FEATURE_NAMES.index(name)
             for name in ENTITY_FEATURE_NAMES}
    ego = obs.features[0]
    assert ego[index["health_observed"]] == 1.0
    assert np.isclose(ego[index["health_fraction"]], 0.75)
    assert ego[index["ammo_observed"]] == 1.0
    assert np.isclose(ego[index["ammo_fraction"]], 0.5)
    assert ego[index["cooldown_observed"]] == 1.0
    assert np.isclose(ego[index["cooldown_fraction"]], 0.5)
    assert ego[index["shoot_legal_observed"]] == 1.0
    assert ego[index["shoot_legal"]] == 0.0

    env.current_step = 100
    ready = JSBSimEntityAdapter(
        1, 1, 0, visibility_range_m=5000.0).build(env, "A1")
    assert ready.features[0, index["cooldown_fraction"]] == 1.0
    assert ready.features[0, index["shoot_legal"]] == 1.0


def test_legacy_flat_observation_and_action_dimensions_are_unchanged():
    task = HierarchicalMultipleCombatShootTask.__new__(
        HierarchicalMultipleCombatShootTask)
    task.load_observation_space()
    task.load_action_space()
    assert task.observation_space.shape == (33,)
    np.testing.assert_array_equal(task.action_space.nvec, [3, 5, 3, 2])
    assert not hasattr(task, "get_weapon_action_state")
    assert not hasattr(task, "get_weapon_action_mask")
    legacy_env = MultipleCombatEnv.__new__(MultipleCombatEnv)
    legacy_env._pbgr_enabled = False
    legacy_env.task = task
    with pytest.raises(RuntimeError, match="PBGR opt-in"):
        legacy_env.get_weapon_action_mask("A1")
    with pytest.raises(RuntimeError, match="PBGR opt-in"):
        legacy_env.get_entity_observation("A1")


def test_pbgr_config_opt_in_installs_task_adapter_and_pre_action_masks():
    pbgr_env = MultipleCombatEnv("2v2/ShootMissile/PBGR_R1")
    legacy_env = MultipleCombatEnv("2v2/ShootMissile/HierarchySelfplay")
    try:
        assert isinstance(
            pbgr_env.task, PBGRHierarchicalMultipleCombatShootTask)
        assert pbgr_env._pbgr_enabled is True
        assert legacy_env._pbgr_enabled is False
        assert hasattr(pbgr_env, "entity_adapter")
        assert not hasattr(legacy_env, "entity_adapter")
        requirements = (
            pbgr_env.entity_adapter.validate_environment_capacity(pbgr_env)
        )
        assert all(
            requirement == {
                "teammates": 1,
                "enemies": 2,
                "incoming_missiles": 4,
            }
            for requirement in requirements.values()
        )

        pbgr_env.seed(0)
        pbgr_env.reset()
        entity = pbgr_env.get_entity_observation(pbgr_env.ego_ids[0])
        assert entity.features.shape[0] == 8
        assert entity.entity_ids[0] == pbgr_env.ego_ids[0]
        masks = pbgr_env.get_weapon_action_masks()
        assert tuple(masks) == tuple(pbgr_env.agents)
        for mask in masks.values():
            assert mask.shape == (2,)
            assert bool(mask[0])
    finally:
        pbgr_env.close()
        legacy_env.close()
