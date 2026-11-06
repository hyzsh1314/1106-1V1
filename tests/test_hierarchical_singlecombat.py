import numpy as np
import pytest

from envs.JSBSim.envs.singlecombat_env import SingleCombatEnv


SCENARIO = "1v1/ShootMissile/HierarchySelfplay"


def test_lag_style_hierarchy_uses_frozen_recurrent_lowlevel_controller():
    env = SingleCombatEnv(SCENARIO, reuse_fdm=True)
    try:
        env.seed(17)
        obs = env.reset()
        assert obs.shape == (2, 24)
        assert np.all((obs[:, 21] >= 0.0) & (obs[:, 21] <= 1.0))
        assert np.all((obs[:, 22] >= 0.0) & (obs[:, 22] <= 1.0))
        assert np.all(np.isin(obs[:, 23], [0.0, 1.0]))
        np.testing.assert_array_equal(env.action_space[0].nvec, [3, 5, 3])
        assert env.action_space[1].n == 2
        assert env.task.lowlevel_policy.training is False
        assert not any(
            parameter.requires_grad
            for parameter in env.task.lowlevel_policy.parameters()
        )
        assert all(
            np.count_nonzero(state) == 0
            for state in env.task._inner_rnn_states.values()
        )

        action = np.asarray([[1, 2, 1, 0], [1, 2, 1, 0]], dtype=np.int64)
        next_obs, reward, done, _ = env.step(action)
        assert np.isfinite(next_obs).all()
        assert np.isfinite(reward).all()
        assert done.shape == (2, 1)
        assert any(
            np.count_nonzero(state) > 0
            for state in env.task._inner_rnn_states.values()
        )
    finally:
        env.close()


def test_hierarchical_shoot_action_rejects_invalid_commands():
    env = SingleCombatEnv(SCENARIO, reuse_fdm=True)
    try:
        env.reset()
        agent_id = next(iter(env.agents))
        with pytest.raises(ValueError, match="four finite integer"):
            env.task.normalize_action(env, agent_id, [1, 2, 1])
        with pytest.raises(ValueError, match="outside the action space"):
            env.task.normalize_action(env, agent_id, [3, 2, 1, 0])
    finally:
        env.close()


def test_weapon_observation_mask_and_request_telemetry_are_consistent():
    env = SingleCombatEnv(SCENARIO, reuse_fdm=True)
    try:
        env.seed(85)
        obs = env.reset()
        for index, agent_id in enumerate(env.agents):
            mask = env.task.get_weapon_action_mask(env, agent_id)
            assert mask.tolist() == [True, bool(obs[index, 23])]

        # Force fire requests from both sides.  Each request must be auditable
        # as either one accepted launch or one explicit rejection.
        action = np.asarray([[1, 2, 1, 1], [1, 2, 1, 1]], dtype=np.int64)
        _obs, _reward, _done, info = env.step(action)
        assert info["weapon_fire_requests"] == 2
        assert (info["weapon_accepted_launches"] +
                info["weapon_rejected_requests"] == 2)
        assert len(info["weapon_legality"]) == 2
        assert all(
            item["requested"] is True
            for item in info["weapon_legality"].values()
        )
    finally:
        env.close()


def test_initial_engagement_geometry_is_reachable_and_reproducible():
    env = SingleCombatEnv(SCENARIO, reuse_fdm=True)
    replica = SingleCombatEnv(SCENARIO, reuse_fdm=True)
    try:
        env.seed(1701)
        replica.seed(1701)
        for _ in range(32):
            env.reset()
            replica.reset()
            positions = [agent.get_position() for agent in env.agents.values()]
            replica_positions = [
                agent.get_position() for agent in replica.agents.values()
            ]
            distance = float(np.linalg.norm(positions[1] - positions[0]))
            assert 7900.0 <= distance <= 16100.0
            np.testing.assert_allclose(positions, replica_positions, atol=1e-6)
    finally:
        env.close()
        replica.close()


def test_timeout_adjudication_is_explicit_and_zero_sum():
    env = SingleCombatEnv(SCENARIO, reuse_fdm=True)
    try:
        env.seed(19)
        env.reset()
        env.max_steps = 1
        hold = np.asarray([[1, 2, 1, 0], [1, 2, 1, 0]], dtype=np.int64)
        _obs, rewards, dones, info = env.step(hold)
        assert np.all(dones)
        assert info["team_outcome"]["reason"].startswith("timeout_")
        assert info["team_outcome"]["ego_tactical_score"] is not None
        assert info["team_outcome"]["enemy_tactical_score"] is not None
        assert np.isclose(float(np.sum(rewards)), 0.0)
    finally:
        env.close()
