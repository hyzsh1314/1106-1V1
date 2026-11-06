import numpy as np

from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
from envs.JSBSim.envs.singlecombat_env import SingleCombatEnv


def test_v22_batched_lowlevel_controller_matches_sequential_path():
    config = "2v2/ShootMissile/PBGR_V22_PLATFORM_CORRECT"
    sequential_env = MultipleCombatEnv(config)
    batched_env = MultipleCombatEnv(config)
    try:
        sequential_env.seed(9107)
        batched_env.seed(9107)
        sequential_env.set_initial_geometry_reset_index(0)
        batched_env.set_initial_geometry_reset_index(0)
        sequential_env.reset()
        batched_env.reset()
        agent_ids = list(sequential_env.agents.keys())
        action_rows = np.asarray([
            [0, 0, 2, 0, 0],
            [1, 2, 1, 1, 0],
            [2, 4, 0, 0, 0],
            [1, 3, 2, 1, 0],
        ], dtype=np.int64)
        actions = dict(zip(agent_ids, action_rows))

        sequential = {
            agent_id: sequential_env.task.normalize_action(
                sequential_env, agent_id, actions[agent_id]
            )
            for agent_id in agent_ids
        }
        batched = batched_env.task.normalize_actions(batched_env, actions)

        for agent_id in agent_ids:
            np.testing.assert_allclose(
                batched[agent_id], sequential[agent_id], rtol=0.0, atol=1e-7
            )
            np.testing.assert_allclose(
                batched_env.task._inner_rnn_states[agent_id],
                sequential_env.task._inner_rnn_states[agent_id],
                rtol=1e-6,
                atol=1e-6,
            )
        assert (
            batched_env.task._selected_target_id
            == sequential_env.task._selected_target_id
        )
    finally:
        sequential_env.close()
        batched_env.close()


def test_1v1_batched_lowlevel_controller_matches_sequential_path():
    config = "1v1/ShootMissile/HierarchySelfplay"
    sequential_env = SingleCombatEnv(config, reuse_fdm=True)
    batched_env = SingleCombatEnv(config, reuse_fdm=True)
    try:
        sequential_env.seed(9107)
        batched_env.seed(9107)
        sequential_env.reset()
        batched_env.reset()
        agent_ids = list(sequential_env.agents.keys())
        action_rows = np.asarray([
            [0, 0, 2, 1],
            [2, 4, 0, 0],
        ], dtype=np.int64)
        actions = dict(zip(agent_ids, action_rows))

        sequential = {
            agent_id: sequential_env.task.normalize_action(
                sequential_env, agent_id, actions[agent_id]
            )
            for agent_id in agent_ids
        }
        batched = batched_env.task.normalize_actions(batched_env, actions)

        for agent_id in agent_ids:
            np.testing.assert_array_equal(
                batched[agent_id], sequential[agent_id]
            )
            np.testing.assert_allclose(
                batched_env.task._inner_rnn_states[agent_id],
                sequential_env.task._inner_rnn_states[agent_id],
                rtol=1e-6,
                atol=1e-6,
            )
        assert (
            batched_env.task._shoot_action
            == sequential_env.task._shoot_action
        )
    finally:
        sequential_env.close()
        batched_env.close()

