import numpy as np

from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv


def test_opt_in_reused_fdm_stays_close_to_recreated_r6_trajectory():
    config = "2v2/NoWeapon/PBGR_R6_Entity_NoWeapon"
    reference = MultipleCombatEnv(config, reuse_fdm=False)
    reused = MultipleCombatEnv(config, reuse_fdm=True)
    try:
        reference.seed(61102)
        reused.seed(61102)
        reference.set_initial_geometry_reset_index(0)
        reused.set_initial_geometry_reset_index(0)
        old_reference_exec = reference.agents["A0100"].jsbsim_exec
        old_reused_exec = reused.agents["A0100"].jsbsim_exec
        ref_obs, ref_state = reference.reset()
        fast_obs, fast_state = reused.reset()

        assert reference.agents["A0100"].jsbsim_exec is not old_reference_exec
        assert reused.agents["A0100"].jsbsim_exec is old_reused_exec
        np.testing.assert_allclose(fast_obs, ref_obs, rtol=0.0, atol=1e-7)
        np.testing.assert_allclose(fast_state, ref_state, rtol=0.0, atol=1e-7)

        action = np.asarray([[20, 20, 20, 20]] * 4, dtype=np.int64)
        for _ in range(20):
            ref_result = reference.step(action)
            fast_result = reused.step(action)
            np.testing.assert_allclose(
                fast_result[0], ref_result[0], rtol=1e-7, atol=2e-6
            )
            np.testing.assert_allclose(
                fast_result[1], ref_result[1], rtol=1e-7, atol=2e-6
            )
            np.testing.assert_allclose(
                fast_result[2], ref_result[2], rtol=1e-7, atol=2e-7
            )
            np.testing.assert_array_equal(fast_result[3], ref_result[3])
    finally:
        reference.close()
        reused.close()

