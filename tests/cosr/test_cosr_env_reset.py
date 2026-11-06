from ac_cosr_wvr.env.cosr_singlecombat_env import COSRSingleCombatEnv

from .conftest import smoke_config, smoke_sampler


def test_cosr_env_reset_uses_explicit_scenario_and_manifest():
    env = COSRSingleCombatEnv(smoke_config())
    try:
        scenario = smoke_sampler(21, 1).sample(0, 0, "validation")
        observations = env.reset(scenario)
        assert observations.shape == (2, 15)
        assert env.episode_manifest["scenario_sha256"] == scenario.sha256()
        assert len(env.episode_manifest["reload_validation"]) == 2
    finally:
        env.close()
