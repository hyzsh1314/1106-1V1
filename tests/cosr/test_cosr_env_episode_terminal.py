from dataclasses import replace

import numpy as np

from ac_cosr_wvr.env.cosr_singlecombat_env import COSRSingleCombatEnv

from .conftest import smoke_config, smoke_sampler


def test_cosr_env_resolves_safety_terminal_without_legacy_agent_order():
    config = smoke_config()
    scenario = smoke_sampler(23, 1).sample(0, 0, "validation")
    red = dict(scenario.red_init); red["altitude_m"] = config["safety"]["min_altitude_m"] - 100.0
    unsafe = replace(scenario, red_init=red)
    env = COSRSingleCombatEnv(config)
    try:
        env.reset(unsafe)
        result = env.step_physical(np.asarray([[0., 0., 0., .5], [0., 0., 0., .5]], dtype=np.float32))
        assert result["terminated"] is True
        assert result["outcome"].winner == "blue"
        assert "red:low_altitude" in result["outcome"].all_reasons
    finally:
        env.close()
