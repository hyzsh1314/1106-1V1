import numpy as np

from ac_cosr_wvr.env.cosr_singlecombat_env import COSRSingleCombatEnv
from ac_cosr_wvr.env.scenario_sampler import ScenarioSampler

from .conftest import smoke_config, smoke_sampler


def test_positive_autogun_jsbsim_smoke_reaches_dwell_kill_and_outcome():
    sampler = smoke_sampler(24, 1)
    probabilities = [1.0 if value == "balanced_head_on" else 0.0 for value in sampler.strata]
    sampler = ScenarioSampler(24, 1, sampler.limits, sampler.stratum_configs, probabilities,
                              sampler.scenario_id, sampler.retry_limit, sampler.neutral_state)
    config = smoke_config()
    # The dedicated unilateral fixture uses a tail-only criterion.  Retain a
    # mutual-terminal check here as a separate resolver regression.
    config["autogun"]["min_target_aspect_deg"] = 0.0
    env = COSRSingleCombatEnv(config)
    try:
        env.reset(sampler.sample(0, 0, "validation"))
        for _ in range(6):
            result = env.step_physical(np.asarray([[0., 0., 0., .5], [0., 0., 0., .5]], dtype=np.float32))
            if result["terminated"]: break
        assert result["terminated"]
        assert result["episode_manifest"]["steps"][-1]["gun"]["red_kill"]
        assert result["outcome"].primary_reason == "simultaneous_failure"
    finally:
        env.close()
