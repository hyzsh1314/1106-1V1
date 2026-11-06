from ac_cosr_wvr.env.cosr_singlecombat_env import COSRSingleCombatEnv

from .conftest import smoke_config, smoke_sampler
from .physical_helpers import run_to_terminal, tail_chase_spec


def test_unilateral_tail_chase_jsbsim_smoke_and_side_swap_flip_winner():
    env = COSRSingleCombatEnv(smoke_config())
    try:
        red = run_to_terminal(env, tail_chase_spec(smoke_sampler(83), 0, False))
        blue = run_to_terminal(env, tail_chase_spec(smoke_sampler(83), 1, True))
        assert red["outcome"].winner == "red" and red["episode_manifest"]["steps"][-1]["gun"]["blue_kill"]
        assert not red["episode_manifest"]["steps"][-1]["gun"]["red_kill"]
        assert blue["outcome"].winner == "blue" and blue["episode_manifest"]["steps"][-1]["gun"]["red_kill"]
        assert not blue["episode_manifest"]["steps"][-1]["gun"]["blue_kill"]
    finally:
        env.close()
