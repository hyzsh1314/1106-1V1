import numpy as np

from ac_cosr_wvr.env.cosr_singlecombat_env import COSRSingleCombatEnv

from .conftest import smoke_config, smoke_sampler


def test_cosr_env_executes_continuous_physical_action_without_discretising():
    env = COSRSingleCombatEnv(smoke_config())
    try:
        env.reset(smoke_sampler(22, 1).sample(0, 0, "validation"))
        actions = np.asarray([[.1, -.1, .0, .55], [-.1, .1, .0, .55]], dtype=np.float32)
        result = env.step_physical(actions)
        assert result["observations"].shape == (2, 15)
        assert result["training_reward"].shape == (2,)
        assert result["safety_cost"].shape == (2,)
        assert result["episode_manifest"]["steps"][0]["physical_actions"] == actions.tolist()
    finally:
        env.close()


def test_cosr_train_logging_does_not_return_or_accumulate_step_manifest():
    env = COSRSingleCombatEnv(smoke_config(), logging_mode="train")
    try:
        env.reset(smoke_sampler(22, 1).sample(0, 0, "validation"))
        actions = np.asarray(
            [[.1, -.1, .0, .55], [-.1, .1, .0, .55]], dtype=np.float32
        )
        result = env.step_physical(actions)
        assert result["episode_manifest"] is None
        assert result["manifest_step_recorded"] is False
        assert env.get_episode_manifest(copy=False)["steps"] == []
        copied = env.get_episode_manifest()
        assert copied is not env.episode_manifest
        assert copied["logging"]["mode"] == "train"
    finally:
        env.close()
