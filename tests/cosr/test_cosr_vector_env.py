import numpy as np

from ac_cosr_wvr.env.vector_env import COSRSubprocVectorEnv

from .conftest import smoke_config, smoke_sampler


def test_cosr_subprocess_vector_env_preserves_synchronous_batch_contract():
    sampler = smoke_sampler(37, 1)
    scenarios = [
        sampler.sample(index, 0, "validation") for index in range(2)
    ]
    actions = np.asarray([
        [[0.0, 0.0, 0.0, 0.55], [0.0, 0.0, 0.0, 0.55]],
        [[0.1, -0.1, 0.0, 0.60], [-0.1, 0.1, 0.0, 0.60]],
    ], dtype=np.float32)
    with COSRSubprocVectorEnv(
        smoke_config(), n_envs=2, logging_mode="train"
    ) as env:
        observations = env.reset(scenarios)
        assert observations.shape == (2, 2, 15)
        result = env.step(actions)
        assert result["observations"].shape == (2, 2, 15)
        assert result["training_reward"].shape == (2, 2)
        assert result["safety_cost"].shape == (2, 2)
        assert result["terminated"].shape == (2,)
        assert result["episode_manifest"] == [None, None]
        manifests = env.get_episode_manifests()
        assert [item["scenario"]["episode_id"] for item in manifests] == [
            scenario.episode_id for scenario in scenarios
        ]
        assert all(item["steps"] == [] for item in manifests)
        replacement = sampler.sample(2, 0, "validation")
        reset_observation = env.reset_at([1], [replacement])
        assert reset_observation.shape == (1, 2, 15)
        assert env.get_episode_manifests()[1]["scenario"]["episode_id"] == (
            replacement.episode_id
        )

