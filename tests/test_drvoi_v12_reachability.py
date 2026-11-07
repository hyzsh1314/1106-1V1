import torch

from benchmarks.drvoi_v12 import (
    aggregate_hypothesis_targets,
    instantaneous_reachability_components,
    instantaneous_reachability_quality,
    rollout_reachability_branch,
)
from envs.torch6dof import TorchWVRConfig, TorchWVRVecEnv


def make_env(max_steps=20):
    return TorchWVRVecEnv(
        TorchWVRConfig(
            num_envs=2, max_episode_steps=max_steps,
            reset_pool_size=16),
        device="cpu", seed=85, compile_dynamics=False)


def test_reachability_components_reconstruct_dynamic_envelope():
    observation = torch.zeros(2, 39)
    observation[:, 13] = torch.tensor([0.5, 1.0])
    observation[:, 28] = torch.tensor([1.0, 0.0])
    observation[:, 11] = torch.tensor([0.0, 1.0])
    observation[:, 12] = torch.tensor([torch.pi, 0.0])
    components = instantaneous_reachability_components(observation)
    assert components["dynamic_range_m"].shape == (2,)
    assert components["range_margin"][0] > components["range_margin"][1]
    assert components["angle_margin"][0] > components["angle_margin"][1]


def test_normalized_softmin_maps_zero_margins_to_zero():
    observation = torch.zeros(1, 39)
    components = instantaneous_reachability_components(observation)
    # Construct an observation whose individual components are not all zero,
    # then verify the implementation against the explicit normalized formula.
    margins = torch.stack((
        components["range_margin"], components["angle_margin"],
        components["closure_margin"],
        components["los_stability_margin"],
    ), -1)
    expected = (
        -torch.logsumexp(-10.0 * margins, -1)
        + torch.log(torch.tensor(4.0))
    ) / 10.0
    torch.testing.assert_close(
        instantaneous_reachability_quality(observation), expected)


def test_hypothesis_aggregation_keeps_soft_probability_and_censoring():
    score = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    legal = torch.tensor([[True, False], [False, False]])
    first = torch.tensor([[2, 31], [31, 31]])
    target = aggregate_hypothesis_targets(
        score, legal, first, temperature=0.25)
    assert target["score"].shape == (2,)
    assert target["score"][0] < score[0].mean()
    torch.testing.assert_close(
        target["probability"], torch.tensor([0.5, 0.0]))
    torch.testing.assert_close(target["time"], torch.tensor([2.0, 0.0]))
    assert target["event_observed"].tolist() == [True, False]


def test_branch_masks_post_terminal_auto_reset_states():
    env = make_env(max_steps=1)
    learner_side = torch.tensor([0, 1])

    def policy_step(environment, state, active):
        controls = torch.zeros(environment.config.num_envs, 2, 4)
        controls[..., 3] = 0.7
        fire = torch.zeros(
            environment.config.num_envs, 2, dtype=torch.bool)
        return controls, fire, state

    result, _ = rollout_reachability_branch(
        env, learner_side, {}, policy_step, horizon=5)
    assert result.model_decisions == 2
    assert result.terminal_step.tolist() == [1, 1]
    assert torch.isfinite(result.score).all()
    assert result.fire_requests.sum() == 0
    assert result.accepted_launches.sum() == 0
    assert torch.isfinite(result.discounted_return).all()
    assert result.terminal_outcome.tolist() == [0.0, 0.0]
    assert result.timeout.tolist() == [True, True]


def test_branch_post_step_receives_actual_terminal_mask():
    env = make_env(max_steps=1)
    learner_side = torch.tensor([0, 1])

    def policy_step(environment, state, active):
        controls = torch.zeros(environment.config.num_envs, 2, 4)
        controls[..., 3] = 0.7
        fire = torch.zeros(environment.config.num_envs, 2, dtype=torch.bool)
        return controls, fire, state

    def post_step(environment, state, done, info):
        state["done_seen"] = done.clone()
        return state

    _, state = rollout_reachability_branch(
        env, learner_side, {}, policy_step, policy_post_step=post_step,
        horizon=5)
    assert state["done_seen"].tolist() == [True, True]
