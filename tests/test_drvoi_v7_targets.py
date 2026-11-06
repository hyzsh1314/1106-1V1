import torch

from benchmarks.drvoi.targets import (
    TargetStandardizer,
    build_delta_targets,
    response_state_v7,
)


def test_v7_rel_and_opp_targets_use_only_legal_observation_indices():
    obs = torch.zeros(2, 39)
    obs[:, 11] = torch.tensor([0.0, torch.pi / 2])
    obs[:, 12] = torch.tensor([torch.pi, 0.0])
    obs[:, 18:21] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    rel = response_state_v7(obs, "rel")
    opp = response_state_v7(obs, "opp")
    assert rel.shape == (2, 12)
    assert opp.shape == (2, 3)
    assert torch.equal(opp, obs[:, 18:21])
    assert torch.allclose(rel[:, 2], torch.tensor([0.0, 1.0]), atol=1e-6)
    assert torch.allclose(rel[:, 5], torch.tensor([-1.0, 1.0]), atol=1e-6)


def test_delta_targets_reject_any_cross_terminal_window():
    obs = torch.zeros(6, 1, 39)
    obs[:, 0, 18] = torch.arange(6)
    done = torch.tensor([[False], [True], [False], [False], [False]])
    targets = build_delta_targets(obs, done, delta=3, branch="opp")
    assert targets.delta.shape == (3, 1, 3)
    assert targets.valid[:, 0].tolist() == [False, False, True]
    assert targets.delta[2, 0, 0] == 3.0


def test_target_standardizer_round_trip_and_state_restore():
    values = torch.tensor([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]])
    standardizer = TargetStandardizer.fit(values)
    transformed = standardizer.transform(values)
    assert torch.allclose(transformed.mean(0), torch.zeros(2), atol=1e-6)
    assert torch.allclose(standardizer.inverse(transformed), values)
    restored = TargetStandardizer.from_state_dict(standardizer.state_dict())
    assert torch.equal(restored.mean, standardizer.mean)
    assert torch.equal(restored.scale, standardizer.scale)
