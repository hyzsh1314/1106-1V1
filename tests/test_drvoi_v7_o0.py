import pytest
import torch

from benchmarks.validate_drvoi_wvr_o0_v7 import spearman, voi_from_q_values


def test_voi_is_zero_when_response_samples_have_same_action_ranking():
    q = torch.tensor([[[2.0, 1.0], [3.0, 1.5]]])
    voi, diverse, top_flip = voi_from_q_values(q)
    assert torch.allclose(voi, torch.zeros(1))
    assert not bool(diverse[0])
    assert not bool(top_flip[0])


def test_voi_is_positive_when_response_changes_best_action():
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    voi, diverse, top_flip = voi_from_q_values(q)
    assert torch.allclose(voi, torch.tensor([0.5]))
    assert bool(diverse[0])
    assert bool(top_flip[0])


def test_spearman_handles_ties_and_monotonic_order():
    assert spearman(
        torch.tensor([1.0, 2.0, 2.0, 4.0]),
        torch.tensor([10.0, 20.0, 20.0, 40.0])) == pytest.approx(1.0)
