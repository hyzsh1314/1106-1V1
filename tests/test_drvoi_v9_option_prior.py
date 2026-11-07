import torch

from benchmarks.drvoi_v9 import confidence_gated_option_prior


def test_agreeing_folds_produce_confident_legal_prior():
    q_a = torch.tensor([[[0.0, 2.0, 1.0], [0.0, 3.0, 1.0]]])
    q_b = q_a + 0.01
    result = confidence_gated_option_prior(
        q_a, q_b, torch.tensor([[0.5, 0.5]]),
        torch.tensor([[True, True, False]]), gap_temperature=0.25)
    assert result["selected_option"].item() == 1
    assert result["confidence"].item() > 0.9
    assert result["prior_logits"][0, 2] < -1.0e20


def test_disagreeing_folds_are_softly_suppressed():
    q_a = torch.tensor([[[0.0, 3.0], [0.0, 3.0]]])
    q_b = torch.tensor([[[3.0, 0.0], [3.0, 0.0]]])
    result = confidence_gated_option_prior(
        q_a, q_b, torch.ones(1, 2), torch.ones(1, 2, dtype=torch.bool),
        gap_temperature=0.25)
    assert result["confidence"].item() < 1.0e-4
    assert torch.isfinite(result["prior_logits"]).all()


def test_weights_are_normalized_and_bad_shapes_rejected():
    q = torch.zeros(2, 3, 4)
    legal = torch.ones(2, 4, dtype=torch.bool)
    result = confidence_gated_option_prior(
        q, q, torch.full((2, 3), 2.0), legal)
    assert torch.allclose(result["confidence"], torch.ones(2))
    try:
        confidence_gated_option_prior(q, q, torch.ones(2, 4), legal)
    except ValueError as exc:
        assert "weights" in str(exc)
    else:
        raise AssertionError("shape mismatch must raise")
