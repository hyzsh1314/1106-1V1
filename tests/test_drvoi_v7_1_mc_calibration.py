import json

import torch

from benchmarks.calibrate_drvoi_v7_1_p1q_mc import calibration_loss, load_split


def test_calibration_loss_prefers_correct_action_order():
    cfg = {
        "raw_value_loss_weight": 1.0,
        "centered_advantage_loss_weight": 2.0,
        "soft_argmax_loss_weight": 0.5,
        "soft_argmax_temperature": 0.1,
    }
    truth = torch.tensor([[0.0, 0.5, 1.0]])
    legal = torch.ones_like(truth, dtype=torch.bool)
    correct = calibration_loss(truth.clone(), truth, legal, cfg)[0]
    reversed_value = calibration_loss(truth.flip(-1), truth, legal, cfg)[0]
    assert correct < reversed_value


def test_mc_loader_rejects_formal_test_as_validation(tmp_path):
    path = tmp_path / "test.pt"
    torch.save({
        "report": {
            "split": "formal_test", "contains_formal_test": True,
            "collection_seed": 1,
        },
        "data": {},
    }, path)
    try:
        load_split([path], "validation")
    except ValueError as error:
        assert "not isolated validation" in str(error)
    else:
        raise AssertionError("formal test data must never load as validation")
