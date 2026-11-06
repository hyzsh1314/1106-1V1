import json
from pathlib import Path

from benchmarks.train_torch_wvr_b0_rhppo_v7 import ALGORITHM


def test_v7_b0_protocol_is_frozen_before_training():
    protocol = json.loads(Path(
        "configs/drvoi_rhppo_v7_wvr.json").read_text(encoding="utf-8"))
    assert ALGORITHM == "v7_b0_extrinsic_rhppo"
    assert protocol["environment"]["frozen"] is True
    assert protocol["environment"]["physics_hz"] == 60
    assert protocol["environment"]["physics_substeps_per_decision"] == 12
    assert protocol["b0"]["training_seeds"] == [85, 185, 285]
    assert protocol["b0"]["training_environment_decisions"] == 1_007_616
    assert protocol["authorization"]["v7_long_training_authorized"] is False
