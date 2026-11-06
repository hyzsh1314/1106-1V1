import json
from pathlib import Path


def test_v7_1_preregistration_freezes_history_and_new_seeds():
    config = json.loads(Path(
        "configs/drvoi_rhppo_v7_1_wvr.json").read_text(encoding="utf-8"))
    assert config["base_commit"] == "73733ddcbfe48b04b9c267d4f86e8bda976d2896"
    assert config["history"]["length"] == 3
    assert config["response"]["primary_branch"] == "rel"
    assert config["response"]["primary_delta"] == 3
    assert config["statistics"]["main_min_delta_nll_per_dimension"] == 0.05
    assert config["data"]["collection_seeds"] == [
        10185, 11185, 12185, 13185, 14185]
    assert config["data"]["old_v7_test_reuse_forbidden"] is True
    assert not any(config["authorization"].values())


def test_v7_1_history_dimensions_are_internally_consistent():
    config = json.loads(Path(
        "configs/drvoi_rhppo_v7_1_wvr.json").read_text(encoding="utf-8"))
    history = config["history"]
    assert history["observation_history_dim"] == 3 * 39
    assert history["action_history_dim"] == 3 * 13
    assert history["context_dim"] == 3 * 39 + 3
    assert history["conditioned_total_dim"] == 3 * 39 + 3 + 3 * 13
