import json
from pathlib import Path

from envs.torch6dof import TorchWVRConfig, TorchWVRVecEnv


def test_drvoi_frozen_environment_contract_matches_platform():
    protocol = json.loads(Path("configs/drvoi_rhppo_v6_wvr.json").read_text(
        encoding="utf-8"))
    environment = protocol["environment"]
    config = TorchWVRConfig(
        num_envs=1, reward_mode="competitive", observation_version="v4")
    assert environment["observation_dim"] == TorchWVRVecEnv.observation_size == 39
    assert environment["action_dims"] == [3, 5, 3, 2]
    assert environment["joint_actions"] == 90
    assert environment["physics_hz"] == config.sim_freq_hz == 60
    assert environment["physics_substeps_per_decision"] == (
        config.substeps_per_decision) == 12
    assert environment["reward_mode"] == config.reward_mode == "competitive"


def test_drvoi_hard_gates_are_frozen_before_long_training():
    protocol = json.loads(Path("configs/drvoi_rhppo_v6_wvr.json").read_text(
        encoding="utf-8"))
    gates = protocol["p1_gates"]
    assert gates["G1_samples_min"] >= 1000
    assert gates["G4_false_positive_max"] == 0.05
    assert gates["G6_mc_spearman_min"] == 0.9
