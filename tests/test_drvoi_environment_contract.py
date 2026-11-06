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


def test_drvoi_v7_protocol_is_preregistered_and_long_training_locked():
    protocol = json.loads(Path("configs/drvoi_rhppo_v7_wvr.json").read_text(
        encoding="utf-8"))
    assert protocol["response"]["primary_branch"] == "rel"
    assert protocol["response"]["primary_delta"] == 3
    assert protocol["response"]["delta_list"] == [1, 3, 5]
    assert protocol["response"]["continuous_only"] is True
    assert protocol["response"]["side_in_primary_candidates"] is False
    assert protocol["authorization"]["v7_long_training_authorized"] is False
    assert protocol["authorization"]["manual_override_allowed"] is False
    assert "response.primary_delta" in protocol["locked_primary_fields"]


def test_v7_machine_readable_schemas_match_environment_contract():
    observation = json.loads(Path("schemas/obs_schema_v4.json").read_text(
        encoding="utf-8"))
    response = json.loads(Path("schemas/response_schema_v7.json").read_text(
        encoding="utf-8"))
    delta = json.loads(Path("schemas/delta_protocol_v7.json").read_text(
        encoding="utf-8"))
    assert observation["dimension"] == TorchWVRVecEnv.observation_size == 39
    assert response["branches"]["rel"]["dimension"] == 12
    assert response["branches"]["opp"]["dimension"] == 3
    assert response["primary_delta"] == delta["primary_delta"] == 3
    assert response["side"]["included_in_primary_candidate"] is False
