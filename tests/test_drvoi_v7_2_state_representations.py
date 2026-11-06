import torch

from benchmarks.collect_drvoi_v7_2_state_sufficiency_data import (
    build_representations,
)


def test_v7_2_representation_dimensions_and_field_sensitivity():
    batch = 4
    base_belief = torch.randn(batch, 256)
    base_action = torch.randn(batch, 13)
    future_belief = torch.randn(batch, 256)
    observation = torch.randn(batch, 39)
    lowlevel = torch.randn(batch, 128)
    result = build_representations(
        base_belief, base_action, future_belief, observation, lowlevel)
    assert {key: value.shape[-1] for key, value in result.items()} == {
        "registered_v7_1": 281,
        "full_observation": 308,
        "future_belief": 295,
        "mechanistic_oracle": 423,
    }
    changed = build_representations(
        base_belief, base_action, future_belief, observation, lowlevel + 1)
    assert torch.equal(
        result["registered_v7_1"], changed["registered_v7_1"])
    assert not torch.equal(
        result["mechanistic_oracle"], changed["mechanistic_oracle"])
