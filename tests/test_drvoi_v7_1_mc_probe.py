import torch

from benchmarks.validate_drvoi_v7_1_p1q_mc_probe import (
    action_ids_to_components,
)


def test_action_id_components_round_trip_all_90_actions():
    ids = torch.arange(90)
    components = action_ids_to_components(ids)
    reconstructed = (((components[:, 0] * 5 + components[:, 1]) * 3
                      + components[:, 2]) * 2 + components[:, 3])
    assert torch.equal(reconstructed, ids)
