import torch

from benchmarks.collect_drvoi_v7_1_development_data import (
    add_explicit_histories,
    reactive_diagnostic_controls,
    validate_development_payload,
)
from benchmarks.validate_drvoi_wvr_history_response_v7_1_dev import (
    ablate_head,
    flatten_history_response,
)


def synthetic_episode(split="train"):
    observation = torch.randn(9, 39)
    action = torch.stack((
        torch.arange(9) % 3,
        torch.arange(9) % 5,
        torch.arange(9) % 3,
        torch.arange(9) % 2,
    ), -1)
    episode = {
        "observation": observation,
        "action": action,
        "done": torch.tensor([False] * 8 + [True]),
        "split": split,
        "episode_id": 1,
        "opponent": 0,
        "collection_seed": 10185,
    }
    add_explicit_histories(episode)
    return episode


def test_development_flatten_uses_120_context_and_39_action_dimensions():
    flat = flatten_history_response([synthetic_episode()], "train", "rel")
    assert flat.context.shape == (6, 120)
    assert flat.action.shape == flat.wrong_action.shape == (6, 39)
    assert flat.target.shape == (6, 12)


def test_development_payload_rejects_confirmation_split():
    report = {"b0_seed": 85, "contains_confirmation_test": False}
    validate_development_payload(report, [synthetic_episode("validation")])
    try:
        validate_development_payload(report, [synthetic_episode("test")])
    except RuntimeError as error:
        assert "forbidden" in str(error)
    else:
        raise AssertionError("test split was not rejected")


def test_single_head_ablation_clears_that_head_in_all_history_slots():
    action = torch.ones(2, 39)
    ablated = ablate_head(action, "heading")
    for offset in (0, 13, 26):
        assert not bool(ablated[:, offset + 3:offset + 8].any())
        assert bool(ablated[:, offset:offset + 3].all())


def test_reactive_diagnostic_controls_have_known_bounded_head_effects():
    low = reactive_diagnostic_controls(torch.tensor([[0, 0, 0, 0]]))
    high = reactive_diagnostic_controls(torch.tensor([[2, 4, 2, 1]]))
    assert bool((low >= -1).all() and (low <= 1).all())
    assert bool((high >= -1).all() and (high <= 1).all())
    assert bool((high[0, :3] > low[0, :3]).all())
    assert high[0, 3] < low[0, 3]
