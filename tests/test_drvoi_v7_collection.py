import torch

from benchmarks.collect_drvoi_v7_response_data import (
    SPLIT_COUNTS,
    build_schedule,
    reactive_opponent_action,
    uniform_legal_actions,
)


def test_cover_schedule_is_exactly_balanced_with_isolated_splits():
    schedules = build_schedule("cover", 6185)
    for entries in schedules.values():
        assert len(entries) == sum(SPLIT_COUNTS.values())
        for split, count in SPLIT_COUNTS.items():
            selected = [value for value in entries if value["split"] == split]
            assert len(selected) == count
            assert sum(value["mode"] == "uniform" for value in selected) == count // 2


def test_uniform_coverage_never_requests_illegal_fire():
    obs = torch.zeros(512, 39)
    obs[::2, 23] = 1.0
    action = uniform_legal_actions(obs, torch.Generator().manual_seed(4))
    assert tuple(action.shape) == (512, 4)
    assert not bool(action[1::2, 3].any())
    assert int(action[:, 0].min()) == 0 and int(action[:, 0].max()) == 2
    assert int(action[:, 1].min()) == 0 and int(action[:, 1].max()) == 4


def test_reactive_positive_control_has_known_direction_and_legal_fire():
    obs = torch.zeros(2, 39)
    obs[0, 23] = 1.0
    delayed = torch.tensor([[0, 1, 2, 0], [2, 4, 0, 1]])
    reaction = reactive_opponent_action(obs, delayed)
    assert torch.equal(reaction[:, :3], torch.tensor([[2, 3, 0], [0, 0, 2]]))
    assert torch.equal(reaction[:, 3], torch.tensor([1, 0]))
