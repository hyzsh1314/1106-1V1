import torch

from benchmarks.drvoi import multi_discrete_to_one_hot
from benchmarks.validate_drvoi_wvr_q_v7 import (
    build_q_data,
    rankdata,
    standardize_context,
)


def q_episode(length=9):
    observation = torch.zeros(length, 39)
    observation[:, 9] = torch.arange(length).float()
    belief = torch.arange(length).float()[:, None].expand(-1, 8).clone()
    action = torch.stack((
        torch.arange(length) % 3,
        torch.arange(length) % 5,
        torch.arange(length) % 3,
        torch.zeros(length, dtype=torch.long),
    ), -1)
    done = torch.zeros(length, dtype=torch.bool)
    done[-1] = True
    return {
        "observation": observation,
        "belief": belief,
        "action": action,
        "reward": torch.arange(length).float(),
        "done": done,
        "episode_id": 8,
        "split": "train",
    }


def test_q_data_uses_delta_stride_and_correct_reward_window():
    data = build_q_data([q_episode()], "train", delta=3, gamma=0.5)
    assert data.context.shape[0] == 2
    assert torch.allclose(
        data.reward, torch.tensor([3.0 + 0.5 * 4.0 + 0.25 * 5.0,
                                   6.0 + 0.5 * 7.0 + 0.25 * 8.0]))
    assert torch.equal(data.done, torch.tensor([False, True]))
    assert torch.equal(
        data.action[0],
        multi_discrete_to_one_hot(torch.tensor([[0, 3, 0, 0]]))[0])
    assert bool((data.next_context[1] == 0).all())


def test_context_standardization_keeps_terminal_next_context_zero():
    data = build_q_data([q_episode()], "train", delta=3, gamma=0.99)
    standardize_context(data)
    assert bool((data.next_context[data.done] == 0).all())


def test_rankdata_uses_average_ranks_for_ties():
    assert torch.equal(
        rankdata(torch.tensor([2.0, 1.0, 2.0])),
        torch.tensor([1.5, 0.0, 1.5]))
