import torch

from benchmarks.drvoi.response import ContinuousResponseEnsemble
from benchmarks.drvoi_v7_1.history import (
    NEUTRAL_ACTION,
    VectorHistoryBuffer,
    assemble_episode_history,
)


def actions(count):
    step = torch.arange(count)
    return torch.stack((step % 3, step % 5, step % 3, step % 2), -1)


def decode_head(one_hot, start, size):
    return one_hot[..., start:start + size].argmax(-1)


def test_episode_history_padding_and_correct_delay_alignment():
    observation = torch.arange(7 * 39).reshape(7, 39).float()
    action = actions(7)
    history = assemble_episode_history(observation, action)
    assert history.observations.shape == (7, 3, 39)
    assert torch.equal(history.valid[0], torch.tensor([False, False, True]))
    assert torch.equal(history.valid[1], torch.tensor([False, True, True]))
    assert torch.equal(history.observations[6, :, 0], observation[4:7, 0])
    assert torch.equal(decode_head(history.actions[6], 0, 3), action[4:7, 0])
    assert torch.equal(
        decode_head(history.wrong_delay_actions[6], 0, 3), action[1:4, 0])


def test_vector_history_reset_clears_only_selected_environment():
    buffer = VectorHistoryBuffer(2)
    buffer.append(torch.ones(2, 39), actions(2))
    second = buffer.append(
        2.0 * torch.ones(2, 39), actions(2).flip(0),
        reset=torch.tensor([True, False]))
    assert torch.equal(second.valid[0], torch.tensor([False, False, True]))
    assert torch.equal(second.valid[1], torch.tensor([False, True, True]))
    assert bool((second.observations[0, :2] == 0).all())
    assert bool((second.observations[1, -2] == 1).all())


def test_v7_1_response_accepts_three_step_action_history():
    model = ContinuousResponseEnsemble(
        input_dim=120, response_dim=12, action_dim=39,
        members=2, hidden_size=16)
    means, variances = model(torch.randn(4, 120), torch.zeros(4, 39))
    assert means.shape == variances.shape == (2, 4, 12)
    assert model.action_dim == 39
