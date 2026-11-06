import torch

from benchmarks.validate_drvoi_v7_1_p1q_development import build_q_data


def test_q_development_keeps_partial_terminal_delta_without_bootstrap():
    length = 10
    episode = {
        "split": "train", "episode_id": 7, "collection_seed": 10185,
        "observation": torch.zeros(length, 39),
        "belief": torch.zeros(length, 8),
        "action": torch.tensor([[1, 2, 1, 0]]).expand(length, -1),
        "reward": torch.arange(length).float(),
    }
    data = build_q_data([episode], "train", delta=3, gamma=0.5)
    assert data.context.shape[0] == 3
    assert data.done.tolist() == [False, False, True]
    assert data.reward[-1] == 9.0
    assert bool((data.next_context[-1] == 0).all())


def test_vectorized_q_returns_match_scalar_discount_definition():
    torch.manual_seed(12)
    length, gamma = 17, 0.93
    reward = torch.randn(length)
    episode = {
        "split": "train", "episode_id": 9, "collection_seed": 10185,
        "observation": torch.randn(length, 39),
        "belief": torch.randn(length, 8),
        "action": torch.stack((
            torch.randint(0, 3, (length,)),
            torch.randint(0, 5, (length,)),
            torch.randint(0, 3, (length,)),
            torch.randint(0, 2, (length,))), -1),
        "reward": reward,
    }
    data = build_q_data([episode], "train", delta=3, gamma=gamma)
    decisions = list(range(3, length, 3))
    expected_mc = torch.stack([
        sum(reward[index] * gamma ** (index - decision)
            for index in range(decision, length))
        for decision in decisions])
    expected_window = torch.stack([
        sum(reward[index] * gamma ** (index - decision)
            for index in range(decision, min(decision + 3, length)))
        for decision in decisions])
    assert torch.allclose(data.monte_carlo_return, expected_mc, atol=2e-6)
    assert torch.allclose(data.reward, expected_window, atol=2e-6)
