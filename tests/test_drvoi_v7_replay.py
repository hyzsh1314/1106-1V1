import torch

from benchmarks.drvoi.replay import EpisodeReplay


def _episode(value, length=3):
    return {
        "inputs": torch.full((length, 2), float(value)),
        "actions": torch.full((length, 1), float(value)),
        "targets": torch.full((length, 2), float(value)),
    }


def test_episode_replay_uses_fixed_member_masks_and_capacity():
    replay = EpisodeReplay(capacity=3, members=2, seed=5)
    replay.add(1, 0, 10, _episode(1), torch.tensor([True, False]))
    replay.add(2, 1, 10, _episode(2), torch.tensor([True, True]))
    replay.add(3, 2, 10, _episode(3), torch.tensor([False, True]))
    replay.add(4, 0, 10, _episode(4), torch.tensor([True, True]))
    assert len(replay) == 3
    assert all(episode.episode_id != 1 for episode in replay.sample(0, 10))
    assert all(bool(episode.member_mask[0]) for episode in replay.sample(0, 10))


def test_episode_replay_restore_preserves_masks_and_rng():
    first = EpisodeReplay(
        capacity=8, members=3, recent_fraction=0.5, recent_window=2, seed=9)
    for episode_id in range(6):
        first.add(episode_id, episode_id % 3, 17, _episode(episode_id))
    restored = EpisodeReplay(
        capacity=8, members=3, recent_fraction=0.5, recent_window=2, seed=99)
    restored.load_state_dict(first.state_dict())
    expected = [episode.episode_id for episode in first.sample(0, 4)]
    actual = [episode.episode_id for episode in restored.sample(0, 4)]
    assert expected == actual


def test_episode_replay_flattens_complete_episode_samples():
    replay = EpisodeReplay(capacity=4, members=1, seed=1)
    one = replay.add(1, 0, 1, _episode(1, 2), torch.tensor([True]))
    two = replay.add(2, 1, 1, _episode(2, 4), torch.tensor([True]))
    flat = replay.flatten([one, two], "cpu")
    assert flat["inputs"].shape == (6, 2)
    assert flat["targets"].shape == (6, 2)
