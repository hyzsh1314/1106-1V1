"""Complete-episode replay for V7 online response training."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch


@dataclass
class ReplayEpisode:
    episode_id: int
    opponent: int
    collection_seed: int
    data: dict[str, torch.Tensor]
    member_mask: torch.Tensor
    insertion_index: int


class EpisodeReplay:
    """Capacity-bounded replay with fixed per-episode ensemble masks."""

    def __init__(self, capacity: int = 2048, members: int = 5,
                 bootstrap_probability: float = 0.8,
                 recent_fraction: float = 0.5,
                 recent_window: int = 256, seed: int = 0) -> None:
        if capacity <= 0 or members <= 0:
            raise ValueError("capacity and members must be positive")
        if not 0.0 < bootstrap_probability <= 1.0:
            raise ValueError("bootstrap_probability must lie in (0,1]")
        if not 0.0 <= recent_fraction <= 1.0:
            raise ValueError("recent_fraction must lie in [0,1]")
        self.capacity = int(capacity)
        self.members = int(members)
        self.bootstrap_probability = float(bootstrap_probability)
        self.recent_fraction = float(recent_fraction)
        self.recent_window = int(recent_window)
        self._episodes: list[ReplayEpisode] = []
        self._episode_ids: set[int] = set()
        self._insertion_count = 0
        self._generator = torch.Generator(device="cpu").manual_seed(seed)

    def __len__(self) -> int:
        return len(self._episodes)

    def add(self, episode_id: int, opponent: int, collection_seed: int,
            data: dict[str, torch.Tensor],
            member_mask: torch.Tensor | None = None) -> ReplayEpisode:
        if episode_id in self._episode_ids:
            raise ValueError("episode_id already exists in replay")
        lengths = {value.shape[0] for value in data.values()}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) <= 0:
            raise ValueError("all episode tensors must share a positive time length")
        if member_mask is None:
            member_mask = torch.rand(
                self.members, generator=self._generator) < self.bootstrap_probability
            if not bool(member_mask.any()):
                member_mask[torch.randint(
                    self.members, (1,), generator=self._generator)] = True
        member_mask = member_mask.bool().cpu()
        if member_mask.shape != (self.members,):
            raise ValueError("member_mask has wrong shape")
        episode = ReplayEpisode(
            episode_id=int(episode_id), opponent=int(opponent),
            collection_seed=int(collection_seed),
            data={key: value.detach().cpu().clone() for key, value in data.items()},
            member_mask=member_mask.clone(), insertion_index=self._insertion_count)
        self._insertion_count += 1
        self._episodes.append(episode)
        self._episode_ids.add(episode.episode_id)
        if len(self._episodes) > self.capacity:
            removed = self._episodes.pop(0)
            self._episode_ids.remove(removed.episode_id)
        return episode

    def _eligible(self, member: int) -> list[ReplayEpisode]:
        if not 0 <= member < self.members:
            raise ValueError("member index is out of range")
        return [episode for episode in self._episodes
                if bool(episode.member_mask[member])]

    def sample(self, member: int, episode_count: int) -> list[ReplayEpisode]:
        """Sample recent/history episodes with approximate opponent balance."""
        if episode_count <= 0:
            raise ValueError("episode_count must be positive")
        eligible = self._eligible(member)
        if not eligible:
            return []
        recent = eligible[-min(self.recent_window, len(eligible)):]
        recent_ids = {episode.episode_id for episode in recent}
        historical = [episode for episode in eligible
                      if episode.episode_id not in recent_ids]
        recent_count = min(
            int(round(episode_count * self.recent_fraction)), len(recent))
        history_count = min(episode_count - recent_count, len(historical))
        if history_count < episode_count - recent_count:
            recent_count = min(episode_count - history_count, len(recent))

        def stratified(pool: list[ReplayEpisode], count: int):
            if count <= 0 or not pool:
                return []
            groups: dict[int, list[ReplayEpisode]] = {}
            for episode in pool:
                groups.setdefault(episode.opponent, []).append(episode)
            selected = []
            opponents = sorted(groups)
            cursor = 0
            while len(selected) < count and any(groups.values()):
                opponent = opponents[cursor % len(opponents)]
                group = groups[opponent]
                if group:
                    index = int(torch.randint(
                        len(group), (1,), generator=self._generator))
                    selected.append(group.pop(index))
                cursor += 1
            return selected

        return stratified(recent.copy(), recent_count) + stratified(
            historical.copy(), history_count)

    @staticmethod
    def flatten(episodes: list[ReplayEpisode],
                device: torch.device | str) -> dict[str, torch.Tensor]:
        if not episodes:
            raise ValueError("cannot flatten an empty episode sample")
        keys = episodes[0].data.keys()
        return {key: torch.cat([episode.data[key] for episode in episodes]).to(device)
                for key in keys}

    def state_dict(self) -> dict[str, object]:
        return {
            "capacity": self.capacity, "members": self.members,
            "bootstrap_probability": self.bootstrap_probability,
            "recent_fraction": self.recent_fraction,
            "recent_window": self.recent_window,
            "episodes": copy.deepcopy(self._episodes),
            "insertion_count": self._insertion_count,
            "generator_state": self._generator.get_state().clone(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        for key in ("capacity", "members", "bootstrap_probability",
                    "recent_fraction", "recent_window"):
            if getattr(self, key) != state[key]:
                raise ValueError(f"replay configuration mismatch: {key}")
        self._episodes = copy.deepcopy(state["episodes"])
        self._episode_ids = {episode.episode_id for episode in self._episodes}
        self._insertion_count = int(state["insertion_count"])
        self._generator.set_state(state["generator_state"].clone())
