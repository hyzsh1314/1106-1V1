"""Candidate policy registry; intentionally separate from opponents."""

from __future__ import annotations

from typing import Dict, Iterable

from ..schemas import PolicyManifest


class CandidatePolicyRegistry(object):
    def __init__(self):
        self._items = {}

    def register(self, manifest):
        if not isinstance(manifest, PolicyManifest):
            raise TypeError("candidate registry accepts PolicyManifest")
        if manifest.policy_id in self._items:
            raise ValueError("duplicate candidate policy id: {0}".format(manifest.policy_id))
        if manifest.action_interface not in ("physical_continuous_v1", "legacy_discrete_mapped_v1"):
            raise ValueError("unsupported action interface: {0}".format(manifest.action_interface))
        self._items[manifest.policy_id] = manifest
        return manifest

    def get(self, policy_id):
        return self._items[policy_id]

    def manifests(self):
        return [self._items[key] for key in sorted(self._items)]

    def ids(self):
        return sorted(self._items)
