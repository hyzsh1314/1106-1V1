"""Physical-control policy protocol, adapters, and manifest registry."""

from .protocol import PhysicalPolicy
from .registry import CandidatePolicyRegistry
from .adapters import RecurrentLAGPPOAdapter

__all__ = ["PhysicalPolicy", "CandidatePolicyRegistry", "RecurrentLAGPPOAdapter"]
