"""Auditable research infrastructure for the AC-COSR-WVR benchmark.

The package intentionally layers on top of the legacy LAG/JSBSim code.  It
does not change legacy training rewards into COSR evaluation semantics.
"""

from .schemas import CANONICAL_SPLITS, PolicyManifest, ScenarioSpec

__all__ = ["CANONICAL_SPLITS", "PolicyManifest", "ScenarioSpec"]
__version__ = "0.1.0"
