"""Experimental batched Torch flight-dynamics backend.

JSBSim remains the validated default backend.  The classes exported here are
opt-in until their fidelity gates pass against the project's F-16 model.
"""

from .f16_dynamics import (
    CompiledTorchF16Runner, TorchF16BatchDynamics, TorchF16Parameters,
)
from .missile import TorchWVRMissileBatch, TorchWVRMissileParameters
from .wvr_env import TorchWVRConfig, TorchWVRVecEnv

__all__ = [
    "CompiledTorchF16Runner", "TorchF16BatchDynamics", "TorchF16Parameters",
    "TorchWVRMissileBatch", "TorchWVRMissileParameters",
    "TorchWVRConfig", "TorchWVRVecEnv",
]
