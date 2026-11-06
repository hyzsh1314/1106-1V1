"""Action conversion and validation at the legacy/COSR boundary."""

from __future__ import annotations

import numpy as np


ACTION_LOW = np.asarray([-1.0, -1.0, -1.0, 0.4], dtype=np.float32)
ACTION_HIGH = np.asarray([1.0, 1.0, 1.0, 0.9], dtype=np.float32)


def validate_physical_action(action, clip=False):
    value = np.asarray(action, dtype=np.float32).reshape(-1)
    if value.shape != (4,) or not np.all(np.isfinite(value)):
        raise ValueError("physical action must be four finite values")
    if clip:
        return np.clip(value, ACTION_LOW, ACTION_HIGH)
    if np.any(value < ACTION_LOW) or np.any(value > ACTION_HIGH):
        raise ValueError("physical action is outside COSR bounds")
    return value


class LegacyDiscreteActionMapper(object):
    """Map LAG's documented four-axis discrete action to physical control."""

    def __call__(self, discrete_action):
        action = np.asarray(discrete_action, dtype=np.int64).reshape(-1)
        if action.shape != (4,):
            raise ValueError("legacy action must have four axes")
        if np.any(action[:3] < 0) or np.any(action[:3] > 40) or action[3] < 0 or action[3] > 29:
            raise ValueError("legacy action index outside documented space")
        physical = np.asarray([
            action[0] / 20.0 - 1.0,
            action[1] / 20.0 - 1.0,
            action[2] / 20.0 - 1.0,
            action[3] / 58.0 + 0.4,
        ], dtype=np.float32)
        return validate_physical_action(physical, clip=True)

    def inverse_nearest(self, physical_action):
        action = validate_physical_action(physical_action, clip=True)
        values = np.asarray([
            round((action[0] + 1.0) * 20.0),
            round((action[1] + 1.0) * 20.0),
            round((action[2] + 1.0) * 20.0),
            round((action[3] - 0.4) * 58.0),
        ], dtype=np.int64)
        return np.clip(values, [0, 0, 0, 0], [40, 40, 40, 29])
