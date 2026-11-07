"""Observation-driven tactical macro-options for the V8 value audit.

The low-level command semantics follow ``lowlevel_controls`` exactly:
altitude index 0 commands climb, heading indices 0/4 command maximum
left/right relative turn, and speed index 0 commands acceleration.
"""

from __future__ import annotations

import torch


OPTION_NAMES = (
    "pure_pursuit",
    "lead_intercept",
    "hard_left_break",
    "hard_right_break",
    "energy_climb",
    "decel_cut_in",
    "launch_and_crank",
)
NUM_MACRO_OPTIONS = len(OPTION_NAMES)


def macro_option_legal_mask(observation: torch.Tensor) -> torch.Tensor:
    if observation.ndim != 2 or observation.shape[-1] != 39:
        raise ValueError("observation must have [batch,39] shape")
    legal = torch.ones(
        observation.shape[0], NUM_MACRO_OPTIONS, dtype=torch.bool,
        device=observation.device)
    legal[:, -1] = observation[:, 23] >= 0.5
    return legal


def _altitude_tracking(observation: torch.Tensor) -> torch.Tensor:
    elevation = observation[:, 25]
    command = torch.where(
        elevation > 0.04, torch.zeros_like(elevation, dtype=torch.long),
        torch.where(
            elevation < -0.04,
            torch.full_like(elevation, 2, dtype=torch.long),
            torch.ones_like(elevation, dtype=torch.long)))
    altitude = observation[:, 0]
    command = torch.where(
        altitude <= 0.6, torch.zeros_like(command),
        torch.where(
            (altitude <= 0.8) & (command == 2),
            torch.ones_like(command), command))
    return command


def _quantized_intercept_heading(observation: torch.Tensor) -> torch.Tensor:
    steering = torch.pi * observation[:, 24] + 3.0 * observation[:, 26]
    return torch.where(
        steering < -0.20, torch.zeros_like(steering, dtype=torch.long),
        torch.where(
            steering < -0.06, torch.ones_like(steering, dtype=torch.long),
            torch.where(
                steering <= 0.06,
                torch.full_like(steering, 2, dtype=torch.long),
                torch.where(
                    steering <= 0.20,
                    torch.full_like(steering, 3, dtype=torch.long),
                    torch.full_like(steering, 4, dtype=torch.long)))))


def macro_option_actions(observation: torch.Tensor, option: torch.Tensor,
                         elapsed: int = 0) -> torch.Tensor:
    """Return one closed-loop primitive command for each active option."""
    if observation.ndim != 2 or observation.shape[-1] != 39:
        raise ValueError("observation must have [batch,39] shape")
    if option.shape != observation.shape[:1]:
        raise ValueError("option must have one id per observation")
    option = option.to(device=observation.device, dtype=torch.long)
    if bool(((option < 0) | (option >= NUM_MACRO_OPTIONS)).any()):
        raise ValueError("option id is out of range")

    batch = observation.shape[0]
    action = torch.zeros(batch, 4, dtype=torch.long, device=observation.device)
    action[:, 0] = _altitude_tracking(observation)
    action[:, 1] = _quantized_intercept_heading(observation)
    action[:, 2] = 0

    side = observation[:, 14]
    pursuit = torch.where(
        side >= 0.0, torch.full_like(option, 4), torch.zeros_like(option))
    action[:, 1] = torch.where(option == 0, pursuit, action[:, 1])
    action[:, 1] = torch.where(option == 2, torch.zeros_like(option), action[:, 1])
    action[:, 1] = torch.where(option == 3, torch.full_like(option, 4), action[:, 1])

    # Energy climb uses the platform's actual altitude-command index 0.
    action[:, 0] = torch.where(option == 4, torch.zeros_like(option), action[:, 0])
    action[:, 1] = torch.where(option == 4, pursuit, action[:, 1])

    action[:, 2] = torch.where(option == 5, torch.full_like(option, 2), action[:, 2])

    # The generic missile is autonomous after launch.  Crank therefore means
    # a moderate LOS-offset turn and energy preservation, not radar support.
    crank = torch.where(
        side >= 0.0, torch.full_like(option, 3), torch.ones_like(option))
    action[:, 1] = torch.where(option == 6, crank, action[:, 1])
    action[:, 2] = torch.where(option == 6, torch.ones_like(option), action[:, 2])
    if elapsed == 0:
        action[:, 3] = ((option == 6) & (observation[:, 23] >= 0.5)).long()
    return action
