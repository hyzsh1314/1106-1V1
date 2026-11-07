import torch

from benchmarks.drvoi_v7_1.macro_options import (
    NUM_MACRO_OPTIONS,
    macro_option_actions,
    macro_option_legal_mask,
)


def observations(batch=7):
    obs = torch.zeros(batch, 39)
    obs[:, 0] = 1.0
    obs[:, 14] = torch.tensor([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0])
    return obs


def test_macro_options_follow_platform_action_semantics():
    obs = observations()
    obs[-1, 23] = 1.0
    option = torch.arange(NUM_MACRO_OPTIONS)
    action = macro_option_actions(obs, option, elapsed=0)
    assert action.shape == (NUM_MACRO_OPTIONS, 4)
    assert action[2, 1] == 0 and action[3, 1] == 4
    assert action[4, 0] == 0  # platform index 0 is climb
    assert action[5, 2] == 2  # decelerate
    assert action[-1, 3] == 1


def test_launch_option_is_masked_and_fires_only_on_first_step():
    obs = observations()
    legal = macro_option_legal_mask(obs)
    assert legal.shape == (7, 7)
    assert not bool(legal[:, -1].any())
    option = torch.full((7,), 6)
    assert not bool(macro_option_actions(obs, option, elapsed=0)[:, 3].any())
    obs[:, 23] = 1.0
    assert bool(macro_option_actions(obs, option, elapsed=0)[:, 3].all())
    assert not bool(macro_option_actions(obs, option, elapsed=1)[:, 3].any())


def test_altitude_safety_overrides_descending_tracking_command():
    obs = observations(batch=7)
    obs[:, 0] = 0.5
    obs[:, 25] = -0.5
    action = macro_option_actions(obs, torch.arange(7), elapsed=2)
    assert bool((action[:, 0] == 0).all())
