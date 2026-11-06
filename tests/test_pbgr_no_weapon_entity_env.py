import numpy as np

from algorithms.pbgr.entities import ENTITY_FEATURE_NAMES, FlatEntityCodec
from envs.JSBSim.envs import MultipleCombatEnv


def test_no_weapon_pbgr_environment_exposes_single_pass_entity_boxes():
    env = MultipleCombatEnv("2v2/NoWeapon/PBGR_R3_Entity_NoWeapon")
    try:
        obs, central = env.reset()
        codec = FlatEntityCodec(4, len(ENTITY_FEATURE_NAMES))
        assert obs.shape == (4, codec.flat_dim)
        assert central.shape == (4, 4 * codec.flat_dim)
        decoded = codec.unpack_array(obs)
        assert np.all(decoded["entity_mask"] == 1.0)
        assert np.all(decoded["slot_reset"] == 1.0)
        actions = np.zeros((4, 4), dtype=np.int64)
        next_obs, next_central, rewards, dones, _info = env.step(actions)
        assert next_obs.shape == obs.shape
        assert next_central.shape == central.shape
        assert rewards.shape == (4, 1)
        assert dones.shape == (4, 1)
        assert np.isfinite(next_obs).all()
        assert np.all(codec.unpack_array(next_obs)["slot_reset"] == 0.0)
    finally:
        env.close()


def test_no_weapon_pbgr_environment_emits_unambiguous_team_terminal_flags():
    env = MultipleCombatEnv("2v2/NoWeapon/PBGR_R3_Entity_NoWeapon")
    try:
        env.reset()
        env.max_steps = 1
        actions = np.zeros((4, 4), dtype=np.int64)
        _obs, _central, _rewards, dones, info = env.step(actions)
        assert np.all(dones)
        assert info["terminated"] is True
        assert info["truncated"] is False
        assert info["team_outcome"]["label"] == "draw"
        assert info["termination_reason"] == "timeout_draw"
    finally:
        env.close()

