import torch

from envs.torch6dof import TorchWVRConfig, TorchWVRVecEnv


def make_env(max_steps=20):
    return TorchWVRVecEnv(
        TorchWVRConfig(
            num_envs=4, max_episode_steps=max_steps,
            initial_min_range_m=10000.0,
            initial_max_range_m=10000.1),
        device="cpu", seed=85, compile_dynamics=False)


def set_symmetric_head_on(env):
    positions = torch.tensor([
        [[-5000.0, 0.0, -6000.0], [5000.0, 0.0, -6000.0]]
    ]).repeat(env.config.num_envs, 1, 1)
    headings = torch.tensor([[0.0, torch.pi]]).repeat(
        env.config.num_envs, 1)
    speeds = torch.full((env.config.num_envs, 2), 250.0)
    env.aircraft_state = env.dynamics.level_state(
        positions.reshape(-1, 3), headings.reshape(-1), speeds.reshape(-1)
    ).reshape(env.config.num_envs, 2, env.dynamics.state_size)
    env.alive.fill_(True)
    env.remaining_missiles.fill_(env.config.missiles_per_aircraft)
    env.last_launch_step.fill_(-env.config.min_launch_interval_steps)
    env.episode_step.zero_()
    env.missile_state.zero_()
    env.previous_potential = env._posture_potential(env.aircraft_state)


def test_gpu_style_env_shapes_finite_zero_sum_and_masked_reset():
    env = make_env(max_steps=1)
    obs = env.observations()
    assert obs.shape == (4, 2, 39)
    control = torch.tensor([0.0, 0.0, 0.0, 0.7]).repeat(4, 2, 1)
    obs, reward, done, info = env.step(
        control, torch.zeros(4, 2, dtype=torch.bool))
    assert done.all()
    assert torch.all(env.episode_step == 0)
    assert obs.shape == (4, 2, 39)
    assert torch.isfinite(obs).all()
    assert torch.isfinite(reward).all()
    torch.testing.assert_close(reward.sum(-1), torch.zeros(4))
    assert torch.all(info["episode_length"] == 1)


def test_legal_launch_inventory_cooldown_and_static_slots():
    env = make_env()
    set_symmetric_head_on(env)
    assert env.legal_fire_mask().all()
    control = torch.tensor([0.0, 0.0, 0.0, 0.7]).repeat(4, 2, 1)
    _, reward, done, info = env.step(
        control, torch.ones(4, 2, dtype=torch.bool))
    assert not done.any()
    assert info["accepted_launches"].all()
    assert info["legal_fire_opportunities"].all()
    assert info["fire_requests"].all()
    assert not info["rejected_fire_requests"].any()
    assert torch.all(env.remaining_missiles == 1)
    assert torch.all(env.missile_state[:, :, 0, 12] == env.missile.FLYING)
    _, _, _, second = env.step(
        control, torch.ones(4, 2, dtype=torch.bool))
    assert not second["accepted_launches"].any()
    assert torch.all(env.remaining_missiles == 1)
    torch.testing.assert_close(reward.sum(-1), torch.zeros(4))


def test_initial_distribution_has_bounded_geometry_and_both_roles():
    env = TorchWVRVecEnv(
        TorchWVRConfig(num_envs=256), device="cpu", seed=127,
        compile_dynamics=False)
    _, _, _, distance, _, _, _ = env._geometry()
    assert float(distance.min()) >= env.config.initial_min_range_m - 1.0
    assert float(distance.max()) <= env.config.initial_max_range_m + 1.0
    delta_north = (
        env.aircraft_state[:, 1, 0] - env.aircraft_state[:, 0, 0])
    assert torch.any(delta_north > 0.0)
    assert torch.any(delta_north < 0.0)


def test_exhausted_weapons_terminate_unreachable_draw():
    env = make_env(max_steps=1000)
    set_symmetric_head_on(env)
    env.remaining_missiles.zero_()
    env.missile_state.zero_()
    control = torch.tensor([0.0, 0.0, 0.0, 0.7]).repeat(4, 2, 1)
    _, reward, done, info = env.step(
        control, torch.zeros(4, 2, dtype=torch.bool))
    assert done.all()
    assert info["weapons_exhausted"].all()
    assert torch.all(info["outcome_red"] == 0.0)
    torch.testing.assert_close(reward.sum(-1), torch.zeros(4))


def test_engagement_reward_preserves_shared_closing_signal():
    config = TorchWVRConfig(
        num_envs=2, reward_mode="engagement", reset_pool_size=8)
    env = TorchWVRVecEnv(config, device="cpu", compile_dynamics=False)
    previous = env.previous_potential.clone()
    # Move both aircraft closer without changing their headings. This tests
    # the reward contract independently of controller quality.
    midpoint = env.aircraft_state[..., :2].mean(dim=1, keepdim=True)
    offset = env.aircraft_state[..., :2] - midpoint
    env.aircraft_state[..., :2] = midpoint + 0.8 * offset
    current = env._posture_potential(env.aircraft_state)
    raw = config.potential_gamma * current - previous
    assert (raw.mean(dim=1) > 0.0).all()


def test_v4_observation_and_legacy_versions_are_replayable():
    env = make_env()
    set_symmetric_head_on(env)
    obs = env.observations()
    assert obs.shape[-1] == 39
    assert torch.isfinite(obs[..., 24:29]).all()
    assert (obs[..., 28] > 0.0).all()  # head-on geometry is closing
    assert torch.all(obs[..., 29:39] == 0.0)
    v3 = TorchWVRVecEnv(
        TorchWVRConfig(
            num_envs=2, observation_version="v3", reset_pool_size=8),
        device="cpu", compile_dynamics=False)
    assert v3.observations().shape == (2, 2, 29)
    legacy = TorchWVRVecEnv(
        TorchWVRConfig(
            num_envs=2, observation_version="v2",
            use_dynamic_launch_envelope=False, reset_pool_size=8),
        device="cpu", compile_dynamics=False)
    assert legacy.observations().shape == (2, 2, 24)


def test_v4_reports_incoming_and_own_flying_missiles():
    env = make_env()
    set_symmetric_head_on(env)
    request = torch.zeros(4, 2, dtype=torch.bool)
    request[:, 0] = True
    env._launch(request, env.legal_fire_mask())
    obs = env.observations()
    assert (obs[:, 0, 30] > 0.0).all()  # red owns a flying missile
    assert (obs[:, 1, 29] > 0.0).all()  # blue sees it incoming
    assert (obs[:, 1, 35] > 0.0).all()  # incoming missile is closing
    assert torch.isfinite(obs[..., 29:39]).all()


def test_terminal_pbrs_uses_zero_absorbing_potential():
    env = TorchWVRVecEnv(
        TorchWVRConfig(
            num_envs=2, max_episode_steps=1, reward_mode="engagement",
            reset_pool_size=8), device="cpu", compile_dynamics=False)
    previous = env.previous_potential.clone()
    control = torch.tensor([0.0, 0.0, 0.0, 0.7]).repeat(2, 2, 1)
    _, reward, done, info = env.step(
        control, torch.zeros(2, 2, dtype=torch.bool))
    assert done.all() and info["timeout"].all()
    expected = -previous - env.config.timeout_penalty
    torch.testing.assert_close(reward, expected)


def test_dynamic_launch_envelope_rejects_long_tail_chase():
    env = make_env()
    set_symmetric_head_on(env)
    assert env.legal_fire_mask().all()
    positions = torch.tensor([[
        [-5000.0, 0.0, -6000.0], [5000.0, 0.0, -6000.0]
    ]]).repeat(env.config.num_envs, 1, 1)
    # Red follows blue at equal speed: no closure and 10 km is outside the
    # conservative tail-chase envelope. Blue also fails the boresight gate.
    headings = torch.zeros(env.config.num_envs, 2)
    speeds = torch.full((env.config.num_envs, 2), 250.0)
    env.aircraft_state = env.dynamics.level_state(
        positions.reshape(-1, 3), headings.reshape(-1), speeds.reshape(-1)
    ).reshape(env.config.num_envs, 2, env.dynamics.state_size)
    assert not env.legal_fire_mask().any()


def test_attack_quality_is_bounded_and_prefers_a_valid_head_on_solution():
    env = make_env()
    set_symmetric_head_on(env)
    head_on = env._attack_quality()
    assert torch.all((head_on >= 0.0) & (head_on <= 1.0))
    env.aircraft_state[:, 0, 0] = -15000.0
    env.aircraft_state[:, 1, 0] = 15000.0
    far = env._attack_quality()
    assert torch.all(head_on > far)


def test_attack_quality_potential_is_opt_in():
    base = TorchWVRVecEnv(
        TorchWVRConfig(num_envs=2, reset_pool_size=8),
        device="cpu", compile_dynamics=False)
    shaped = TorchWVRVecEnv(
        TorchWVRConfig(
            num_envs=2, reset_pool_size=8, attack_quality_weight=0.5),
        device="cpu", seed=85, compile_dynamics=False)
    torch.testing.assert_close(base.aircraft_state, shaped.aircraft_state)
    quality = shaped._attack_quality()
    torch.testing.assert_close(
        shaped.previous_potential,
        base.previous_potential + 0.5 * quality)
