import torch
from types import SimpleNamespace

from benchmarks.drvoi_v12 import (
    FrozenReachabilityPredictor,
    FlatOptionBatch,
    INTERCEPT_EXPERT,
    MRATArbitrator,
    MRATRouter,
    PersistentReachabilityShield,
    ReachabilityEnsemble,
    option_termination_loss,
    reachability_distillation_loss,
    regularized_step_reward,
    variable_duration_gae,
    compute_flat_smdp_gae,
    conservative_router_logits,
)
from benchmarks.evaluate_torch_wvr_mrat_rhppo import (
    _routing_proposal,
    _segment_boundary,
    _update_gate_latch,
    _update_gate_sprt,
)


def test_reachability_ensemble_shapes_lcb_and_censored_loss():
    model = ReachabilityEnsemble(feature_dim=6, members=3, hidden_size=16)
    features = torch.randn(8, 6)
    expert = torch.arange(8).remainder(2)
    prediction = model(features, expert)
    assert prediction.member_score.shape == (3, 8)
    assert prediction.lower_confidence_bound(1.0).shape == (8,)
    loss = reachability_distillation_loss(
        prediction,
        target_score=torch.randn(8),
        target_probability=torch.rand(8),
        target_time=torch.rand(8) * 30.0,
        event_observed=torch.tensor(
            [True, False, True, False, True, True, False, False]),
        censor_horizon=30.0)
    assert torch.isfinite(loss["loss"])
    loss["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_frozen_predictor_returns_paired_expert_outputs():
    model = ReachabilityEnsemble(feature_dim=6, members=3, hidden_size=16)
    predictor = FrozenReachabilityPredictor(
        model, torch.zeros(6), torch.ones(6), lcb_z=1.0)
    output = predictor.predict_both(torch.randn(5, 6))
    assert output["score_mean"].shape == (5, 2)
    assert output["lcb"].shape == (5, 2)
    assert output["member_score"].shape == (3, 5, 2)


def test_router_exposes_policy_q_and_termination_distributions():
    router = MRATRouter(input_dim=10, hidden_size=16)
    features = torch.randn(5, 10)
    outputs = router(features)
    assert outputs["router_logits"].shape == (5, 2)
    assert outputs["q_values"].shape == (5, 2)
    assert router.state_value(features).shape == (5,)
    termination = router.termination_distribution(
        features, torch.tensor([0, 1, 0, 1, 0]), torch.rand(5))
    assert termination.probs.shape == (5,)


def test_conservative_router_masks_los_without_advantage_margin():
    logits, allowed = conservative_router_logits(
        torch.zeros(3, 2),
        torch.tensor([0.1, 0.0, -0.2]),
        torch.tensor([0.2, 0.04, 0.0]),
        margin=0.05,
    )
    assert allowed.tolist() == [True, False, True]
    assert logits[1, 1] < -1.0e8
    assert torch.softmax(logits, -1)[1].tolist() == [1.0, 0.0]


def test_intervention_and_switch_regularizers_apply_only_to_marked_steps():
    reward, intervention, transition = regularized_step_reward(
        torch.tensor([1.0, 1.0, 1.0]),
        torch.tensor([0, 1, 1]),
        torch.tensor([False, False, True]),
        intervention_weight=0.2,
        switch_weight=0.05,
    )
    torch.testing.assert_close(intervention, torch.tensor([0.0, 0.2, 0.2]))
    torch.testing.assert_close(transition, torch.tensor([0.0, 0.0, 0.05]))
    torch.testing.assert_close(reward, torch.tensor([1.0, 0.8, 0.75]))


def test_persistent_shield_requires_sustained_entry_and_release():
    shield = PersistentReachabilityShield(
        bad_threshold=0.0, safe_threshold=0.2,
        intercept_advantage=0.1, force_steps=3, release_steps=2)
    learned = torch.tensor([-0.2])
    intercept = torch.tensor([0.2])
    for index in range(2):
        step = shield.step(
            learned, intercept, torch.tensor([index == 0]))
        assert not bool(step.forced_intercept[0])
    step = shield.step(learned, intercept, torch.tensor([False]))
    assert bool(step.newly_forced[0]) and bool(step.forced_intercept[0])
    recovered = torch.tensor([0.3])
    step = shield.step(recovered, intercept, torch.tensor([False]))
    assert bool(step.forced_intercept[0])
    step = shield.step(recovered, intercept, torch.tensor([False]))
    assert bool(step.newly_released[0]) and not bool(step.forced_intercept[0])


def test_arbitrator_obeys_dwell_termination_and_forced_override():
    arbitrator = MRATArbitrator(minimum_dwell=2, maximum_dwell=5)
    proposed = torch.tensor([0])
    first = arbitrator.step(
        proposed, torch.tensor([False]), torch.tensor([False]),
        torch.tensor([True]))
    assert first.expert.item() == 0 and first.dwell.item() == 1
    no_early = arbitrator.step(
        torch.tensor([1]), torch.tensor([True]), torch.tensor([False]),
        torch.tensor([False]))
    assert no_early.expert.item() == 0
    switched = arbitrator.step(
        torch.tensor([1]), torch.tensor([True]), torch.tensor([False]),
        torch.tensor([False]))
    assert switched.expert.item() == 1 and bool(switched.switched[0])
    # Terminating and reselecting expert 1 creates a fresh option even though
    # the expert identity does not change.
    arbitrator.step(
        torch.tensor([1]), torch.tensor([False]), torch.tensor([False]),
        torch.tensor([False]))
    same_reselected = arbitrator.step(
        torch.tensor([1]), torch.tensor([True]), torch.tensor([False]),
        torch.tensor([False]))
    assert same_reselected.expert.item() == 1
    assert not bool(same_reselected.switched[0])
    assert same_reselected.dwell.item() == 1
    forced = arbitrator.step(
        torch.tensor([0]), torch.tensor([False]), torch.tensor([True]),
        torch.tensor([False]))
    assert forced.expert.item() == INTERCEPT_EXPERT


def test_evaluation_closes_same_expert_reselection_as_option_boundary():
    arbitration = SimpleNamespace(
        decision=torch.tensor([True, False, True]),
        switched=torch.tensor([False, True, True]),
    )
    reset = torch.tensor([False, False, True])
    assert _segment_boundary(arbitration, reset).tolist() == [True, True, False]


def test_shield_only_evaluation_defaults_to_learned_expert():
    logits = torch.tensor([[-2.0, 3.0], [4.0, -1.0]])
    assert _routing_proposal(logits, "router").tolist() == [1, 0]
    assert _routing_proposal(logits, "shield_only").tolist() == [0, 0]


def test_gate_latch_persists_until_episode_reset():
    latched = torch.tensor([False, True, True])
    allowed = torch.tensor([True, False, False])
    reset = torch.tensor([False, False, True])
    assert _update_gate_latch(latched, allowed, reset).tolist() == [
        True, True, False]


def test_gate_sprt_accumulates_and_resets_evidence():
    evidence = torch.tensor([0.0, 4.0])
    gate = torch.tensor([True, False])
    reset = torch.tensor([False, True])
    updated, trigger = _update_gate_sprt(
        evidence, gate, reset, evasive_rate=0.7, easy_rate=0.3,
        alpha=0.2, beta=0.2)
    assert updated[0] > 0.0
    assert updated[1] < 0.0
    assert not bool(trigger.any())


def test_shield_and_arbitrator_state_round_trip():
    shield = PersistentReachabilityShield(
        bad_threshold=0.0, safe_threshold=0.2,
        intercept_advantage=0.1, force_steps=1, release_steps=2)
    shield.step(torch.tensor([-1.0]), torch.tensor([1.0]), torch.tensor([True]))
    restored_shield = PersistentReachabilityShield(
        bad_threshold=0.0, safe_threshold=0.2,
        intercept_advantage=0.1, force_steps=1, release_steps=2)
    restored_shield.load_state_dict(shield.state_dict())
    assert bool(restored_shield.forced[0])

    arbitrator = MRATArbitrator(minimum_dwell=2, maximum_dwell=5)
    arbitrator.step(
        torch.tensor([1]), torch.tensor([False]), torch.tensor([False]),
        torch.tensor([True]))
    restored_arbitrator = MRATArbitrator(minimum_dwell=2, maximum_dwell=5)
    restored_arbitrator.load_state_dict(arbitrator.state_dict())
    assert restored_arbitrator.expert.tolist() == [1]
    assert restored_arbitrator.dwell.tolist() == [1]


def test_variable_duration_gae_reduces_to_one_step_and_masks_terminal():
    reward = torch.tensor([[1.0], [2.0]])
    duration = torch.ones(2, 1, dtype=torch.long)
    done = torch.tensor([[False], [True]])
    values = torch.tensor([[0.5], [1.0], [9.0]])
    advantage, returns = variable_duration_gae(
        reward, duration, done, values, gamma=0.5, gae_lambda=1.0)
    # Second segment is terminal: delta=2-1=1. First adds 0.5*next adv.
    torch.testing.assert_close(advantage[:, 0], torch.tensor([1.5, 1.0]))
    torch.testing.assert_close(returns[:, 0], torch.tensor([2.0, 2.0]))


def test_flat_smdp_gae_respects_interleaved_environment_sequences():
    batch = FlatOptionBatch(
        features=torch.zeros(4, 3),
        expert=torch.zeros(4, dtype=torch.long),
        old_log_prob=torch.zeros(4),
        value=torch.tensor([0.5, 2.0, 1.0, 3.0]),
        segment_return=torch.tensor([1.0, 4.0, 2.0, 6.0]),
        next_value=torch.tensor([1.0, 3.0, 0.0, 0.0]),
        duration=torch.ones(4, dtype=torch.long),
        done=torch.tensor([False, False, True, True]),
        policy_mask=torch.ones(4, dtype=torch.bool),
        env_id=torch.tensor([0, 1, 0, 1]),
    )
    advantage, returns = compute_flat_smdp_gae(
        batch, gamma=0.5, gae_lambda=1.0)
    # env 0: deltas 1.0 and 1.0 -> [1.5, 1.0]
    # env 1: deltas 3.5 and 3.0 -> [5.0, 3.0]
    torch.testing.assert_close(
        advantage, torch.tensor([1.5, 5.0, 1.0, 3.0]))
    torch.testing.assert_close(returns, advantage + batch.value)


def test_termination_loss_gradient_suppresses_positive_continuation_advantage():
    probability = torch.tensor([0.5, 0.5], requires_grad=True)
    loss = option_termination_loss(
        probability,
        active_q=torch.tensor([2.0, 0.0]),
        state_value=torch.tensor([1.0, 1.0]),
        eligible=torch.tensor([True, True]))
    loss.backward()
    # Gradient descent decreases p for positive continuation advantage and
    # increases p for negative continuation advantage.
    assert probability.grad[0] > 0.0
    assert probability.grad[1] < 0.0
