from benchmarks.drvoi.gates import evaluate_o0_gate, evaluate_q_ready


def _q_metrics():
    metrics = {
        "response_sensitivity_control": True, "finite": True,
        "td_p50": 0.1, "td_p90": 0.2, "td_p99": 0.5,
    }
    for fold in ("a", "b"):
        metrics.update({
            f"fold_{fold}_episodes": 200,
            f"fold_{fold}_tuples": 20000,
            f"fold_{fold}_nonzero_return_episodes": 100,
            f"fold_{fold}_heldout_return_spearman": 0.3,
            f"fold_{fold}_heldout_return_spearman_ci_low": 0.1,
        })
    return metrics


def test_q_ready_requires_every_condition_not_update_count():
    config = {
        "min_episodes_per_fold": 200, "min_tuples_per_fold": 20000,
        "min_nonzero_return_episodes_per_fold": 100,
        "min_return_spearman": 0.2,
    }
    assert evaluate_q_ready(_q_metrics(), config)["ready"]
    failed = _q_metrics()
    failed["fold_b_tuples"] = 19999
    report = evaluate_q_ready(failed, config)
    assert not report["ready"] and "fold_b_tuples" in report["reasons"]


def test_o0_gate_rejects_zero_voi_and_failed_restore():
    config = {
        "min_valid_voi_contexts": 10000,
        "min_positive_voi_nonzero_fraction": 0.01,
        "min_candidate_argmax_diversity_fraction": 0.01,
        "mc_s20_s100_spearman_min": 0.9,
        "mc_nonzero_sign_agreement_min": 0.95,
    }
    metrics = {
        "valid_voi_contexts": 10000,
        "positive_voi_nonzero_fraction": 0.0,
        "candidate_argmax_diversity_fraction": 0.0,
        "response_shuffle_pass": False,
        "mc_s20_s100_spearman": 1.0,
        "mc_nonzero_sign_agreement": 1.0,
        "checkpoint_restore_pass": False,
        "finite": True,
    }
    report = evaluate_o0_gate(metrics, config)
    assert not report["pass"]
    assert "positive_voi_nonzero_fraction" in report["reasons"]
    assert "checkpoint_restore" in report["reasons"]
