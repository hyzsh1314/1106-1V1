"""Machine-readable V7 authorization gates."""

from __future__ import annotations

import math


def evaluate_q_ready(metrics: dict[str, object], config: dict[str, object]):
    reasons = []
    for fold in ("a", "b"):
        if int(metrics[f"fold_{fold}_episodes"]) < int(config["min_episodes_per_fold"]):
            reasons.append(f"fold_{fold}_episodes")
        if int(metrics[f"fold_{fold}_tuples"]) < int(config["min_tuples_per_fold"]):
            reasons.append(f"fold_{fold}_tuples")
        if int(metrics[f"fold_{fold}_nonzero_return_episodes"]) < int(
                config["min_nonzero_return_episodes_per_fold"]):
            reasons.append(f"fold_{fold}_nonzero_return_episodes")
        rho = float(metrics[f"fold_{fold}_heldout_return_spearman"])
        lower = float(metrics[f"fold_{fold}_heldout_return_spearman_ci_low"])
        if rho < float(config["min_return_spearman"]) or lower <= 0.0:
            reasons.append(f"fold_{fold}_return_semantics")
    if not bool(metrics.get("response_sensitivity_control", False)):
        reasons.append("response_sensitivity_control")
    finite = bool(metrics.get("finite", False)) and all(
        math.isfinite(float(metrics[key])) for key in (
            "td_p50", "td_p90", "td_p99"))
    if not finite:
        reasons.append("numerical_health")
    return {"ready": not reasons, "reasons": reasons, "metrics": metrics}


def evaluate_o0_gate(metrics: dict[str, object], config: dict[str, object]):
    reasons = []
    if int(metrics["valid_voi_contexts"]) < int(config["min_valid_voi_contexts"]):
        reasons.append("valid_voi_contexts")
    if float(metrics["positive_voi_nonzero_fraction"]) < float(
            config["min_positive_voi_nonzero_fraction"]):
        reasons.append("positive_voi_nonzero_fraction")
    if float(metrics["candidate_argmax_diversity_fraction"]) < float(
            config["min_candidate_argmax_diversity_fraction"]):
        reasons.append("candidate_argmax_diversity_fraction")
    if not bool(metrics.get("response_shuffle_pass", False)):
        reasons.append("response_shuffle")
    if float(metrics.get("mc_s20_s100_spearman", float("-inf"))) < float(
            config["mc_s20_s100_spearman_min"]):
        reasons.append("mc_rank_stability")
    if float(metrics.get("mc_nonzero_sign_agreement", 0.0)) < float(
            config["mc_nonzero_sign_agreement_min"]):
        reasons.append("mc_sign_stability")
    if not bool(metrics.get("checkpoint_restore_pass", False)):
        reasons.append("checkpoint_restore")
    if not bool(metrics.get("finite", False)):
        reasons.append("numerical_health")
    return {"pass": not reasons, "reasons": reasons, "metrics": metrics}
