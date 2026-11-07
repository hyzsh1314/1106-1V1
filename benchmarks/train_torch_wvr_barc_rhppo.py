#!/usr/bin/env python3
"""Train V10 BARC-RHPPO without modifying the frozen V9 experiment script.

BARC-RHPPO is COACH-RHPPO plus a kinematic option-admissibility shield and a
finite behaviour anchor on each option-conditioned low-level actor.  The V9
trainer is reused as an experiment engine; only the model, legal-option mask,
and checkpoint identity are replaced.
"""

from benchmarks.drvoi_v10 import (
    ReachabilityAnchoredOptionActorCritic,
    reachability_option_legal_mask,
)
import benchmarks.train_torch_wvr_coach_rhppo as engine


def main() -> None:
    engine.ALGORITHM = "barc_rhppo"
    engine.METHOD = (
        "behaviour_anchored_reachability_constrained_"
        "counterfactual_hierarchy")
    engine.AdaptiveOptionRecurrentActorCritic = (
        ReachabilityAnchoredOptionActorCritic)
    engine.macro_option_legal_mask = reachability_option_legal_mask
    engine.main()


if __name__ == "__main__":
    main()
