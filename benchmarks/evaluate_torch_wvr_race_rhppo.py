#!/usr/bin/env python3
"""Evaluate V11 RACE-RHPPO reachability-arbitrated controller experts."""

from benchmarks.drvoi_v10 import reachability_shielded_actions
import benchmarks.evaluate_torch_wvr_coach_rhppo as engine


def _arbitrate(observation, learned_action, reset):
    action, _ = reachability_shielded_actions(observation, learned_action)
    return action


def main() -> None:
    # V11 deliberately consumes the frozen V9 policy.  The only intervention
    # is the auditable reachability arbitration at action execution time.
    engine.ACTION_TRANSFORM = _arbitrate
    engine.EVALUATION_SUITE = "RACE-RHPPO-fixed-opponent"
    engine.PROTOCOL_VERSION = "RACE-RHPPO-V11-WVR-1"
    engine.main()


if __name__ == "__main__":
    main()
