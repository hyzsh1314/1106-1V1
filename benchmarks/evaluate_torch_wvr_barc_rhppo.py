#!/usr/bin/env python3
"""Evaluate a frozen V10 BARC-RHPPO policy against a fixed opponent."""

from benchmarks.drvoi_v10 import (
    ReachabilityAnchoredOptionActorCritic,
    reachability_option_legal_mask,
)
import benchmarks.evaluate_torch_wvr_coach_rhppo as engine


def main() -> None:
    engine.MODEL_CLASS = ReachabilityAnchoredOptionActorCritic
    engine.OPTION_LEGAL_MASK = reachability_option_legal_mask
    engine.EXPECTED_ALGORITHM = "barc_rhppo"
    engine.EVALUATION_SUITE = "BARC-RHPPO-fixed-opponent"
    engine.PROTOCOL_VERSION = "BARC-RHPPO-V10-WVR-1"
    engine.main()


if __name__ == "__main__":
    main()
