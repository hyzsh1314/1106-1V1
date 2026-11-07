#!/usr/bin/env python3
"""Evaluate V11.3 RACE-RHPPO with sustained reachability arbitration."""

from benchmarks.drvoi_v10 import SustainedReachabilityArbitrator
import benchmarks.evaluate_torch_wvr_coach_rhppo as engine


def main() -> None:
    engine.ACTION_TRANSFORM = SustainedReachabilityArbitrator(enter_steps=12)
    engine.EVALUATION_SUITE = "RACE-RHPPO-sustained-fixed-opponent"
    engine.PROTOCOL_VERSION = "RACE-RHPPO-V11.3-WVR-1"
    engine.main()


if __name__ == "__main__":
    main()
