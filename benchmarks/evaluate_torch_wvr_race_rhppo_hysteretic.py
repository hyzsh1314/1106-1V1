#!/usr/bin/env python3
"""Evaluate V11.1 RACE-RHPPO with hysteretic expert arbitration."""

from benchmarks.drvoi_v10 import HystereticReachabilityArbitrator
import benchmarks.evaluate_torch_wvr_coach_rhppo as engine


def main() -> None:
    engine.ACTION_TRANSFORM = HystereticReachabilityArbitrator()
    engine.EVALUATION_SUITE = "RACE-RHPPO-hysteretic-fixed-opponent"
    engine.PROTOCOL_VERSION = "RACE-RHPPO-V11.1-WVR-1"
    engine.main()


if __name__ == "__main__":
    main()
