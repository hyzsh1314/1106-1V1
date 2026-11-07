#!/usr/bin/env python3
"""Evaluate V11.2 RACE-RHPPO with episode-latched expert arbitration."""

from benchmarks.drvoi_v10 import EpisodeLatchedReachabilityArbitrator
import benchmarks.evaluate_torch_wvr_coach_rhppo as engine


def main() -> None:
    engine.ACTION_TRANSFORM = EpisodeLatchedReachabilityArbitrator()
    engine.EVALUATION_SUITE = "RACE-RHPPO-episode-latched-fixed-opponent"
    engine.PROTOCOL_VERSION = "RACE-RHPPO-V11.2-WVR-1"
    engine.main()


if __name__ == "__main__":
    main()
