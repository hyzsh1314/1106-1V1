"""cProfile entry point for one WVR environment hot path."""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats

from benchmarks.benchmark_environment import (
    SCENARIOS,
    make_environment,
    neutral_action,
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="r6")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--weapon-mode", choices=("hold", "fire"), default="hold")
    parser.add_argument("--sort", default="cumulative")
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args(argv)
    env = make_environment(SCENARIOS[args.scenario], seed=20261102)
    try:
        action = neutral_action(env, weapon_mode=args.weapon_mode)
        env.reset()
        profile = cProfile.Profile()
        profile.enable()
        for _ in range(args.steps):
            result = env.step(action)
            if result[3].all():
                env.reset()
        profile.disable()
        stream = io.StringIO()
        pstats.Stats(profile, stream=stream).sort_stats(args.sort).print_stats(
            args.limit
        )
        print(stream.getvalue())
    finally:
        env.close()


if __name__ == "__main__":
    main()

