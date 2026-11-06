#!/usr/bin/env python3
"""Issue machine-readable V7 stage authorization from completed gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(response: dict, q_gate: dict, o0_gate: dict) -> dict:
    stages = {
        "P1R_P1C": bool(response.get("all_gates_pass", False)),
        "P1Q": bool(q_gate.get("ready", False)),
        "O0": bool(o0_gate.get("pass", False)),
    }
    passed = all(stages.values())
    return {
        "protocol_version": "DRVOI-RHPPO-V7-WVR-1",
        "stages": stages,
        "short_screen_100k_to_500k_authorized": passed,
        "v7_long_training_authorized": False,
        "long_training_reason": (
            "short_screen_not_completed" if passed
            else "one_or_more_prerequisite_gates_failed"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--q-gate", type=Path, required=True)
    parser.add_argument("--o0-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(
        json.loads(args.response.read_text(encoding="utf-8")),
        json.loads(args.q_gate.read_text(encoding="utf-8")),
        json.loads(args.o0_gate.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
