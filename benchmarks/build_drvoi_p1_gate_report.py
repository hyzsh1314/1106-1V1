#!/usr/bin/env python3
"""Aggregate DRVOI P1 gates and execute deterministic G7 invariants."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import torch

from benchmarks.drvoi import CrossFittedQTransitionAssembler, cross_fitted_voi
from benchmarks.drvoi.q_model import ActionEmbeddingQ, double_q_target


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_g7():
    assembler = CrossFittedQTransitionAssembler()
    assert assembler.fold_for_episode(10) == 0
    assert assembler.fold_for_episode(11) == 1
    assert assembler.observe_context(
        0, 10, torch.tensor([1.0]), torch.tensor(2), torch.tensor(0.0), False
    ) is None
    transition = assembler.observe_context(
        0, 10, torch.tensor([2.0]), torch.tensor(3), torch.tensor(0.75), False)
    assert transition is not None
    assert float(transition.next_reward_ext) == 0.75
    assert torch.equal(transition.context, torch.tensor([1.0]))
    assert torch.equal(transition.next_context, torch.tensor([2.0]))
    online = ActionEmbeddingQ(4, hidden_size=8, action_embedding_size=4)
    target = ActionEmbeddingQ(4, hidden_size=8, action_embedding_size=4)
    terminal_target = double_q_target(
        torch.tensor([1.25]), torch.tensor([True]), torch.zeros(1, 4),
        online, target)
    assert torch.equal(terminal_target, torch.tensor([1.25]))
    signature = inspect.signature(double_q_target).parameters
    assert "reward_int" not in signature and "intrinsic" not in signature
    voi_signature = inspect.signature(cross_fitted_voi).parameters
    assert "next_obs" not in voi_signature and "reward" not in voi_signature
    return {
        "episode_fold_isolation": True,
        "temporal_alignment": True,
        "terminal_bootstrap_mask": True,
        "extrinsic_only_q_interface": True,
        "no_hindsight_voi_interface": True,
        "pass": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hot-tab", type=Path, required=True)
    parser.add_argument("--wvr-response", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    hot = load(args.hot_tab)
    response = load(args.wvr_response)
    gates = {
        "G1_action_conditioning": response["gates"]["G1_action_conditioning"],
        "G2_probability_calibration": response["gates"]["G2_probability_calibration"],
        "G3_oracle_voi": hot["gates"]["G3_oracle_voi"],
        "G4_zero_information": hot["gates"]["G4_zero_information"],
        "G5_noise_stability": hot["gates"]["G5_noise_stability"],
        "G6_mc_convergence": hot["gates"]["G6_mc_convergence"],
        "G7_q_alignment": check_g7(),
    }
    report = {
        "protocol": "DRVOI-RHPPO-V6-WVR-1",
        "scientific_result": False,
        "gate_sources": {
            "hot_tab": str(args.hot_tab),
            "wvr_response": str(args.wvr_response),
        },
        "gates": gates,
        "all_gates_pass": all(item["pass"] for item in gates.values()),
        "authorization": (
            "O0 short online smoke may start" if all(
                item["pass"] for item in gates.values())
            else "online DRVOI training remains blocked"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["all_gates_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
