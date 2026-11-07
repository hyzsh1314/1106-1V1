#!/usr/bin/env python3
"""Train two seed-separated V8 option critics for online cross-evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from benchmarks.validate_drvoi_v8_macro_option_value import (
    load_split,
    sha256,
    target_keys,
    train_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, nargs=2, required=True)
    parser.add_argument("--validation", type=Path, nargs="+", required=True)
    parser.add_argument("--config", type=Path, default=Path(
        "configs/drvoi_rhppo_v8_macro_option.json"))
    parser.add_argument("--target", default="return_30")
    parser.add_argument("--representation", default="registered_v7_1")
    parser.add_argument("--seed", type=int, default=78185)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    config_hash = sha256(args.config)
    targets = target_keys(protocol)
    if args.target not in targets:
        raise ValueError(f"unknown target {args.target}")
    if args.representation != "registered_v7_1":
        raise ValueError("online cross-fitted Q must use registered_v7_1")
    train, train_seeds, _ = load_split(
        args.train, "train", config_hash, targets)
    validation, validation_seeds, _ = load_split(
        args.validation, "validation", config_hash, targets)
    if len(train_seeds) != 2:
        raise ValueError("cross-fitting requires exactly two collection seeds")
    device = torch.device("cuda")
    for data in (train, validation):
        for key in (args.representation, args.target, "legal",
                    "collection_seed"):
            data[key] = data[key].to(device)

    folds = []
    for fold, collection_seed in enumerate(train_seeds):
        selected = train["collection_seed"] == collection_seed
        report, artifact = train_model(
            train[args.representation][selected],
            validation[args.representation],
            {"return": train[args.target][selected],
             "legal": train["legal"][selected]},
            {"return": validation[args.target],
             "legal": validation["legal"]},
            protocol["critic"], protocol["metrics"],
            args.seed + fold * 1000)
        folds.append({
            "fold": fold,
            "collection_seed": collection_seed,
            "model": artifact["model"],
            "context_mean": artifact["mean"],
            "context_scale": artifact["scale"],
            "validation_metrics": report["best_validation_metrics"],
            "best_epoch": report["best_epoch"],
        })
        metrics = report["best_validation_metrics"]
        print(json.dumps({
            "fold": fold, "collection_seed": collection_seed,
            "pairwise_accuracy": metrics["pairwise_accuracy"],
            "weighted_normalized_argmax_regret": metrics[
                "weighted_normalized_argmax_regret"],
        }), flush=True)

    payload = {
        "suite": "DRVOI-RHPPO-V8-cross-fitted-option-Q",
        "scientific_result": False,
        "protocol_version": protocol["protocol_version"],
        "target": args.target,
        "representation": args.representation,
        "context_dim": int(train[args.representation].shape[-1]),
        "num_options": int(train["legal"].shape[-1]),
        "hidden_size": protocol["critic"]["hidden_size"],
        "train_seeds": train_seeds,
        "validation_seeds": validation_seeds,
        "folds": folds,
        "config_sha256": config_hash,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    args.output.with_suffix(".json").write_text(json.dumps({
        **{key: value for key, value in payload.items() if key != "folds"},
        "folds": [{
            key: value for key, value in fold.items()
            if key not in {"model", "context_mean", "context_scale"}}
            for fold in folds],
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
