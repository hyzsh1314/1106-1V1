#!/usr/bin/env python3
"""Audit MRAT B0 logs and compare its fixed-opponent results with V9/V11."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from benchmarks.drvoi_v12 import validate_update_metrics, write_json_atomic


OPPONENTS = ("hold", "pursuit", "evasive")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method-label", default="V12 MRAT B0")
    parser.add_argument("--stage-label", default="B0")
    parser.add_argument(
        "--extra-method", action="append", default=[], metavar="LABEL=PATH",
        help="Additional fixed-opponent run to include in the comparison.")
    parser.add_argument(
        "--evaluation-only", action="store_true",
        help="Analyze fixed-opponent outputs without a training metrics file.")
    return parser.parse_args()


def _load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    methods = {
        "V9 COACH-RHPPO": root / "runs/drvoi_rhppo_v9/coach-online/development-1m-seed85",
        "V11.2 episode-latched": root / "runs/drvoi_rhppo_v11_2/race-episode-latched/frozen-v9-seed85",
        "V11.3 sustained": root / "runs/drvoi_rhppo_v11_3/race-sustained/frozen-v9-seed85",
        args.method_label: args.run_dir,
    }
    for specification in args.extra_method:
        if "=" not in specification:
            raise ValueError("--extra-method must use LABEL=PATH")
        label, path_text = specification.split("=", 1)
        path = Path(path_text)
        methods[label] = path if path.is_absolute() else root / path
    rows = []
    for method, directory in methods.items():
        for opponent in OPPONENTS:
            report = _load(directory / f"eval-{opponent}.json")
            rows.append({
                "method": method,
                "opponent": opponent,
                "episodes": report["episodes"],
                "win_rate": report["win_rate"],
                "loss_rate": report["loss_rate"],
                "draw_rate": report["draw_rate"],
                "timeout_rate": report["timeout_rate"],
                "crash_rate": report["crash_rate"],
                "mean_episode_reward": report["mean_episode_reward"],
                "accepted_launches": report["accepted_launches"],
                "hits": report["hits"],
                "hit_per_launch": report["hit_per_launch"],
                "legal_request_rate": report["legal_request_rate"],
                "intercept_fraction": (
                    report.get("intercept_expert_steps", 0)
                    / max(report.get("learned_expert_steps", 0)
                          + report.get("intercept_expert_steps", 0), 1)),
                "forced_fraction": (
                    report.get("forced_intervention_steps", 0)
                    / max(report.get("environment_decisions_executed", 0), 1)),
                "expert_switches": report.get("expert_switches", 0),
                "mean_option_dwell": report.get("mean_option_dwell", 0.0),
                "option_segments": report.get("option_segments", 0),
                "option_boundaries": report.get("option_boundaries", 0),
                "same_expert_reselections": report.get(
                    "same_expert_reselections", 0),
            })

    metric_records = []
    metrics_path = args.run_dir / "metrics.jsonl"
    if not args.evaluation_only:
        with metrics_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                record = json.loads(line)
                schema = record.pop("metric_schema", None)
                validate_update_metrics(record)
                metric_records.append({
                    "line": line_number, "schema": schema, **record})
    if not metric_records and not args.evaluation_only:
        raise ValueError("B0 metrics.jsonl is empty")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "fixed-opponent-comparison.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    macro = {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        macro[method] = {
            "macro_win_rate": sum(row["win_rate"] for row in selected) / 3,
            "macro_timeout_rate": sum(row["timeout_rate"] for row in selected) / 3,
            "macro_crash_rate": sum(row["crash_rate"] for row in selected) / 3,
            "total_hits": sum(row["hits"] for row in selected),
            "total_launches": sum(row["accepted_launches"] for row in selected),
        }
    summary = {
        "protocol_version": f"MRAT-RHPPO-V12-{args.stage_label}-ANALYSIS-1",
        "scientific_result": False,
        "comparison_rows": rows,
        "macro": macro,
        "training_updates_validated": len(metric_records),
        "last_training_metrics": metric_records[-1] if metric_records else None,
        "interpretation_boundary": (
            f"{args.stage_label} is a development experiment. It cannot "
            "establish algorithmic superiority or replace the preregistered "
            "multi-seed comparison."),
    }
    write_json_atomic(args.output_dir / "summary.json", summary)
    lines = [
        f"# DRVOI-RHPPO V12 MRAT {args.stage_label} 审计报告",
        "",
        f"> {args.stage_label} 为开发实验，不构成正式多种子算法优越性证据。",
        "",
        "## 固定对手结果",
        "",
        "| 方法 | 对手 | 胜率 | 超时率 | 发射 | 命中 | LOS占比 | 强制占比 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['opponent']} | "
            f"{row['win_rate']:.3f} | {row['timeout_rate']:.3f} | "
            f"{row['accepted_launches']} | {row['hits']} | "
            f"{row['intercept_fraction']:.3f} | {row['forced_fraction']:.3f} |")
    lines.extend([
        "",
        "## 训练日志完整性",
        "",
        (f"`metrics.jsonl` 共 {len(metric_records)} 次更新，全部通过 V12 强制字段与有限值校验。"
         if metric_records else "本阶段仅执行结构诊断评估，不产生训练更新。"),
        "",
        "详细机器可读结果见 `summary.json` 与 `fixed-opponent-comparison.csv`。",
    ])
    (args.output_dir / "analysis-report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "rows": len(rows),
        "validated_updates": len(metric_records),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
