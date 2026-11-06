#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/d/Ph.D/毕业/WVR-1102/WVR-Env"
python_bin="/home/hongyun0405/.venvs/bvr1022/bin/python"
root="runs/drvoi_rhppo_v7/b0-1m"

test -f "$root/COMPLETED"
PYTHONPATH=. "$python_bin" benchmarks/summarize_drvoi_v7_b0.py \
    --root "$root" --output "$root/summary.json" > "$root/summary.log" 2>&1
